import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class EWCLoRANet(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.image_encoder = backbone
        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]
        self.feature_dim = kwargs.get("embd_dim", backbone.out_dim)

        self.classifier_pool = nn.ModuleList()
        for task_id in range(self.task_num):
            out_dim = self.init_cls_num if task_id == 0 else self.inc_cls_num
            self.classifier_pool.append(nn.Linear(self.feature_dim, out_dim, bias=True))

        self._cur_task = -1

    def update_fc(self):
        self._cur_task += 1

    def forward(self, image, use_new=True, register_hook=False):
        features = self.image_encoder(image, use_new=use_new, register_hook=register_hook)
        logits = self.classifier_pool[self._cur_task](features)
        return {"logits": logits, "features": features}

    def interface(self, image, use_new=True):
        features = self.image_encoder(image, use_new=use_new)
        logits = [self.classifier_pool[task_id](features) for task_id in range(self._cur_task + 1)]
        return torch.cat(logits, dim=1)

    def accumulate_and_reset_lora(self):
        self.image_encoder.accumulate_and_reset_lora()


class EWCLoRA(nn.Module):
    def __init__(self, backbone, device, **kwargs):
        super().__init__()
        self.device = device
        self.kwargs = kwargs

        self.init_cls_num = kwargs["init_cls_num"]
        self.inc_cls_num = kwargs["inc_cls_num"]
        self.task_num = kwargs["task_num"]

        self.gamma = kwargs["gamma"]
        self.ewc_lambda = kwargs["lambda"]
        self.fc_lrate = kwargs.get("fc_lrate", 5e-3)
        self.fisher_max_batches = kwargs.get("fisher_max_batches", None)

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._pending_accumulation = False

        self.network = EWCLoRANet(backbone, **kwargs)
        self.omega_W = []
        self.count_updates = 0

    def _iter_lora_modules(self):
        for module in self.network.image_encoder.modules():
            if hasattr(module, "lora_new_A_k") and hasattr(module, "lora_new_B_k"):
                yield module

    def freeze_network(self):
        classifier_key = f"classifier_pool.{self._cur_task}"
        trainable_keys = (
            classifier_key,
            "lora_new_A_k",
            "lora_new_B_k",
            "lora_new_A_v",
            "lora_new_B_v",
        )
        for name, param in self.network.named_parameters():
            param.requires_grad_(any(key in name for key in trainable_keys))

    def before_task(self, task_idx, buffer, train_loader, test_loaders):
        if self._pending_accumulation:
            self.network.accumulate_and_reset_lora()
            self._pending_accumulation = False

        self._cur_task = task_idx
        task_size = self.init_cls_num if task_idx == 0 else self.inc_cls_num
        self._total_classes = self._known_classes + task_size

        self.network.update_fc()
        self.freeze_network()
        self.network = self.network.to(self.device)

    def observe(self, data):
        x = data["image"].to(self.device)
        y = data["label"].to(self.device)

        logits = self.network(x, use_new=True)["logits"]
        local_targets = y - self._known_classes

        loss = F.cross_entropy(logits, local_targets)
        if self.count_updates > 0:
            loss = loss + 0.5 * self.ewc_lambda * self.compute_ewc_penalty()

        preds = logits.argmax(dim=1) + self._known_classes
        acc = preds.eq(y).float().mean().item()
        return preds, acc, loss

    def inference(self, data):
        x = data["image"].to(self.device)
        y = data["label"].to(self.device)

        logits = self.network.interface(x, use_new=True)
        preds = logits.argmax(dim=1)
        acc = preds.eq(y).float().mean().item()
        return preds, acc

    def after_task(self, task_idx, buffer, train_loader, test_loaders):
        fisher = FisherComputer(
            model=self.network,
            dataloader=train_loader,
            task_start=self._known_classes,
            device=self.device,
        ).compute(max_batches=self.fisher_max_batches)

        if len(self.omega_W) == 0:
            self.omega_W = fisher
        else:
            self.omega_W = [
                self.gamma * old_fisher + new_fisher
                for old_fisher, new_fisher in zip(self.omega_W, fisher)
            ]

        self.count_updates += 1
        self._known_classes = self._total_classes
        self._pending_accumulation = True

    def compute_ewc_penalty(self):
        if len(self.omega_W) == 0:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        idx = 0
        for module in self._iter_lora_modules():
            delta_w_k = module.lora_new_B_k.weight @ module.lora_new_A_k.weight
            penalty = penalty + (self.omega_W[idx].to(self.device) * delta_w_k.pow(2)).sum()
            idx += 1

            delta_w_v = module.lora_new_B_v.weight @ module.lora_new_A_v.weight
            penalty = penalty + (self.omega_W[idx].to(self.device) * delta_w_v.pow(2)).sum()
            idx += 1

        return penalty

    def get_parameters(self, config):
        optim_cfg = config["optimizer"]["kwargs"]
        encoder_lr = optim_cfg.get("lr", 5e-4)
        weight_decay = optim_cfg.get("weight_decay", 0.0)

        encoder_params = [p for p in self.network.image_encoder.parameters() if p.requires_grad]
        classifier_params = [p for p in self.network.classifier_pool.parameters() if p.requires_grad]

        param_groups = []
        if len(encoder_params) > 0:
            param_groups.append(
                {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay}
            )
        if len(classifier_params) > 0:
            param_groups.append(
                {"params": classifier_params, "lr": self.fc_lrate, "weight_decay": weight_decay}
            )
        return param_groups


class FisherComputer:
    def __init__(self, model, dataloader, task_start, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.task_start = task_start
        self.device = device
        self.fisher_W = self._init_fisher_storage()

    def _iter_lora_modules(self):
        for module in self.model.image_encoder.modules():
            if hasattr(module, "lora_new_A_k") and hasattr(module, "lora_new_B_k"):
                yield module

    def _init_fisher_storage(self):
        fisher = []
        for module in self._iter_lora_modules():
            fisher.append(torch.zeros_like(module.lora_new_B_k.weight @ module.lora_new_A_k.weight))
            fisher.append(torch.zeros_like(module.lora_new_B_v.weight @ module.lora_new_A_v.weight))
        return fisher

    def compute(self, max_batches=None):
        self.model.eval()
        num_samples = 0

        for batch_id, batch in enumerate(tqdm(self.dataloader, desc="Computing Fisher", leave=False)):
            if max_batches is not None and batch_id >= max_batches:
                break

            for module in self._iter_lora_modules():
                module.delta_w_k_new_grad = None
                module.delta_w_v_new_grad = None

            inputs = batch["image"].to(self.device)
            targets = batch["label"].to(self.device) - self.task_start

            self.model.zero_grad()
            logits = self.model(inputs, use_new=True, register_hook=True)["logits"]
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            batch_size = inputs.size(0)
            num_samples += batch_size

            fisher_idx = 0
            for module in self._iter_lora_modules():
                if module.delta_w_k_new_grad is not None:
                    self.fisher_W[fisher_idx] += module.delta_w_k_new_grad.detach().pow(2) * batch_size
                fisher_idx += 1

                if module.delta_w_v_new_grad is not None:
                    self.fisher_W[fisher_idx] += module.delta_w_v_new_grad.detach().pow(2) * batch_size
                fisher_idx += 1

        if num_samples == 0:
            return self.fisher_W
        return [fisher / num_samples for fisher in self.fisher_W]
