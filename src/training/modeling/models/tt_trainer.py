import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.training.modeling.metrics import MR, minADE, minFDE
from src.training.modeling.metrics.prediction_avg_ade import PredAvgADE
from src.training.modeling.metrics.prediction_avg_fde import PredAvgFDE
from src.training.modeling.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        # metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, None, prefix)

        return losses["loss"] if self.training else 0.0

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:

        ignore_index = -100
        target_lane_logits = res["target_lane_logits"]
        target_lane = data["agent"]['agent_lane_id_target']
        agent_mask = data["agent"]["valid_mask"][:, :,  self.history_steps:]
        target_lane = target_lane.masked_fill(~agent_mask, ignore_index)

        test_target_lane = target_lane.clone().cpu().numpy()
        lane_cand_valid = data['agent']['lane_cand_valid'][:, :, self.history_steps:]
        lane_cand_mask = ~lane_cand_valid
        flat_target_lane = target_lane.reshape(-1)
        valid_target_mask = flat_target_lane != ignore_index
        if valid_target_mask.any():
            flat_lane_mask = lane_cand_mask.reshape(-1, lane_cand_mask.shape[-1])
            valid_targets = flat_target_lane[valid_target_mask]
            assert valid_targets.min() >= 0 and valid_targets.max() < lane_cand_mask.shape[-1], (
                "target_lane indices out of range for lane candidates"
            )
            masked_targets = flat_lane_mask[valid_target_mask].gather(1, valid_targets.unsqueeze(-1)).squeeze(-1)
            assert not masked_targets.any(), "lane_cand_mask must be False for ground-truth target_lane indices"

        B, N, T, K = target_lane_logits.shape

        # lane_cand_mask = lane_cand_mask.contiguous()
        # agent_mask = agent_mask.contiguous()
        # test_lane_cand_mask=lane_cand_mask.view(-1, K).cpu().numpy()
        # test_1 = target_lane_logits.clone().view(-1, K).detach().cpu().numpy()
        # target_lane_logits = target_lane_logits.masked_fill(lane_cand_mask, -1e6)
        agent_category = data['agent']['category']  # [B,N]
        category_mask = ((agent_category == 2) | (agent_category == 3)).unsqueeze(-1)  # [B,N,1]
        target_lane = target_lane.masked_fill(category_mask, ignore_index)

        loss = F.cross_entropy(
            target_lane_logits.view(-1, K),
            target_lane.view(-1),
            reduction='none',
            ignore_index=ignore_index
        )#.view(B, N, T)
        # test_target_lane_logits = target_lane_logits.view(-1, K).detach().cpu().numpy()
        # loss_max = (loss> 1e5).nonzero().cpu().numpy()
        # loss_max_item = loss_max[0]
        # test = target_lane.view(-1).cpu().numpy()
        # test_agent_mask = agent_mask.view(-1).cpu().numpy()
        # test_lane_cand_mask=lane_cand_mask.view(-1, K).cpu().numpy()
        valid = torch.ones_like(loss, dtype=torch.float, device=loss.device)
        loss = (loss * valid).sum() / (valid.sum().clamp_min(1.0))
        return {
            "loss": loss,
        }

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])

        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    def on_after_backward(self):
        # 查找未使用的参数
        unused = []
        for n, p in self.named_parameters():
            # 只关心需要训练的参数
            if p.requires_grad and p.grad is None:
                unused.append(n)
        if unused:
            self.print(f"[UNUSED PARAMS] {unused[:10]}{' ...' if len(unused) > 10 else ''}")
        # 计算并记录梯度范数
        grads = [
            param.grad.detach()
            for param in self.parameters()
            if param.grad is not None
        ]
        if grads:
            total_norm = torch.linalg.vector_norm(
                torch.stack([g.norm(2) for g in grads]), ord=2
            )
            self.log(
                "grad_norm_l2",
                total_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        if self.trainer.is_global_zero:
            unused = [
                name
                for name, param in self.named_parameters()
                if param.requires_grad and param.grad is None
            ]
            if unused:
                self.print("[unused this step]:\n" + "\n".join(unused))
