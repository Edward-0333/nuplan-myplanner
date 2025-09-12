import logging
from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import pytorch_lightning.loggers
import pytorch_lightning.plugins
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.planning.script.builders.data_augmentation_builder import build_agent_augmentor
from nuplan.planning.script.builders.objectives_builder import build_objectives
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.splitter_builder import build_splitter
from nuplan.planning.script.builders.training_callback_builder import build_callbacks
from nuplan.planning.script.builders.training_metrics_builder import build_training_metrics
from nuplan.planning.script.builders.utils.utils_checkpoint import extract_last_checkpoint_from_experiment
from nuplan.planning.training.data_loader.datamodule import DataModule
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger

import os
from src.training.dataloader.datamoudle import CustomDataModule

logger = logging.getLogger(__name__)


def build_lightning_datamodule(
    cfg: DictConfig, worker: WorkerPool, model: TorchModuleWrapper
) -> pl.LightningDataModule:
    """
    Build the lightning datamodule from the config.
    :param cfg: Omegaconf dictionary.
    :param model: NN model used for training.
    :param worker: Worker to submit tasks which can be executed in parallel.
    :return: Instantiated datamodule object.
    """
    # Build features and targets
    feature_builders = model.get_list_of_required_feature()
    target_builders = model.get_list_of_computed_target()

    # Build splitter
    splitter = build_splitter(cfg.splitter)

    # Create feature preprocessor
    feature_preprocessor = FeaturePreprocessor(
        cache_path=cfg.cache.cache_path,
        force_feature_computation=cfg.cache.force_feature_computation,
        feature_builders=feature_builders,
        target_builders=target_builders,
    )

    # Create data augmentation
    augmentors = (
        build_agent_augmentor(cfg.data_augmentation)
        if "data_augmentation" in cfg
        else None
    )

    # Build dataset scenarios
    scenarios = build_scenarios(cfg, worker, model)

    # Create datamodule
    datamodule: pl.LightningDataModule = CustomDataModule(
        feature_preprocessor=feature_preprocessor,
        splitter=splitter,
        all_scenarios=scenarios,
        dataloader_params=cfg.data_loader.params,
        augmentors=augmentors,
        worker=worker,
        scenario_type_sampling_weights=cfg.scenario_type_weights.scenario_type_sampling_weights,
        **cfg.data_loader.datamodule,
    )

    return datamodule


def build_lightning_module(cfg: DictConfig, torch_module_wrapper: TorchModuleWrapper) -> pl.LightningModule:
    """
    Builds the lightning module from the config.
    :param cfg: omegaconf dictionary
    :param torch_module_wrapper: NN model used for training
    :return: built object.
    """
    # Create the complete Module
    if "trainer" in cfg:
        model = instantiate(
            cfg.trainer,
            model=torch_module_wrapper,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
        )
    else:
        # Build loss
        objectives = build_objectives(cfg)

        # Build metrics to evaluate the performance of predictions
        metrics = build_training_metrics(cfg)

        # Create the complete Module
        model = LightningModuleWrapper(
            model=torch_module_wrapper,
            objectives=objectives,
            metrics=metrics,
            batch_size=cfg.data_loader.params.batch_size,
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler if 'lr_scheduler' in cfg else None,
            warm_up_lr_scheduler=cfg.warm_up_lr_scheduler if 'warm_up_lr_scheduler' in cfg else None,
            objective_aggregate_mode=cfg.objective_aggregate_mode,
        )

    return cast(pl.LightningModule, model)


def build_trainer_tt(cfg: DictConfig) -> pl.Trainer:

    # Todo: 加上wandb
    """
    Builds the lightning trainer from the config.
    :param cfg: omegaconf dictionary
    :return: built object.
    """
    params = cfg.lightning.trainer.params

    # callbacks = build_callbacks(cfg)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints"),
            filename="{epoch}-{val_minFDE:.3f}",
            monitor=cfg.lightning.trainer.checkpoint.monitor,
            mode=cfg.lightning.trainer.checkpoint.mode,
            save_top_k=cfg.lightning.trainer.checkpoint.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        # RichProgressBar(),
        # LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.wandb.mode == "disable":
        training_logger = TensorBoardLogger(
            save_dir=cfg.group,
            name=cfg.experiment,
            log_graph=False,
            version="",
            prefix="",
        )
    else:
        if cfg.wandb.artifact is not None:
            os.system(f"wandb artifact get {cfg.wandb.artifact}")
            _, _, artifact = cfg.wandb.artifact.split("/")
            checkpoint = os.path.join(os.getcwd(), f"artifacts/{artifact}/model.ckpt")
            run_id = artifact.split(":")[0][-8:]
            cfg.checkpoint = checkpoint
            cfg.wandb.run_id = run_id

        training_logger = WandbLogger(
            save_dir=cfg.group,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            log_model=cfg.wandb.log_model,
            resume=cfg.checkpoint is not None,
            id=cfg.wandb.run_id,
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=training_logger,
        **params,
    )
    return trainer
