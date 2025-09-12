import logging
import math
import os
from pathlib import Path
from shutil import rmtree
from typing import cast

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.optim.lr_scheduler import OneCycleLR

from nuplan.planning.script.builders.utils.utils_type import is_target_type
from nuplan.planning.simulation.callback.timing_callback import TimingCallback
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool

logger = logging.getLogger(__name__)


def update_config_for_training(cfg: DictConfig) -> None:
    """
    Updates the config based on some conditions.
    :param cfg: omegaconf dictionary that is used to run the experiment.
    """
    # Make the configuration editable.
    OmegaConf.set_struct(cfg, False)

    if cfg.cache.cache_path is None:
        logger.warning("Parameter cache_path is not set, caching is disabled")
    else:
        if not str(cfg.cache.cache_path).startswith("s3://"):
            if cfg.cache.cleanup_cache and Path(cfg.cache.cache_path).exists():
                rmtree(cfg.cache.cache_path)

            Path(cfg.cache.cache_path).mkdir(parents=True, exist_ok=True)

    if cfg.lightning.trainer.overfitting.enable:
        cfg.data_loader.params.num_workers = 0

    OmegaConf.resolve(cfg)

    # Finalize the configuration and make it non-editable.
    OmegaConf.set_struct(cfg, True)

    # Log the final configuration after all overrides, interpolations and updates.
    if cfg.log_config:
        logger.info(
            f"Creating experiment name [{cfg.experiment}] in group [{cfg.group}] with config..."
        )
        logger.info("\n" + OmegaConf.to_yaml(cfg))