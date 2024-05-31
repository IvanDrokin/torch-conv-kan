import gc
from functools import partial
import logging
import math
import os
import shutil
from pathlib import Path
from copy import deepcopy

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm
import wandb

from kan_convs import KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, WavKANConv2DLayer
from .metrics import get_metrics

logger = get_logger(__name__)


def eval_model(model, dataset_test, cfg):
    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.find_unused_parameters)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    accelerator.init_trackers(
        project_name=cfg.wandb.project_name,
        config=dict(cfg),
        init_kwargs={"wandb": {"entity": cfg.wandb.entity}}
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)


    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    predictions = []
    targets = []
    for step, batch in enumerate(test_dataloader):
        images, labels = batch
        # We could avoid this line since we set the accelerator with `device_placement=True`.
        with torch.no_grad():
            predicts = model(images, train=False)
            if isinstance(predicts, tuple):
                predicts, _ = predicts
            predicts = torch.softmax(predicts, dim=1)

        all_predictions, all_targets = accelerator.gather_for_metrics((predicts, labels))

        if accelerator.is_main_process:
            targets.append(all_targets.detach().cpu().numpy())
            predictions.append(all_predictions.detach().cpu().numpy())

    output_metric = None
    if accelerator.is_main_process:
        targets = np.concatenate(targets, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        metrics = get_metrics(targets, predictions, cfg.metrics)

        if output_metric is None:
            output_metric_header = [k for k in metrics.keys()]
            values = [[metrics[k] for k in output_metric_header], ]
            output_metric = (values, output_metric_header)
        else:
            output_metric[0].append([metrics[k] for k in output_metric[1]])
        del targets, predictions
        test_table = wandb.Table(data=output_metric[0], columns=output_metric[1])
        wandb_tracker = accelerator.get_tracker("wandb")
        wandb_tracker.log({"test_set_metrics": test_table})

    accelerator.end_training()

    return model
