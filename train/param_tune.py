import gc
import logging
import math
import os
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import accelerate
import numpy as np
import ray.cloudpickle as pickle
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from lion_pytorch import Lion
from omegaconf import OmegaConf, open_dict
from packaging import version
from ray import train
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import random_split
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from kan_convs import KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, \
    WavKANConv2DLayer
from .losses import Dice
from .metrics import get_metrics

logger = get_logger(__name__)

from .trainer import get_polynomial_decay_schedule_with_warmup, OutputHook


def train_model(model, dataset_train, dataset_val, loss_func, cfg):
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

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    #
                    sub_dir = cfg.model_name
                    torch.save(model.state_dict(), os.path.join(output_dir, sub_dir))
                    i -= 1

        accelerator.register_save_state_pre_hook(save_model_hook)

    model.train()

    if cfg.use_torch_compile:
        compiled_model = torch.compile(model, mode="max-autotune", dynamic=False)
        torch.set_float32_matmul_precision('high')
        params_to_optimize = compiled_model.parameters()
    else:
        compiled_model = None
        params_to_optimize = model.parameters()

    if cfg.optim.type == 'adamW':
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            lr=cfg.optim.learning_rate,
            betas=(cfg.optim.adam_beta1, cfg.optim.adam_beta2),
            weight_decay=cfg.optim.adam_weight_decay,
            eps=cfg.optim.adam_epsilon,
        )
    elif cfg.optim.type == 'lion':
        optimizer = Lion(params_to_optimize,
                         lr=cfg.optim.learning_rate,
                         weight_decay=cfg.optim.learning_rate,
                         use_triton=cfg.optim.use_triton)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            model.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        cfg.optim.lr_warmup_steps * accelerator.num_processes,
        cfg.max_train_steps * accelerator.num_processes,
        lr_end=cfg.optim.lr_end,
        power=cfg.optim.lr_power,
        last_epoch=-1
    )

    # Prepare everything with our `accelerator`.
    output_hook = OutputHook()
    for module in model.modules():
        if isinstance(module, (KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer,
                               KACNConv2DLayer, KAGNConv2DLayer, WavKANConv2DLayer)):
            module.register_forward_hook(output_hook)

    if cfg.metrics.report_type == 'classification':
        metric_acc = partial(accuracy, task="multiclass", top_k=1, num_classes=cfg.model.num_classes)
        metric_acc_top5 = partial(accuracy, task="multiclass", top_k=5, num_classes=cfg.model.num_classes)
        metric_acc, metric_acc_top5 = accelerator.prepare(metric_acc, metric_acc_top5)
    if cfg.metrics.report_type == 'segmentation':
        dice_metric = Dice()
        dice_metric = accelerator.prepare(dice_metric)

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler,
    )
    output_hook = accelerator.prepare(output_hook)
    if cfg.use_torch_compile:
        compiled_model = accelerator.prepare(compiled_model)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.epochs * num_update_steps_per_epoch

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset_train)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = start_epoch
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for epoch in range(first_epoch, cfg.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                # Convert images to latent space
                images, labels = batch

                if cfg.use_torch_compile:
                    output = compiled_model(images)
                else:
                    output = model(images)
                if cfg.model.is_moe:
                    output, moe_loss = output
                else:
                    moe_loss = None

                l2_penalty = 0.
                l1_penalty = 0.
                for _output in output_hook:
                    if cfg.model.l1_activation_penalty > 0:
                        l1_penalty += torch.norm(_output, 1, dim=0).mean()
                    if cfg.model.l2_activation_penalty > 0:
                        l2_penalty += torch.norm(_output, 2, dim=0).mean()
                l2_penalty *= cfg.model.l2_activation_penalty
                l1_penalty *= cfg.model.l2_activation_penalty

                if isinstance(output, tuple):
                    loss = 0.
                    for _output in output:
                        loss = loss + loss_func(_output, labels)
                else:
                    loss = loss_func(output, labels)
                loss = loss + l1_penalty + l2_penalty
                if moe_loss is not None:
                    loss += moe_loss

                if cfg.metrics.report_type == 'classification':
                    if isinstance(output, tuple):
                        acc = metric_acc(output[-1], labels)
                        acc_t5 = metric_acc_top5(output[-1], labels)
                    else:
                        acc = metric_acc(output, labels)
                        acc_t5 = metric_acc_top5(output, labels)

                    additional_metrics = {"train_acc": acc.detach().item(),
                                          "train_acc_top5": acc_t5.detach().item()}
                if cfg.metrics.report_type == 'segmentation':
                    if isinstance(output, tuple):
                        model_out = output[0]
                    else:
                        model_out = output
                    dice_val = dice_metric(model_out, labels)
                    additional_metrics = {"train_dice": dice_val.detach().item(), }

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if cfg.use_torch_compile:
                        params_to_clip = compiled_model.parameters()
                    else:
                        params_to_clip = model.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.optim.set_grads_to_none)
                output_hook.clear()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs.update(additional_metrics)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.max_train_steps:
                break
        gc.collect()
        model.eval()
        predictions = []
        targets = []
        for step, batch in enumerate(val_dataloader):
            images, labels = batch
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            with torch.no_grad():
                predicts = model(images, train=False)
                if cfg.model.is_moe:
                    predicts, _ = predicts
                if isinstance(predicts, tuple):
                    if cfg.metrics.report_type == 'classification':
                        predicts = predicts[-1]
                    else:
                        predicts = predicts[0]
                if cfg.model.num_classes > 1:
                    predicts = torch.softmax(predicts, dim=1)
                else:
                    predicts = torch.sigmoid(predicts)

                output_hook.clear()

            all_predictions, all_targets = accelerator.gather_for_metrics((predicts, labels))

            if accelerator.is_main_process:
                targets.append(all_targets.detach().cpu().numpy())
                predictions.append(all_predictions.detach().cpu().numpy())

        if accelerator.is_main_process:

            targets = np.concatenate(targets, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            metrics = get_metrics(targets, predictions, cfg.metrics)
            accelerator.log(metrics, step=global_step)

            del targets, predictions

            if cfg.checkpoints_total_limit is not None:
                checkpoints = os.listdir(cfg.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= cfg.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(cfg.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": accelerator.unwrap_model(optimizer).state_dict(),
            }
            if cfg.raytune.save_checkpoints and epoch % cfg.raytune.save_every_n_epochs == 0:
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
            else:
                checkpoint = None
            train.report(
                {cfg.tracking_metric: metrics[cfg.tracking_metric]},
                checkpoint=checkpoint,
            )

        gc.collect()

    accelerator.wait_for_everyone()
    accelerator.end_training()


def ray_tune_train_model_wrapper(search_cfg, model, data_function, loss_func, original_cfg):
    cfg = deepcopy(original_cfg)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        if 'epochs' in search_cfg:
            cfg.epochs = search_cfg['epochs']
        if 'batch_size' in search_cfg:
            cfg.train_batch_size = search_cfg['batch_size']
            cfg.val_batch_size = search_cfg['batch_size']
        for check_keys in ['groups', 'degree', 'width_scale', 'dropout',
                           'dropout_linear', 'l1_decay', 'l2_activation_penalty',
                           'l1_activation_penalty', 'num_init_features', 'growth_rate',
                           'dropout_poly', 'dropout_degree', 'dropout_full', 'drop_type']:
            if check_keys in search_cfg:
                cfg['model'][check_keys] = search_cfg[check_keys]
        for check_keys in ['learning_rate', 'adam_beta1', 'adam_beta2', 'adam_weight_decay',
                           'adam_epsilon', 'lr_warmup_steps', 'lr_power',
                           'lr_end']:
            if check_keys in search_cfg:
                cfg['optim'][check_keys] = search_cfg[check_keys]

    data = data_function(cfg)

    if 'val' in data:
        train_data, val_data = data['train'], data['val']
    else:
        test_abs = int(len(data['train']) * 0.8)
        train_data, val_data = random_split(
            data['train'], [test_abs, len(data['train']) - test_abs]
        )

    loss_func = loss_func(search_cfg)
    train_model(model(cfg), train_data, val_data, loss_func, cfg)


def tune_params(search_config, model, config, data_function, loss_func, num_samples=10, max_num_epochs=10,
                gpus_per_trial=2, cpus_per_trial=2):
    """

    :param search_config: dict
    {
        "l1": tune.choice([2**i for i in range(9)]),
        "l2": tune.choice([2**i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    :param model:
    :param config:
    :param num_samples:
    :param data_function:
    :param loss_func:
    :param max_num_epochs:
    :param gpus_per_trial:
    :return:
    """
    if config.raytune.optuna:

        algo = OptunaSearch(
            search_config,
            metric=config.raytune.metric,
            mode=config.raytune.mode,
        )
        result = tune.run(
            partial(ray_tune_train_model_wrapper, model=model, data_function=data_function,
                    loss_func=loss_func, original_cfg=config),
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            num_samples=num_samples,
            search_alg=algo
        )
    else:

        scheduler = ASHAScheduler(
            metric=config.raytune.metric,
            mode=config.raytune.mode,
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        result = tune.run(
            partial(ray_tune_train_model_wrapper, model=model, data_function=data_function,
                    loss_func=loss_func, original_cfg=config),
            resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
            config=search_config,
            num_samples=num_samples,
            scheduler=scheduler,
        )

    best_trial = result.get_best_trial(config.raytune.metric, config.raytune.mode, "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result[config.raytune.metric]}")
    return best_trial.config
