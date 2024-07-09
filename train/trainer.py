import gc
import logging
import math
import os
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from torch.utils.data import default_collate
from torchvision.transforms import v2
import wandb
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from lion_pytorch import Lion
from packaging import version
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from kan_convs import KANConv2DLayer, KALNConv2DLayer, FastKANConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer, \
    WavKANConv2DLayer, BottleNeckKAGNConv2DLayer, MoEBottleNeckKAGNConv2DLayer
from .lbfgs import LBFGS
from .losses import Dice
from .metrics import get_metrics

logger = get_logger(__name__)


class OutputHook(list):
    """ Hook to capture module outputs.
    """

    def __call__(self, module, input, output):
        self.append(output)


def get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_model(model, dataset_train, dataset_val, loss_func, cfg, dataset_test=None, cam_reporter=None):
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
    wandb_params = {"entity": cfg.wandb.entity, }
    if "runname" in cfg.wandb:
        wandb_params = {"entity": cfg.wandb.entity, 'name': cfg.wandb.runname}

    accelerator.init_trackers(
        project_name=cfg.wandb.project_name,
        config=dict(cfg),
        init_kwargs={"wandb": wandb_params}
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
    elif cfg.optim.type == 'lbfgs':
        optimizer = LBFGS(params_to_optimize,
                          lr=cfg.optim.learning_rate,
                          max_iter=cfg.optim.max_iter,
                          tolerance_grad=1e-7,
                          tolerance_change=1e-9,
                          tolerance_ys=1e-32,
                          history_size=cfg.optim.history_size,
                          line_search_fn="strong_wolfe"
                          )

    cutmix = v2.CutMix(num_classes=cfg.model.num_classes)
    mixup = v2.MixUp(num_classes=cfg.model.num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
        collate_fn=lambda _batch: cutmix_or_mixup(*default_collate(_batch)) if cfg.use_mixup else default_collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    test_dataloader = None
    if dataset_test is not None:
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
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
                               KACNConv2DLayer, KAGNConv2DLayer, WavKANConv2DLayer,
                               BottleNeckKAGNConv2DLayer, MoEBottleNeckKAGNConv2DLayer)):
            module.register_forward_hook(output_hook)

    if cfg.metrics.report_type == 'classification' or cfg.metrics.report_type == 'classification_minimum':
        metric_acc = partial(accuracy, task="multiclass", top_k=1, num_classes=cfg.model.num_classes)
        metric_acc_top5 = partial(accuracy, task="multiclass", top_k=5, num_classes=cfg.model.num_classes)
        metric_acc, metric_acc_top5 = accelerator.prepare(metric_acc, metric_acc_top5)

    if cfg.metrics.report_type in ['segmentation', 'med_segmentation']:
        dice_metric = Dice()
        dice_metric = accelerator.prepare(dice_metric)

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler,
    )
    output_hook = accelerator.prepare(output_hook)
    if cfg.use_torch_compile:
        compiled_model = accelerator.prepare(compiled_model)
    if test_dataloader is not None:
        test_dataloader = accelerator.prepare(test_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.epochs * num_update_steps_per_epoch

    save_checkpoints = True
    if "save_checkpoints" in cfg:
        save_checkpoints = cfg.save_checkpoints
        if not save_checkpoints:
            logger.info(f"CHECKPOINT SAVING DISABLED")

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
    first_epoch = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    image_logs = None
    for epoch in range(first_epoch, cfg.epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):

            if cfg.optim.type == 'lbfgs':
                global additional_metrics, loss

                def closure():
                    global additional_metrics, loss
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

                    if cfg.metrics.report_type == 'classification' or cfg.metrics.report_type == 'classification_minimum':
                        if isinstance(output, tuple):
                            acc = metric_acc(output[-1], labels)
                            acc_t5 = metric_acc_top5(output[-1], labels)
                        else:
                            acc = metric_acc(output, labels)
                            acc_t5 = metric_acc_top5(output, labels)

                        additional_metrics = {"train_acc": acc.detach().item(),
                                              "train_acc_top5": acc_t5.detach().item()}
                    if cfg.metrics.report_type in ['segmentation', 'med_segmentation']:
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
                    # optimizer.step()
                    # optimizer.zero_grad(set_to_none=cfg.optim.set_grads_to_none)
                    output_hook.clear()
                    return loss

                optimizer.step(closure)
                lr_scheduler.step()
            else:

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

                if cfg.metrics.report_type == 'classification' or cfg.metrics.report_type == 'classification_minimum':
                    if isinstance(output, tuple):
                        acc = metric_acc(output[-1], labels)
                        acc_t5 = metric_acc_top5(output[-1], labels)
                    else:
                        acc = metric_acc(output, labels)
                        acc_t5 = metric_acc_top5(output, labels)

                    additional_metrics = {"train_acc": acc.detach().item(),
                                          "train_acc_top5": acc_t5.detach().item()}

                if cfg.metrics.report_type in ['segmentation', 'med_segmentation']:
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
                    if cfg.metrics.report_type == 'classification' or cfg.metrics.report_type == 'classification_minimum':
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

            wandb_tracker = accelerator.get_tracker("wandb")

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
            if cam_reporter is not None:

                logger.info(f"Running CAM Visualization")
                report = cam_reporter.create_report(deepcopy(accelerator.unwrap_model(model)))

                for key_layer, image_layer in report.items():
                    wandb_tracker.log({key_layer: [wandb.Image(image_layer), ]}, step=global_step)
                logger.info(f"CAM Visualization logged")
            if save_checkpoints:
                save_checkpoint_name = f"checkpoint-{epoch}-{cfg.tracking_metric}-{metrics[cfg.tracking_metric]}"

                save_path = os.path.join(cfg.output_dir, save_checkpoint_name)
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        gc.collect()
    # Create the pipeline using using the trained modules and save it.

    accelerator.wait_for_everyone()
    if test_dataloader is not None:

        accelerator.wait_for_everyone()

        predictions = []
        targets = []
        for step, batch in enumerate(test_dataloader):
            images, labels = batch
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            with torch.no_grad():
                predicts = model(images, train=False)
                if cfg.model.is_moe:
                    predicts, _ = predicts
                if isinstance(predicts, tuple):
                    predicts = predicts[-1]
                predicts = torch.softmax(predicts, dim=1)
                output_hook.clear()

            all_predictions, all_targets = accelerator.gather_for_metrics((predicts, labels))

            if accelerator.is_main_process:
                targets.append(all_targets.detach().cpu().numpy())
                predictions.append(all_predictions.detach().cpu().numpy())

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            targets = np.concatenate(targets, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            metrics = get_metrics(targets, predictions, cfg.metrics)

            output_metric_header = [k for k in metrics.keys()]
            values = [[metrics[k] for k in output_metric_header], ]

            if "extra_table_log" in cfg:
                for k, v in cfg.extra_table_log.items():
                    output_metric_header.append(k)
                    values[0].append(v)

            output_metric = (values, output_metric_header)

            del targets, predictions

            test_table = wandb.Table(data=output_metric[0], columns=output_metric[1])
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.log({"test_set_metrics": test_table})

    accelerator.wait_for_everyone()

    accelerator.end_training()

    return model
