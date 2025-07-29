#!/usr/bin/env python
"""
Training script for FloeDock.
"""

import copy
import math
import os
import random
import resource
from functools import partial
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
from torch_geometric.nn import DataParallel

from datasets.pdbbind import construct_loader
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.parsing import parse_train_args
from utils.training import (
    train_epoch,
    test_epoch,
    loss_function,
    finetune_epoch,
    inference_epoch,
)
from utils.utils import (
    save_yaml_file,
    get_optimizer_and_scheduler,
    get_model,
    ExponentialMovingAverage,
)

# Increase file-descriptor limit
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

# Safe GPU count
gpus = list(range(torch.cuda.device_count()))
print("Available GPUs:", len(gpus))


def train(
    args,
    model,
    optimizer,
    scheduler,
    ema_weights,
    train_loader,
    val_loader,
    t_to_sigma,
    run_dir: Path,
    start_epoch: int,
):
    """Main training loop."""
    best_val_loss = math.inf
    best_inf_value = math.inf if args.inference_earlystop_goal == "min" else 0
    best_epoch = 0
    best_inf_epoch = 0

    loss_fn = partial(
        loss_function,
        lddt_weight=args.lddt_weight,
        affinity_weight=args.affinity_weight,
        tr_weight=args.tr_weight,
        rot_weight=args.rot_weight,
        tor_weight=args.tor_weight,
        res_tr_weight=args.res_tr_weight,
        res_rot_weight=args.res_rot_weight,
        res_chi_weight=args.res_chi_weight,
        no_torsion=args.no_torsion,
    )

    print("Starting training...")
    for epoch in range(start_epoch, args.n_epochs):
        logs = {}

        if not args.only_test:
            # ----- training -----
            train_losses = train_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                t_to_sigma,
                loss_fn,
                ema_weights,
            )
            print(
                f"Epoch {epoch}: Train loss {train_losses['loss']:.4f} "
                f"lddt {train_losses['lddt_loss']:.4f} "
                f"affinity {train_losses['affinity_loss']:.4f} "
                f"tr {train_losses['tr_loss']:.4f} "
                f"rot {train_losses['rot_loss']:.4f} "
                f"tor {train_losses['tor_loss']:.4f} "
                f"res_tr {train_losses['res_tr_loss']:.4f} "
                f"res_rot {train_losses['res_rot_loss']:.4f} "
                f"res_chi {train_losses['res_chi_loss']:.4f}"
            )

            # ----- optional finetune -----
            if (
                args.finetune_freq
                and (epoch + 1) % args.finetune_freq == 0
                and best_inf_value > 20.0
            ):
                idxs = np.random.choice(
                    len(train_loader.dataset),
                    size=args.num_finetune_complexes,
                    replace=False,
                )
                complexes = [train_loader.dataset[i] for i in idxs]
                ft_losses, _ = finetune_epoch(
                    model,
                    complexes,
                    args.device,
                    t_to_sigma,
                    args,
                    optimizer,
                    loss_fn,
                    ema_weights,
                )
                print(
                    f"Epoch {epoch}: Finetune loss {ft_losses['loss']:.4f} "
                    f"lddt {ft_losses['lddt_loss']:.4f} "
                    f"affinity {ft_losses['affinity_loss']:.4f}"
                )

            # ----- validation -----
            ema_weights.store(model.parameters())
            if args.use_ema:
                ema_weights.copy_to(model.parameters())

            val_losses = test_epoch(
                model,
                val_loader,
                args.device,
                t_to_sigma,
                loss_fn,
                args.test_sigma_intervals,
            )
            print(
                f"Epoch {epoch}: Val loss {val_losses['loss']:.4f} "
                f"lddt {val_losses['lddt_loss']:.4f} "
                f"affinity {val_losses['affinity_loss']:.4f}"
            )

            # ----- validation-inference -----
            if (
                args.val_inference_freq
                and (epoch + 1) % args.val_inference_freq == 0
            ):
                inf_metrics = inference_epoch(
                    model,
                    val_loader.dataset.complex_graphs[: args.num_inference_complexes],
                    args.device,
                    t_to_sigma,
                    args,
                )
                print(
                    f"Epoch {epoch}: ValInf "
                    f"lddt_rmse {inf_metrics['lddt_rmse']:.3f} "
                    f"affinity_rmse {inf_metrics['affinity_rmse']:.3f}"
                )
                logs.update({f"valinf_{k}": v for k, v in inf_metrics.items()})

            if not args.use_ema:
                ema_weights.copy_to(model.parameters())
            ema_state = copy.deepcopy(
                model.module.state_dict()
                if next(model.parameters()).is_cuda
                else model.state_dict()
            )
            ema_weights.restore(model.parameters())

        else:
            # inference-only mode
            inf_metrics = inference_epoch(
                model,
                val_loader.dataset.complex_graphs[: args.num_inference_complexes],
                args.device,
                t_to_sigma,
                args,
            )
            print(
                f"Epoch {epoch}: InfOnly "
                f"lddt_rmse {inf_metrics['lddt_rmse']:.3f} "
                f"affinity_rmse {inf_metrics['affinity_rmse']:.3f}"
            )
            assert False, "Inference-only test complete."

        # ----- logging -----
        if args.wandb:
            logs.update(
                {
                    **{f"train_{k}": v for k, v in train_losses.items()},
                    **{f"val_{k}": v for k, v in val_losses.items()},
                    "current_lr": optimizer.param_groups[0]["lr"],
                }
            )
            import wandb

            wandb.log(logs, step=epoch + 1)

        # ----- checkpointing -----
        state_dict = (
            model.module.state_dict()
            if next(model.parameters()).is_cuda
            else model.state_dict()
        )
        torch.save(
            {"epoch": epoch, "model": state_dict, "optimizer": optimizer.state_dict()},
            run_dir / "last_model.pt",
        )

        # best validation loss
        if val_losses["loss"] <= best_val_loss:
            best_val_loss = val_losses["loss"]
            best_epoch = epoch
            torch.save(state_dict, run_dir / "best_model.pt")
            torch.save(ema_state, run_dir / "best_ema_model.pt")

        # best inference metric
        if args.val_inference_freq and (
            args.inference_earlystop_goal == "min"
            and inf_metrics[args.inference_earlystop_metric] <= best_inf_value
            or args.inference_earlystop_goal == "max"
            and inf_metrics[args.inference_earlystop_metric] >= best_inf_value
        ):
            best_inf_value = inf_metrics[args.inference_earlystop_metric]
            best_inf_epoch = epoch
            torch.save(state_dict, run_dir / "best_inference_epoch_model.pt")
            torch.save(ema_state, run_dir / "best_ema_inference_epoch_model.pt")

        # scheduler step
        if scheduler:
            if args.val_inference_freq:
                scheduler.step(best_inf_value)
            else:
                scheduler.step(val_losses["loss"])

    print(f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best inf metric {best_inf_value:.4f} at epoch {best_inf_epoch}")


def load_config(args_dict: Dict[str, Any], yaml_path: Path) -> Dict[str, Any]:
    """Merge YAML config into CLI args."""
    yaml_cfg = yaml.safe_load(yaml_path.read_text())
    for k, v in yaml_cfg.items():
        if isinstance(v, list) and k in args_dict:
            args_dict[k].extend(v)
        else:
            args_dict[k] = v
    return args_dict


def main():
    args = parse_train_args()
    if args.config:
        load_config(vars(args), Path(args.config))

    assert args.inference_earlystop_goal in {"min", "max"}
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # Ensure CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Build loaders
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    train_loader, val_loader = construct_loader(args, t_to_sigma)

    # Build model & optimizer
    model = get_model(args, device, t_to_sigma=t_to_sigma)
    optimizer, scheduler = get_optimizer_and_scheduler(
        args,
        model,
        scheduler_mode=(
            args.inference_earlystop_goal
            if args.val_inference_freq
            else "min"
        ),
    )
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    # Resume checkpoint
    start_epoch = 0
    if args.restart_dir:
        ckpt_path = Path(args.restart_dir) / "last_model.pt"
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            optimizer.load_state_dict(ckpt["optimizer"])
            model.load_state_dict(ckpt["model"], strict=True)
            if hasattr(args, "ema_rate"):
                ema_weights.load_state_dict(ckpt["ema_weights"], device=device)
            start_epoch = ckpt["epoch"] + 1
            print(f"Resume from epoch {start_epoch}")
        except Exception as e:
            print("Resume failed:", e)
            model.load_state_dict(
                torch.load(Path(args.restart_dir) / "best_model.pt", map_location="cpu"),
                strict=True,
            )
            print("Loaded best_model.pt instead")

    # W&B
    if args.wandb:
        import wandb

        wandb.init(
            entity="entity",
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=vars(args),
        )
        wandb.log({"numel": sum(p.numel() for p in model.parameters())})

    # Prepare run directory
    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save source code snapshot
    if not args.only_test:
        for folder in ["datasets", "models", "utils"]:
            shutil.copytree(folder, run_dir / folder, dirs_exist_ok=True)
        for py_file in Path.cwd().glob("*.py"):
            shutil.copy2(py_file, run_dir)
        save_yaml_file(run_dir / "model_parameters.yml", vars(args))

    # Launch training
    train(
        args,
        model,
        optimizer,
        scheduler,
        ema_weights,
        train_loader,
        val_loader,
        t_to_sigma,
        run_dir,
        start_epoch,
    )


if __name__ == "__main__":
    main()
