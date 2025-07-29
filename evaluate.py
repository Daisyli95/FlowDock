#!/usr/bin/env python
"""
Inference pipeline for FlowDock (score + optional confidence model).
"""

import copy
import os
import pickle
import resource
import time
from argparse import ArgumentParser, FileType, Namespace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from datasets.moad import MOAD
from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule, t_to_sigma as t2s
from utils.gnina_utils import get_gnina_poses
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.utils import ExponentialMovingAverage, get_model
from utils.visualise import PDBFile

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------
# Resource limits
# ----------------------------------------------------------------------
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))
torch.multiprocessing.set_sharing_strategy("file_system")

# ----------------------------------------------------------------------
# Helper: build dataset
# ----------------------------------------------------------------------
def build_dataset(args, model_args, confidence=False):
    kwargs = dict(
        transform=None,
        root=args.data_dir,
        limit_complexes=args.limit_complexes,
        chain_cutoff=args.chain_cutoff,
        receptor_radius=model_args.receptor_radius,
        cache_path=args.cache_path,
        remove_hs=model_args.remove_hs,
        max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        matching=not model_args.no_torsion,
        keep_original=True,
        popsize=args.matching_popsize,
        maxiter=args.matching_maxiter,
        all_atoms=getattr(model_args, "all_atoms", False),
        atom_radius=getattr(model_args, "atom_radius", None),
        atom_max_neighbors=getattr(model_args, "atom_max_neighbors", None),
        esm_embeddings_path=args.esm_embeddings_path,
        require_ligand=True,
        num_workers=args.num_workers,
        knn_only_graph=not getattr(args, "not_knn_only_graph", False),
        include_miscellaneous_atoms=getattr(args, "include_miscellaneous_atoms", False),
        num_conformers=args.samples_per_complex if args.resample_rdkit and not confidence else 1,
    )

    if args.dataset != "moad":
        kwargs.update(
            dataset=args.dataset,
            split_path=args.split_path,
            protein_file=args.protein_file,
            ligand_file=args.ligand_file,
        )
        return PDBBind(**kwargs)
    else:
        kwargs.update(
            split=args.split,
            esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
            unroll_clusters=args.unroll_clusters,
            remove_pdbbind=args.remove_pdbbind,
            min_ligand_size=args.min_ligand_size,
            max_receptor_size=args.max_receptor_size,
            remove_promiscuous_targets=args.remove_promiscuous_targets,
            no_randomness=True,
            skip_matching=args.skip_matching,
        )
        return MOAD(**kwargs)

# ----------------------------------------------------------------------
# Helper: load model checkpoint
# ----------------------------------------------------------------------
def load_model_checkpoint(model_dir, ckpt_file, model_args, device, old=False):
    model = get_model(model_args, device, t_to_sigma=t2s(model_args), no_parallel=True, old=old)
    ckpt_path = Path(model_dir) / ckpt_file
    state = torch.load(ckpt_path, map_location="cpu")
    if ckpt_file == "last_model.pt":
        model.load_state_dict(state["model"], strict=True)
        ema = ExponentialMovingAverage(model.parameters(), decay=model_args.ema_rate)
        ema.load_state_dict(state["ema_weights"], device=device)
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict(state, strict=True)
    return model.eval()

# ----------------------------------------------------------------------
# Helper: override missing attributes in Namespace (legacy checkpoints)
# ----------------------------------------------------------------------
def fix_legacy_args(args_ns):
    defaults = dict(
        separate_noise_schedule=False,
        lm_embeddings_path=None,
        tr_only_confidence=True,
        high_confidence_threshold=0.0,
        include_confidence_prediction=False,
        confidence_weight=1,
        asyncronous_noise_schedule=False,
        correct_torsion_sigmas=False,
        esm_embeddings_path=None,
    )
    for k, v in defaults.items():
        if not hasattr(args_ns, k):
            setattr(args_ns, k, v)
    return args_ns

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    p = ArgumentParser()
    p.add_argument("--config", type=FileType("r"))
    # Model paths
    p.add_argument("--model_dir", default="workdir/test_score")
    p.add_argument("--ckpt", default="best_ema_inference_epoch_model.pt")
    p.add_argument("--confidence_model_dir", type=str)
    p.add_argument("--confidence_ckpt", default="best_model_epoch75.pt")
    # I/O
    p.add_argument("--run_name", default="test")
    p.add_argument("--project", default="ligbind_inf")
    p.add_argument("--out_dir", type=str)
    p.add_argument("--cache_path", default="data/cache")
    p.add_argument("--data_dir", default="../../ligbind/data/BindingMOAD_2020_ab_processed_biounit/")
    p.add_argument("--split_path", default="data/BindingMOAD_2020_ab_processed/splits/val.txt")
    # Sampling
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--inference_steps", type=int, default=40)
    p.add_argument("--samples_per_complex", type=int, default=4)
    p.add_argument("--no_model", action="store_true")
    p.add_argument("--no_random", action="store_true")
    p.add_argument("--no_final_step_noise", action="store_true")
    p.add_argument("--ode", action="store_true")
    # Dataset
    p.add_argument("--dataset", choices=["moad", "pdbbind"], default="moad")
    p.add_argument("--split", default="val")
    p.add_argument("--limit_complexes", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--resample_rdkit", action="store_true")
    p.add_argument("--skip_matching", action="store_true")
    # GNINA
    p.add_argument("--gnina_minimize", action="store_true")
    p.add_argument("--gnina_poses_to_optimize", type=int, default=1)
    # W&B
    p.add_argument("--wandb", action="store_true")
    # Miscellaneous (add more as needed)
    args, _ = p.parse_known_args()
    return args

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Load YAML config if provided
    if args.config:
        cfg = yaml.safe_load(args.config)
        for k, v in cfg.items():
            if isinstance(v, list) and hasattr(args, k):
                getattr(args, k).extend(v)
            else:
                setattr(args, k, v)

    # Default output directory
    if not args.out_dir:
        args.out_dir = f"inference_out_dir_not_specified/{args.run_name}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Fix legacy attributes
    with open(Path(args.model_dir) / "model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.safe_load(f))
    score_model_args = fix_legacy_args(score_model_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = build_dataset(args, score_model_args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Confidence model
    confidence_model = None
    confidence_model_args = None
    if args.confidence_model_dir:
        with open(Path(args.confidence_model_dir) / "model_parameters.yml") as f:
            confidence_model_args = Namespace(**yaml.safe_load(f))
        confidence_model_args = fix_legacy_args(confidence_model_args)
        confidence_model = load_model_checkpoint(
            args.confidence_model_dir,
            args.confidence_ckpt,
            confidence_model_args,
            device,
            old=getattr(args, "old_confidence_model", False),
        )

    # Score model
    model = None
    if not args.no_model:
        model = load_model_checkpoint(
            args.model_dir,
            args.ckpt,
            score_model_args,
            device,
            old=getattr(args, "old_score_model", False),
        )

    # Scheduler
    tr_schedule = get_t_schedule(
        sigma_schedule="expbeta",
        inference_steps=args.inference_steps,
        inf_sched_alpha=1.0,
        inf_sched_beta=1.0,
        t_max=1.0,
    )

    results = {"rmsds": [], "confidences": [], "names": []}
    for orig in tqdm(test_loader, disable=not args.tqdm):
        data_list = [copy.deepcopy(orig) for _ in range(args.samples_per_complex)]

        randomize_position(
            data_list,
            score_model_args.no_torsion,
            args.no_random,
            score_model_args.tr_sigma_max,
            False,  # pocket_knowledge
            5.0,    # pocket_cutoff
        )

        if not args.no_model:
            data_list, confidence = sampling(
                data_list=data_list,
                model=model,
                inference_steps=args.inference_steps,
                tr_schedule=tr_schedule,
                rot_schedule=tr_schedule,
                tor_schedule=tr_schedule,
                device=device,
                t_to_sigma=t2s(score_model_args),
                model_args=score_model_args,
                no_random=args.no_random,
                ode=args.ode,
                confidence_model=confidence_model,
                confidence_model_args=confidence_model_args,
                batch_size=args.batch_size,
                no_final_step_noise=args.no_final_step_noise,
            )
        else:
            confidence = None

        # Compute RMSD
        filterHs = data_list[0]["ligand"].x[:, 0] != 0
        orig_pos = orig["ligand"].orig_pos[filterHs.cpu()] - orig.original_center.cpu()
        lig_pos = np.stack([g["ligand"].pos[filterHs].cpu().numpy() for g in data_list])
        rmsd = np.min(np.sqrt(((lig_pos - orig_pos.numpy()) ** 2).sum(axis=-1)).mean(axis=-1))

        results["rmsds"].append(rmsd)
        results["names"].append(orig.name[0])
        results["confidences"].append(confidence.cpu().numpy() if confidence is not None else None)

    # Save
    np.savez(Path(args.out_dir) / "results.npz", **results)
    print("Inference complete. Results saved to", args.out_dir)


if __name__ == "__main__":
    main()
