#!/usr/bin/env python
"""
Single-file entry point for FlowDock inference.
Usage:
    python inference.py --protein_ligand_csv data.csv --out_dir results
"""
from __future__ import annotations

import datetime
import logging
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from Bio.PDB import PDBParser
from rdkit import Chem, RDLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import get_t_schedule, t_to_sigma
from utils.inference_tools import (Timer, seed_everything, save_protein,
                                   write_mol_with_coords)
from utils.sampling import randomize_position, sampling
from utils.utils import get_model

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# -------------------- 配置 -------------------- #
@dataclass
class RunConfig:
    # 推理超参数
    config: Path | None = None
    protein_ligand_csv: Path | None = None
    protein_path: Path = Path("data/dummy_data/1a0q_protein.pdb")
    ligand: str = "COc(cc1)ccc1C#N"
    out_dir: Path = Path("results/user_inference")
    esm_embeddings_path: Path = Path("data/esm2_output")

    model_dir: Path = Path("workdir/paper_score_model")
    ckpt: str = "best_ema_inference_epoch_model.pt"
    confidence_model_dir: Path | None = None
    confidence_ckpt: str = "best_model_epoch75.pt"

    # 采样 & 后处理
    samples_per_complex: int = 10
    savings_per_complex: int = 1
    batch_size: int = 32
    inference_steps: int = 20
    actual_steps: int | None = None
    ode: bool = False
    no_random: bool = False
    no_final_step_noise: bool = False
    protein_dynamic: bool = False
    relax: bool = False
    keep_local_structures: bool = False
    save_visualisation: bool = False
    use_existing_cache: bool = False

    # 杂项
    cache_path: Path = Path("data/cache")
    num_workers: int = 1
    seed: int = 42
    sigma_schedule: str = "expbeta"


# -------------------- 引擎 -------------------- #
class InferenceEngine:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    # ---------- 对外 API ---------- #
    def run(self, pairs: List[Tuple[str, str, str]]) -> None:
        """pairs: [(name, protein_path, ligand_path_or_smiles), ...]"""
        dataset = self._build_dataset(pairs)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_rows = []
        for orig_graph in tqdm(loader, desc="Inference"):
            try:
                rows = self._predict_single(orig_graph)
            except Exception as e:
                logging.warning("First attempt failed for %s: %s", orig_graph.name[0], e)
                try:
                    rows = self._predict_single(orig_graph)
                except Exception as e2:
                    logging.error("Second attempt failed for %s: %s", orig_graph.name[0], e2)
                    continue
            all_rows.extend(rows)

        # 输出 CSV
        pd.DataFrame(all_rows).to_csv(self.cfg.out_dir / "complete_affinity_prediction.csv", index=False)

    # ---------- 内部实现 ---------- #
    def _build_model(self) -> None:
        with open(self.cfg.model_dir / "model_parameters.yml") as f:
            self.score_args = yaml.safe_load(f)
        self.t_to_sigma = t_to_sigma(self.score_args)

        self.model = get_model(self.score_args, self.device, t_to_sigma=self.t_to_sigma, no_parallel=True)
        state = torch.load(self.cfg.model_dir / self.cfg.ckpt, map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        if self.cfg.confidence_model_dir:
            with open(self.cfg.confidence_model_dir / "model_parameters.yml") as f:
                conf_args = yaml.safe_load(f)
            self.conf_model = get_model(conf_args, self.device, t_to_sigma=self.t_to_sigma,
                                        no_parallel=True, confidence_mode=True)
            ckpt = torch.load(self.cfg.confidence_model_dir / self.cfg.confidence_ckpt, map_location="cpu")
            self.conf_model.load_state_dict(ckpt, strict=True)
            self.conf_model.eval()
        else:
            self.conf_model = None

    def _build_dataset(self, pairs):
        names, proteins, ligands = zip(*pairs)
        return PDBBind(
            transform=None, root="",
            name_list=list(names),
            protein_path_list=list(proteins),
            ligand_descriptions=list(ligands),
            receptor_radius=self.score_args["receptor_radius"],
            cache_path=self.cfg.cache_path,
            remove_hs=self.score_args["remove_hs"],
            c_alpha_max_neighbors=self.score_args["c_alpha_max_neighbors"],
            all_atoms=self.score_args["all_atoms"],
            atom_radius=self.score_args["atom_radius"],
            atom_max_neighbors=self.score_args["atom_max_neighbors"],
            esm_embeddings_path=self.cfg.esm_embeddings_path,
            num_workers=self.cfg.num_workers,
            use_existing_cache=self.cfg.use_existing_cache,
            keep_local_structures=self.cfg.keep_local_structures,
            require_ligand=True,
            require_receptor=True,
        )

    def _predict_single(self, graph):
        data_list = [graph.clone() for _ in range(self.cfg.samples_per_complex)]
        randomize_position(
            data_list,
            no_torsion=False,  # 从 model_args 中读取
            no_random=self.cfg.no_random,
            tr_sigma_max=self.score_args["tr_sigma_max"],
            rot_sigma_max=self.score_args["rot_sigma_max"],
            tor_sigma_max=self.score_args["tor_sigma_max"],
            res_tr_sigma_max=self.score_args["res_tr_sigma_max"],
            res_rot_sigma_max=self.score_args["res_rot_sigma_max"],
        )

        # adjustor
        tr_schedule = get_t_schedule(self.cfg.inference_steps)
        rot_schedule = tor_schedule = res_tr_schedule = res_rot_schedule = res_chi_schedule = tr_schedule

        # sampling
        final_data, steps_data, lddt_pred, affinity_pred = sampling(
            data_list, self.model,
            inference_steps=self.cfg.actual_steps or self.cfg.inference_steps,
            tr_schedule=tr_schedule, rot_schedule=rot_schedule,
            tor_schedule=tor_schedule,
            device=self.device,
            t_to_sigma=self.t_to_sigma,
            model_args=self.score_args,
            no_random=self.cfg.no_random,
            ode=self.cfg.ode,
            batch_size=self.cfg.batch_size,
            no_final_step_noise=self.cfg.no_final_step_noise,
            protein_dynamic=self.cfg.protein_dynamic,
        )

      
        return self._save_outputs(graph, final_data, lddt_pred, affinity_pred)

    def _save_outputs(self, graph, final_data, lddt_pred, affinity_pred):
        name = graph.name[0]
        write_dir = self.cfg.out_dir / f"index{name.split('_')[-1]}"
        write_dir.mkdir(parents=True, exist_ok=True)

        ref_lig_path = write_dir / "ref_ligandFile.sdf"
        Chem.SDWriter(str(ref_lig_path)).write(graph.mol[0])

     
        lddt = lddt_pred.view(-1).cpu().numpy()
        affinity = affinity_pred.view(-1).cpu().numpy()

        rows = []
        for rank, (pos, l, a) in enumerate(zip(
                [d["ligand"].pos for d in final_data], lddt, affinity
        )):
            mol = graph.mol[0]
            lig_path = write_dir / f"rank{rank + 1}_ligand_lddt{l:.2f}_affinity{a:.2f}.sdf"
            write_mol_with_coords(mol, pos + graph.original_center, str(lig_path))

            rec_path = write_dir / f"rank{rank + 1}_receptor_lddt{l:.2f}_affinity{a:.2f}.pdb"
            save_protein(graph.rec_pdb[0], str(rec_path))

            rows.append({"name": name, "rank": rank + 1, "lddt": l, "affinity": a})

        return rows


# -------------------- main -------------------- #
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protein_ligand_csv", type=Path)
    parser.add_argument("--protein_path", type=Path, default="data/dummy_data/1a0q_protein.pdb")
    parser.add_argument("--ligand", default="COc(cc1)ccc1C#N")
    parser.add_argument("--out_dir", type=Path, default="results/user_inference")
    for f in RunConfig.__dataclass_fields__.values():
        if f.name not in {"config"}:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    cli_args = parser.parse_args()

    cfg = RunConfig(**{k: v for k, v in vars(cli_args).items() if v is not None})
    if cfg.config:
        cfg = RunConfig(**(yaml.safe_load(cfg.config.read_text()) | vars(cli_args)))

    seed_everything(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

  
    if cfg.protein_ligand_csv:
        df = pd.read_csv(cfg.protein_ligand_csv)
        pairs = [(f"idx_{i}", Path(r.protein_path), r.ligand) for i, r in df.iterrows()]
    else:
        pairs = [("idx_0", cfg.protein_path, cfg.ligand)]

    engine = InferenceEngine(cfg)
    with Timer("Total inference"):
        engine.run(pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
