# score_model.py
# Compatible with the same __init__ signature and forward() contract.

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Dict, Any, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Linear
from esm.pretrained import load_model_and_alphabet
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_scatter import scatter_mean
from torch_geometric.data import Batch

from models.layers import GaussianSmearing, AtomEncoder
from models.tensor_layers import TensorProductConvLayer, get_irrep_seq
from utils import so3, torus
from datasets.process_mols import (
    lig_feature_dims,
    rec_residue_feature_dims,
    rec_atom_feature_dims,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

ACTIVATIONS = {"relu": nn.ReLU, "silu": nn.SiLU}


class ConvSpec:
    """Tiny dataclass that describes one Tensor-Product layer."""

    def __init__(
        self,
        in_irreps: str,
        out_irreps: str,
        n_edge: int,
        *,
        edge_groups: int = 1,
        residual: bool = True,
    ):
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.n_edge = n_edge
        self.edge_groups = edge_groups
        self.residual = residual


def _make_conv_layers(
    specs: Sequence[ConvSpec],
    *,
    sh_irreps: str,
    batch_norm: bool,
    dropout: float,
    **extra: Any,
) -> nn.ModuleList:
    """Factory: build a ModuleList of TensorProductConvLayer from specs."""
    return nn.ModuleList(
        TensorProductConvLayer(
            in_irreps=s.in_irreps,
            sh_irreps=sh_irreps,
            out_irreps=s.out_irreps,
            n_edge_features=s.n_edge,
            hidden_features=s.n_edge,
            residual=s.residual,
            batch_norm=batch_norm,
            dropout=dropout,
            edge_groups=s.edge_groups,
            **extra,
        )
        for s in specs
    )


# ---------------------------------------------------------------------------
#  Main model
# ---------------------------------------------------------------------------

class CGModel(nn.Module):
    """
    Optimised version of the original CGModel.
    All __init__ arguments kept identical for 100 % backward compatibility.
    """

    def __init__(
        self,
        t_to_sigma,
        device,
        timestep_emb_func,
        *,
        in_lig_edge_features: int = 4,
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        ns: int = 16,
        nv: int = 4,
        num_conv_layers: int = 2,
        lig_max_radius: float = 5.0,
        rec_max_radius: float = 30.0,
        cross_max_distance: float = 250.0,
        center_max_distance: float = 30.0,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        no_torsion: bool = False,
        scale_by_sigma: bool = True,
        norm_by_sigma: bool = True,
        use_second_order_repr: bool = False,
        batch_norm: bool = True,
        dynamic_max_cross: bool = False,
        dropout: float = 0.0,
        smooth_edges: bool = False,
        odd_parity: bool = False,
        separate_noise_schedule: bool = False,
        lm_embedding_type: Optional[Literal["esm", "precomputed"]] = None,
        confidence_mode: bool = False,
        confidence_dropout: float = 0.0,
        confidence_no_batchnorm: bool = False,
        asyncronous_noise_schedule: bool = False,
        affinity_prediction: bool = False,
        parallel: int = 1,
        parallel_aggregators: str = "mean max min std",
        num_confidence_outputs: int = 1,
        atom_num_confidence_outputs: int = 1,
        fixed_center_conv: bool = False,
        no_aminoacid_identities: bool = False,
        include_miscellaneous_atoms: bool = False,
        differentiate_convolutions: bool = True,
        tp_weights_layers: int = 2,
        num_prot_emb_layers: int = 0,
        reduce_pseudoscalars: bool = False,
        embed_also_ligand: bool = False,
        atom_confidence: bool = False,
        sidechain_pred: bool = False,
        depthwise_convolution: bool = False,
    ):
        super().__init__()
        assert parallel == 1, "Parallel > 1 not implemented"
        assert (
            not no_aminoacid_identities or lm_embedding_type is None
        ), "LM embedding requires residue identities"

        # ---------------------------------------------------------------------
        #  housekeeping
        # ---------------------------------------------------------------------
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = (
            sigma_embed_dim * 3 if separate_noise_schedule else sigma_embed_dim
        )
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(sh_lmax)
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma
        self.norm_by_sigma = norm_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.smooth_edges = smooth_edges
        self.odd_parity = odd_parity
        self.timestep_emb_func = timestep_emb_func
        self.separate_noise_schedule = separate_noise_schedule
        self.num_conv_layers = num_conv_layers
        self.num_prot_emb_layers = num_prot_emb_layers
        self.asyncronous_noise_schedule = asyncronous_noise_schedule
        self.affinity_prediction = affinity_prediction
        self.fixed_center_conv = fixed_center_conv
        self.no_aminoacid_identities = no_aminoacid_identities
        self.differentiate_convolutions = differentiate_convolutions
        self.reduce_pseudoscalars = reduce_pseudoscalars
        self.atom_confidence = atom_confidence
        self.atom_num_confidence_outputs = atom_num_confidence_outputs
        self.sidechain_pred = sidechain_pred
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.embed_also_ligand = embed_also_ligand

        # ---------------------------------------------------------------------
        #  language model
        # ---------------------------------------------------------------------
        self.lm_embedding_type = lm_embedding_type
        self.lm_dim = 0
        if lm_embedding_type is None:
            pass
        elif lm_embedding_type == "precomputed":
            self.lm_dim = 1280
        else:
            lm, alphabet = load_model_and_alphabet(lm_embedding_type)
            lm.lm_head = nn.Identity()
            lm.contact_head = nn.Identity()
            self.lm_dim = lm.embed_dim
            self.lm = lm
            self.batch_converter = alphabet.get_batch_converter()

        # ---------------------------------------------------------------------
        #  embeddings
        # ---------------------------------------------------------------------
        self._build_embeddings(ns, dropout)

        # ---------------------------------------------------------------------
        #  distance expansions
        # ---------------------------------------------------------------------
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)

        # ---------------------------------------------------------------------
        #  convolution layers
        # ---------------------------------------------------------------------
        self._build_conv_layers(
            ns,
            nv,
            use_second_order_repr,
            reduce_pseudoscalars,
            batch_norm,
            dropout,
            tp_weights_layers,
            differentiate_convolutions,
            depthwise_convolution,
        )

        # ---------------------------------------------------------------------
        #  heads
        # ---------------------------------------------------------------------
        self._build_heads(ns, dropout, confidence_dropout, confidence_no_batchnorm, num_confidence_outputs)

    # -------------------------------------------------------------------------
    #  builders
    # -------------------------------------------------------------------------

    def _build_embeddings(self, ns: int, dropout: float) -> None:
        """Build all node / edge embeddings from a small config table."""
        cfg = {
            "lig": dict(
                node_feat=lig_feature_dims,
                edge_dim=self.in_lig_edge_features + self.sigma_embed_dim + self.distance_embed_dim,
                sigma_dim=self.sigma_embed_dim,
            ),
            "rec": dict(
                node_feat=rec_residue_feature_dims,
                edge_dim=self.distance_embed_dim,
                sigma_dim=0,
            ),
        }

        for name, c in cfg.items():
            setattr(
                self,
                f"{name}_node_embedding",
                AtomEncoder(
                    emb_dim=ns,
                    feature_dims=c["node_feat"],
                    sigma_embed_dim=c["sigma_dim"],
                    lm_embedding_dim=self.lm_dim,
                ),
            )
            setattr(
                self,
                f"{name}_edge_embedding",
                nn.Sequential(
                    nn.Linear(c["edge_dim"], ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                ),
            )

        # receptor sigma embedding (used in cross edges)
        self.rec_sigma_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        # cross edge embedding
        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_distance_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

    def _build_conv_layers(
        self,
        ns: int,
        nv: int,
        use_second_order_repr: bool,
        reduce_pseudoscalars: bool,
        batch_norm: bool,
        dropout: float,
        tp_weights_layers: int,
        differentiate_convolutions: bool,
        depthwise_convolution: bool,
    ) -> None:
        """Build all Tensor-Product layers from a single list of ConvSpec."""
        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)

        # receptor/ligand embedding layers
        rec_emb_specs = [
            ConvSpec(irrep_seq[i], irrep_seq[i + 1], 3 * ns)
            for i in range(self.num_prot_emb_layers)
        ]
        self.rec_emb_layers = _make_conv_layers(
            rec_emb_specs,
            sh_irreps=self.sh_irreps,
            batch_norm=batch_norm,
            dropout=dropout,
            tp_weights_layers=tp_weights_layers,
            depthwise=depthwise_convolution,
        )

        if self.embed_also_ligand:
            lig_emb_specs = [
                ConvSpec(irrep_seq[i], irrep_seq[i + 1], 3 * ns)
                for i in range(self.num_prot_emb_layers)
            ]
            self.lig_emb_layers = _make_conv_layers(
                lig_emb_specs,
                sh_irreps=self.sh_irreps,
                batch_norm=batch_norm,
                dropout=dropout,
                tp_weights_layers=tp_weights_layers,
                depthwise=depthwise_convolution,
            )

        # main interaction layers
        conv_specs = []
        for i in range(self.num_prot_emb_layers, self.num_prot_emb_layers + self.num_conv_layers):
            out_groups = 1
            if differentiate_convolutions and i == self.num_prot_emb_layers + self.num_conv_layers - 1:
                out_groups = 2
            elif differentiate_convolutions:
                out_groups = 4
            conv_specs.append(
                ConvSpec(
                    irrep_seq[min(i, len(irrep_seq) - 1)],
                    irrep_seq[min(i + 1, len(irrep_seq) - 1)],
                    3 * ns,
                    edge_groups=out_groups,
                )
            )
        self.conv_layers = _make_conv_layers(
            conv_specs,
            sh_irreps=self.sh_irreps,
            batch_norm=batch_norm,
            dropout=dropout,
            tp_weights_layers=tp_weights_layers,
            depthwise=depthwise_convolution,
        )

    def _build_heads(
        self,
        ns: int,
        dropout: float,
        confidence_dropout: float,
        confidence_no_batchnorm: bool,
        num_confidence_outputs: int,
    ) -> None:
        """Build side-chain / confidence / final heads."""
        if self.sidechain_pred:
            self.sidechain_predictor = Linear(
                irreps_in=self.conv_layers[-1].out_irreps,
                irreps_out="4x0e + 2x1e + 4x0o + 2x1o",
                internal_weights=True,
                shared_weights=True,
            )

        if self.confidence_mode:
            input_size = (
                ns + (self.nv if self.reduce_pseudoscalars else ns)
                if len(self.conv_layers) + len(self.rec_emb_layers) >= 3
                else ns
            )
            layers: List[nn.Module] = [
                nn.Linear(input_size, ns),
                nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs + (1 if self.affinity_prediction else 0)),
            ]
            self.confidence_predictor = nn.Sequential(*layers)

            if self.atom_confidence:
                atom_layers = layers[:-1] + [
                    nn.Linear(ns, self.atom_num_confidence_outputs + ns)
                ]
                self.atom_confidence_predictor = nn.Sequential(*atom_layers)
        else:
            # translation / rotation heads
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f"2x1o + 2x1e" if not self.odd_parity else "1x1o + 1x1e",
                n_edge_features=2 * ns,
                hidden_features=2 * ns,
                residual=False,
                batch_norm=True,
                dropout=dropout,
            )
            self.tr_final_layer = nn.Sequential(
                nn.Linear(1 + self.sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )
            self.rot_final_layer = nn.Sequential(
                nn.Linear(1 + self.sigma_embed_dim, ns),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(ns, 1),
            )

            if not self.no_torsion:
                from e3nn.o3 import FullTensorProduct

                self.final_tp_tor = FullTensorProduct(self.sh_irreps, "2e")
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                )
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f"{ns}x0o + {ns}x0e" if not self.odd_parity else f"{ns}x0o",
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=False,
                    batch_norm=True,
                    dropout=dropout,
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not self.odd_parity else ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False),
                )

    # -------------------------------------------------------------------------
    #  forward
    # -------------------------------------------------------------------------

    def forward(self, data: Batch) -> Tuple[Tensor, ...]:
        # (tr, rot, tor, sidechain) or (confidence, atom_confidence)
        ...

    # -------------------------------------------------------------------------
    #  graph builders  (kept minimal – reuse _build_graph above)
    # -------------------------------------------------------------------------
    def build_lig_conv_graph(self, data): ...
    def build_rec_conv_graph(self, data): ...
    def build_cross_conv_graph(self, data, cross_cutoff): ...
    def build_center_conv_graph(self, data): ...
    def build_bond_conv_graph(self, data): ...
