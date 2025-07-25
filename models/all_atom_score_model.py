# all-atom-model.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Literal, Sequence, Any

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from esm.pretrained import load_model_and_alphabet
from torch import Tensor
from torch_cluster import radius, radius_graph
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean

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

AGGREGATORS: Dict[str, Any] = {
    "mean": lambda x: torch.mean(x, dim=1),
    "max": lambda x: torch.max(x, dim=1)[0],
    "min": lambda x: torch.min(x, dim=1)[0],
    "std": lambda x: torch.std(x, dim=1),
}


class _ConvSpec:
    """Small dataclass describing one Tensor-Product layer."""

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
    specs: Sequence[_ConvSpec],
    *,
    sh_irreps: str,
    batch_norm: bool,
    dropout: float,
    **kwargs: Any,
) -> nn.ModuleList:
    """Factory: build ModuleList of TensorProductConvLayer from specs."""
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
            **kwargs,
        )
        for s in specs
    )


# ---------------------------------------------------------------------------
#  Model
# ---------------------------------------------------------------------------

class AAModel(nn.Module):
    """
    Optimised all-atom model for flexible protein–ligand docking.
    Constructor signature kept 100 % identical to the original codebase.
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
        asynchronous_noise_schedule: bool = False,
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
        crop_beyond: Optional[float] = None,
    ):
        super().__init__()

        # sanity checks
        assert not sidechain_pred, "side-chain prediction not implemented"
        assert not depthwise_convolution, "depthwise convolution not implemented"
        assert not include_miscellaneous_atoms, "misc atoms not supported yet"
        if parallel > 1:
            assert affinity_prediction

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
        self.dynamic_max_cross = dynamic_max_cross
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
        self.asynchronous_noise_schedule = asynchronous_noise_schedule
        self.affinity_prediction = affinity_prediction
        self.parallel = parallel
        self.parallel_aggregators = parallel_aggregators.split()
        self.fixed_center_conv = fixed_center_conv
        self.no_aminoacid_identities = no_aminoacid_identities
        self.differentiate_convolutions = differentiate_convolutions
        self.reduce_pseudoscalars = reduce_pseudoscalars
        self.atom_confidence = atom_confidence
        self.atom_num_confidence_outputs = atom_num_confidence_outputs
        self.crop_beyond = crop_beyond

        # language model handling
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
        self._build_conv_layers(ns, nv, use_second_order_repr, reduce_pseudoscalars, batch_norm, dropout,
                                tp_weights_layers, differentiate_convolutions)

        # ---------------------------------------------------------------------
        #  heads
        # ---------------------------------------------------------------------
        self._build_heads(ns, dropout, confidence_dropout, confidence_no_batchnorm, num_confidence_outputs)

    # -------------------------------------------------------------------------
    #  builders
    # -------------------------------------------------------------------------

    def _build_embeddings(self, ns: int, dropout: float) -> None:
        """Create all node / edge embeddings from a small config table."""
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
            "atom": dict(
                node_feat=rec_atom_feature_dims,
                edge_dim=self.distance_embed_dim,
                sigma_dim=0,
            ),
            "lr": dict(edge_dim=self.sigma_embed_dim + self.cross_distance_embed_dim),
            "la": dict(edge_dim=self.sigma_embed_dim + self.cross_distance_embed_dim),
            "ar": dict(edge_dim=self.distance_embed_dim),
        }

        for name, c in cfg.items():
            if "node_feat" in c:
                setattr(
                    self,
                    f"{name}_node_embedding",
                    AtomEncoder(ns, c["node_feat"], c["sigma_dim"], lm_embedding_dim=self.lm_dim),
                )
            edge_dim = c["edge_dim"]
            setattr(
                self,
                f"{name}_edge_embedding",
                nn.Sequential(
                    nn.Linear(edge_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns),
                ),
            )

        # receptor sigma embedding
        self.rec_sigma_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim, ns),
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
    ) -> None:
        """Generate all Tensor-Product layers via _ConvSpec list."""
        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)

        # receptor (residue + atom) embedding layers
        rec_specs = [
            _ConvSpec(irrep_seq[i], irrep_seq[i + 1], 3 * ns)
            for i in range(self.num_prot_emb_layers)
        ]
        self.rec_emb_layers = _make_conv_layers(
            rec_specs,
            sh_irreps=self.sh_irreps,
            batch_norm=batch_norm,
            dropout=dropout,
            tp_weights_layers=tp_weights_layers,
        )

        if self.embed_also_ligand:
            lig_specs = [
                _ConvSpec(irrep_seq[i], irrep_seq[i + 1], 3 * ns)
                for i in range(self.num_prot_emb_layers)
            ]
            self.lig_emb_layers = _make_conv_layers(
                lig_specs,
                sh_irreps=self.sh_irreps,
                batch_norm=batch_norm,
                dropout=dropout,
                tp_weights_layers=tp_weights_layers,
            )

        # main interaction layers
        conv_specs = []
        for i in range(self.num_prot_emb_layers, self.num_prot_emb_layers + self.num_conv_layers):
            groups = 1
            if differentiate_convolutions:
                if i == self.num_prot_emb_layers + self.num_conv_layers - 1:
                    groups = 3
                else:
                    groups = 9
            conv_specs.append(
                _ConvSpec(
                    irrep_seq[min(i, len(irrep_seq) - 1)],
                    irrep_seq[min(i + 1, len(irrep_seq) - 1)],
                    3 * ns,
                    edge_groups=groups,
                )
            )
        self.conv_layers = _make_conv_layers(
            conv_specs,
            sh_irreps=self.sh_irreps,
            batch_norm=batch_norm,
            dropout=dropout,
            tp_weights_layers=tp_weights_layers,
        )

    def _build_heads(
        self,
        ns: int,
        dropout: float,
        confidence_dropout: float,
        confidence_no_batchnorm: bool,
        num_confidence_outputs: int,
    ) -> None:
        """Build confidence / affinity / final score heads."""
        if self.confidence_mode:
            input_size = (
                ns + (self.nv if self.reduce_pseudoscalars else ns)
                if len(self.conv_layers) + len(self.rec_emb_layers) >= 3
                else ns
            )
            out_dim = num_confidence_outputs + (1 if self.affinity_prediction else 0)

            if self.parallel > 1:
                out_dim += ns  # store intermediate features for affinity predictor

            layers: List[nn.Module] = [
                nn.Linear(input_size, ns),
                nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, out_dim),
            ]
            self.confidence_predictor = nn.Sequential(*layers)

            if self.atom_confidence:
                atom_layers = layers[:-1] + [
                    nn.Linear(ns, self.atom_num_confidence_outputs + ns)
                ]
                self.atom_confidence_predictor = nn.Sequential(*atom_layers)

            if self.parallel > 1:
                self.affinity_predictor = nn.Sequential(
                    nn.Linear(len(self.parallel_aggregators) * ns, ns),
                    nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, ns),
                    nn.Identity() if confidence_no_batchnorm else nn.BatchNorm1d(ns),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, 1),
                )
        else:
            # tr / rot / tor heads
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f"2x1o + 2x1e" if not self.odd_parity else "1x1o + 1x1e",
                n_edge_features=2 * ns,
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

    def forward(self, data):
        """
        Returns
        -------
        (tr_pred, rot_pred, tor_pred, None)
            or (confidence, atom_confidence)
        """
        if self.crop_beyond is not None:
            raise NotImplementedError("Cropping not implemented yet")

        if self.no_aminoacid_identities:
            data["receptor"].x = torch.zeros_like(data["receptor"].x)

        # sigma handling -------------------------------------------------------
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(
                *[data.complex_t[k] for k in ("tr", "rot", "tor")]
            )
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[k] for k in ("tr", "rot", "tor")]

        # embeddings -----------------------------------------------------------
        lig_n, lig_ei, lig_ea, lig_sh, lig_ew, \
            rec_n, rec_ei, rec_ea, rec_sh, rec_ew, \
            atom_n, atom_ei, atom_ea, atom_sh, atom_ew, \
            ar_ei, ar_ea, ar_sh, ar_ew = self.embedding(data)

        # cross graphs ---------------------------------------------------------
        cross_cutoff = (
            tr_sigma * 3 + 20
            if self.dynamic_max_cross
            else torch.tensor(self.cross_max_distance, device=data["ligand"].pos.device)
        )
        lr_ei, lr_ea, lr_sh, lr_ew, \
            la_ei, la_ea, la_sh, la_ew = self.build_cross_lig_conv_graph(data, cross_cutoff)
        lr_ea = self.lr_edge_embedding(lr_ea)
        la_ea = self.la_edge_embedding(la_ea)

        # pack everything into one big graph -----------------------------------
        n_lig, n_rec = len(lig_n), len(rec_n)
        node_attr = torch.cat([lig_n, rec_n, atom_n], dim=0)

        # edge index remapping
        rec_ei = rec_ei + n_lig
        atom_ei = atom_ei + n_lig + n_rec
        lr_ei[1] += n_lig
        la_ei[1] += n_lig + n_rec
        ar_ei[0] += n_lig + n_rec
        ar_ei[1] += n_lig

        edge_index = torch.cat(
            [
                lig_ei,
                lr_ei,
                la_ei,
                rec_ei,
                lr_ei[[1, 0]],
                ar_ei[[1, 0]],
                atom_ei,
                la_ei[[1, 0]],
                ar_ei,
            ],
            dim=1,
        )
        edge_attr = torch.cat(
            [
                lig_ea,
                lr_ea,
                la_ea,
                rec_ea,
                lr_ea,
                ar_ea,
                atom_ea,
                la_ea,
                ar_ea,
            ],
            dim=0,
        )
        edge_sh = torch.cat(
            [
                lig_sh,
                lr_sh,
                la_sh,
                rec_sh,
                lr_sh,
                ar_sh,
                atom_sh,
                la_sh,
                ar_sh,
            ],
            dim=0,
        )
        edge_weight = (
            torch.cat(
                [
                    lig_ew,
                    lr_ew,
                    la_ew,
                    rec_ew,
                    lr_ew,
                    ar_ew,
                    atom_ew,
                    la_ew,
                    ar_ew,
                ],
                dim=0,
            )
            if torch.is_tensor(lig_ew)
            else torch.ones(edge_index.size(1), 1, device=edge_index.device)
        )

        # convolutions ---------------------------------------------------------
        splits = [lig_ea.numel(), lr_ea.numel(), la_ea.numel(), rec_ea.numel(), lr_ea.numel(), ar_ea.numel(),
                  atom_ea.numel(), la_ea.numel(), ar_ea.numel()]
        cs = [0] + torch.cumsum(torch.tensor(splits), 0).tolist()

        for l, layer in enumerate(self.conv_layers):
            if l < len(self.conv_layers) - 1:
                ea = torch.cat([edge_attr, node_attr[edge_index[0], : self.ns], node_attr[edge_index[1], : self.ns]], -1)
                if self.differentiate_convolutions:
                    ea = [ea[cs[i] : cs[i + 1]] for i in range(len(cs) - 1)]
                node_attr = layer(node_attr, edge_index, ea, edge_sh, edge_weight=edge_weight)
            else:
                ea = torch.cat(
                    [edge_attr[: cs[3]], node_attr[edge_index[0, : cs[3]], : self.ns], node_attr[edge_index[1, : cs[3]], : self.ns]], -1
                )
                if self.differentiate_convolutions:
                    ea = [ea[: cs[0]], ea[cs[0] : cs[1]], ea[cs[1] : cs[2]]]
                node_attr = layer(node_attr, edge_index[:, : cs[3]], ea, edge_sh[: cs[3]], edge_weight=edge_weight[: cs[3]])

        lig_node_attr = node_attr[: len(lig_n)]

        # confidence / affinity -------------------------------------------------
        if self.confidence_mode:
            scalar_lig_attr = (
                torch.cat([lig_node_attr[:, : self.ns], lig_node_attr[:, -(self.nv if self.reduce_pseudoscalars else self.ns) :]], dim=1)
                if len(self.conv_layers) + len(self.rec_emb_layers) >= 3
                else lig_node_attr[:, : self.ns]
            )
            if self.atom_confidence:
                scalar_lig_attr = self.atom_confidence_predictor(scalar_lig_attr)
                atom_conf = scalar_lig_attr[:, : self.atom_num_confidence_outputs]
                scalar_lig_attr = scalar_lig_attr[:, self.atom_num_confidence_outputs :]
            else:
                atom_conf = torch.zeros(len(lig_node_attr), device=lig_node_attr.device)

            graph_attr = scatter_mean(scalar_lig_attr, data["ligand"].batch, dim=0)
            conf = self.confidence_predictor(graph_attr).squeeze(-1)

            if self.parallel > 1:
                conf, aff = conf[:, 0], conf[:, 1:]
                aff = aff.view(data.num_graphs, self.parallel, -1)
                aff = torch.cat([AGGREGATORS[a](aff) for a in self.parallel_aggregators], dim=-1)
                aff = self.affinity_predictor(aff).squeeze(-1)
                return (conf, aff), atom_conf
            return conf, atom_conf

        # tr / rot / tor --------------------------------------------------------
        center_ei, center_ea, center_sh = self.build_center_conv_graph(data)
        center_ea = self.center_edge_embedding(center_ea)
        if self.fixed_center_conv:
            center_ea = torch.cat([center_ea, lig_node_attr[center_ei[1], : self.ns]], -1)
        else:
            center_ea = torch.cat([center_ea, lig_node_attr[center_ei[0], : self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_ei, center_ea, center_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

        # time embedding for norm
        if self.separate_noise_schedule:
            graph_sigma_emb = torch.cat(
                [self.timestep_emb_func(data.complex_t[k]) for k in ("tr", "rot", "tor")], dim=1
            )
        elif asynchronous_noise_schedule:
            graph_sigma_emb = self.timestep_emb_func(data.complex_t["t"])
        else:
            graph_sigma_emb = self.timestep_emb_func(data.complex_t["tr"])

        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1, keepdim=True)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1, keepdim=True)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(tr_pred.device)

        # torsion --------------------------------------------------------------
        if self.no_torsion or data["ligand"].edge_mask.sum() == 0:
            return tr_pred, rot_pred, torch.empty(0, device=self.device), None

        bonds, tor_ei, tor_ea, tor_sh, tor_ew = self.build_bond_conv_graph(data)
        bond_vec = data["ligand"].pos[bonds[1]] - data["ligand"].pos[bonds[0]]
        bond_attr = lig_node_attr[bonds[0]] + lig_node_attr[bonds[1]]

        bond_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization="component")
        tor_sh = self.final_tp_tor(tor_sh, bond_sh[tor_ei[0]])

        tor_ea = torch.cat(
            [tor_ea, lig_node_attr[tor_ei[1], : self.ns], bond_attr[tor_ei[0], : self.ns]], -1
        )
        tor_pred = self.tor_bond_conv(
            lig_node_attr,
            tor_ei,
            tor_ea,
            tor_sh,
            out_nodes=data["ligand"].edge_mask.sum(),
            reduce="mean",
            edge_weight=tor_ew,
        )
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data["ligand"].batch][data["ligand", "ligand"].edge_index[0]][data["ligand"].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(
                torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float().to(tor_pred.device)
            )
        return tr_pred, rot_pred, tor_pred, None

    # -------------------------------------------------------------------------
    #  graph builders
    # -------------------------------------------------------------------------

    def get_edge_weight(self, edge_vec: Tensor, max_norm: float) -> Tensor:
        if self.smooth_edges:
            norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)
            return 0.5 * (torch.cos(norm) + 1.0).unsqueeze(-1)
        return torch.ones(edge_vec.size(0), 1, device=edge_vec.device)

    # ------ ligand -----------------------------------------------------------
    def build_lig_conv_graph(self, data):
        if self.separate_noise_schedule:
            data["ligand"].node_sigma_emb = torch.cat(
                [self.timestep_emb_func(data["ligand"].node_t[k]) for k in ("tr", "rot", "tor")], dim=1
            )
        elif self.asynchronous_noise_schedule:
            data["ligand"].node_sigma_emb = self.timestep_emb_func(data["ligand"].node_t["t"])
        else:
            data["ligand"].node_sigma_emb = self.timestep_emb_func(data["ligand"].node_t["tr"])

        if self.parallel == 1:
            radius_edges = radius_graph(data["ligand"].pos, self.lig_max_radius, data["ligand"].batch)
        else:
            # parallel splitting (kept for backward compatibility)
            batches = torch.zeros(data.num_graphs, device=data["ligand"].x.device).long()
            batches = batches.index_add(
                0,
                data["ligand"].batch,
                torch.ones(len(data["ligand"].batch), device=data["ligand"].x.device).long(),
            )
            outer_batches = data.num_graphs
            b = [
                torch.ones(batches[i].item() // self.parallel, device=data["ligand"].x.device).long()
                * (self.parallel * i + j)
                for i in range(outer_batches)
                for j in range(self.parallel)
            ]
            data["ligand"].batch_parallel = torch.cat(b)
            radius_edges = radius_graph(data["ligand"].pos, self.lig_max_radius, data["ligand"].batch_parallel)

        edge_index = torch.cat([data["ligand", "ligand"].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat(
            [
                data["ligand", "ligand"].edge_attr,
                torch.zeros(
                    radius_edges.size(1), self.in_lig_edge_features, device=data["ligand"].x.device
                ),
            ],
            dim=0,
        )
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], dim=1)
        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], dim=1)

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst] - data["ligand"].pos[src]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], dim=1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    # ------ receptor --------------------------------------------------------
    def build_rec_conv_graph(self, data):
        edge_index = data["receptor", "receptor"].edge_index
        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst] - data["receptor"].pos[src]
        edge_attr = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)
        return data["receptor"].x, edge_attr, edge_sh, edge_weight

    # ------ receptor atoms --------------------------------------------------
    def build_atom_conv_graph(self, data):
        edge_index = data["atom", "atom"].edge_index
        src, dst = edge_index
        edge_vec = data["atom"].pos[dst] - data["atom"].pos[src]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return data["atom"].x, edge_attr, edge_sh, edge_weight

    # ------ cross graphs ----------------------------------------------------
    def build_cross_lig_conv_graph(self, data, cross_cutoff):
        # ligand <-> receptor residues
        if torch.is_tensor(cross_cutoff):
            lr_ei = radius(
                data["receptor"].pos / cross_cutoff[data["receptor"].batch],
                data["ligand"].pos / cross_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            lr_ei = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                cross_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        lr_vec = data["receptor"].pos[lr_ei[1]] - data["ligand"].pos[lr_ei[0]]
        lr_len = self.cross_distance_expansion(lr_vec.norm(dim=-1))
        lr_sigma = data["ligand"].node_sigma_emb[lr_ei[0]]
        lr_attr = torch.cat([lr_sigma, lr_len], dim=1)
        lr_sh = o3.spherical_harmonics(self.sh_irreps, lr_vec, normalize=True, normalization="component")
        lr_w = self.get_edge_weight(lr_vec, cross_cutoff if not torch.is_tensor(cross_cutoff) else cross_cutoff[data["ligand"].batch[lr_ei[0]]])

        # ligand <-> receptor atoms
        la_ei = radius(
            data["atom"].pos,
            data["ligand"].pos,
            self.lig_max_radius,
            data["atom"].batch,
            data["ligand"].batch,
            max_num_neighbors=10000,
        )
        la_vec = data["atom"].pos[la_ei[1]] - data["ligand"].pos[la_ei[0]]
        la_len = self.lig_distance_expansion(la_vec.norm(dim=-1))
        la_sigma = data["ligand"].node_sigma_emb[la_ei[0]]
        la_attr = torch.cat([la_sigma, la_len], dim=1)
        la_sh = o3.spherical_harmonics(self.sh_irreps, la_vec, normalize=True, normalization="component")
        la_w = self.get_edge_weight(la_vec, self.lig_max_radius)

        return lr_ei, lr_attr, lr_sh, lr_w, la_ei, la_attr, la_sh, la_w

    def build_cross_rec_conv_graph(self, data):
        # atom <-> receptor residues (fixed)
        ar_ei = data["atom", "receptor"].edge_index
        ar_vec = data["receptor"].pos[ar_ei[1]] - data["atom"].pos[ar_ei[0]]
        ar_attr = self.rec_distance_expansion(ar_vec.norm(dim=-1))
        ar_sh = o3.spherical_harmonics(self.sh_irreps, ar_vec, normalize=True, normalization="component")
        return ar_attr, ar_sh, torch.ones(ar_ei.size(1), 1, device=ar_ei.device)

    # -------------------------------------------------------------------------
    #  centre / bond graph builders
    # -------------------------------------------------------------------------

    def build_center_conv_graph(self, data):
        ei = torch.stack([data["ligand"].batch, torch.arange(len(data["ligand"].batch), device=data["ligand"].x.device)])
        center_pos = scatter_mean(data["ligand"].pos, data["ligand"].batch, dim=0)
        vec = data["ligand"].pos[ei[1]] - center_pos[ei[0]]
        attr = self.center_distance_expansion(vec.norm(dim=-1))
        sigma_emb = data["ligand"].node_sigma_emb[ei[1]]
        attr = torch.cat([attr, sigma_emb], dim=1)
        sh = o3.spherical_harmonics(self.sh_irreps, vec, normalize=True, normalization="component")
        return ei, attr, sh

    def build_bond_conv_graph(self, data):
        bonds = data["ligand", "ligand"].edge_index[:, data["ligand"].edge_mask].long()
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        ei = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
            max_num_neighbors=10000,
        )
        vec = data["ligand"].pos[ei[1]] - bond_pos[ei[0]]
        attr = self.final_edge_embedding(self.lig_distance_expansion(vec.norm(dim=-1)))
        sh = o3.spherical_harmonics(self.sh_irreps, vec, normalize=True, normalization="component")
        ew = self.get_edge_weight(vec, self.lig_max_radius)
        return bonds, ei, attr, sh, ew

