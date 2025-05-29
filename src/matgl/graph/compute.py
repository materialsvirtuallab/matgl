"""Computing various graph based operations."""

from __future__ import annotations

import typing
import warnings

import numpy as np
import torch
from torch_geometric.data import Data

import matgl

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def compute_pair_vector_and_distance(graph: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate bond vectors and distances using PyTorch Geometric Data object.

    Args:
        graph: PyTorch Geometric Data object containing pos, edge_index, and pbc_offshift.

    Returns:
        Tuple containing:
        - bond_vec (torch.Tensor): Vector from source node to destination node.
        - bond_dist (torch.Tensor): Bond distance between two atoms.
    """
    # Get source and destination node indices from edge_index
    src_idx, dst_idx = graph.edge_index

    # Get positions of source and destination nodes
    src_pos = graph.pos[src_idx]
    dst_pos = graph.pos[dst_idx]

    # Apply periodic boundary condition offsets
    if hasattr(graph, "pbc_offshift") and graph.pbc_offshift is not None:
        dst_pos = dst_pos + graph.pbc_offshift

    # Compute bond vectors and distances
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def compute_theta(
    graph: Data, cosine: bool = False, directed: bool = True, eps: float = 1e-7
) -> dict[str, torch.Tensor]:
    """
    Calculate bond angles from edges in a PyTorch Geometric line graph.

    Args:
        graph: PyTorch Geometric Data object representing a line graph, with bond_vec, bond_dist, and optional src_bond_sign.
        cosine: Whether to return the cosine of the angle or the angle itself.
        directed: Whether the line graph was created with directed edges. If True, bond vectors for source nodes are flipped using src_bond_sign.
        eps: Small value to clamp cosine values for numerical stability (avoid acos of values > 1.0).

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'cos_theta' or 'theta': Cosine of bond angle or angle in radians.
            - 'triple_bond_lengths': Bond distances for destination bonds.
    """
    # Get source and destination bond vectors
    src_idx, dst_idx = graph.edge_index
    vec1 = graph.bond_vec[src_idx]
    if directed and hasattr(graph, "src_bond_sign"):
        vec1 = vec1 * graph.src_bond_sign[src_idx]
    vec2 = graph.bond_vec[dst_idx]

    # Compute cosine of the angle
    key = "cos_theta" if cosine else "theta"
    val = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
    val = val.clamp(min=-1 + eps, max=1 - eps)  # Numerical stability

    if not cosine:
        val = torch.acos(val)

    graph[key] = val
    graph.triple_bond_lengths = graph.bond_dist[dst_idx]
    return graph, key


def compute_theta_and_phi(graph: Data) -> dict[str, torch.Tensor]:
    """
    Calculate bond angles theta and phi using a PyTorch Geometric line graph.

    Args:
        data: PyTorch Geometric Data object representing a line graph.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - cos_theta: Cosine of the bond angle.
            - phi: Tensor of zeros (placeholder for phi angle).
            - triple_bond_lengths: Bond distances for destination bonds.
    """
    graph, key = compute_theta(graph, cosine=True, directed=False)
    graph.phi = torch.zeros_like(graph[key])
    return graph


def separate_node_edge_keys(graph: Data) -> tuple[list[str], list[str], list[str]]:
    """Separates keys in a PyTorch Geometric Data object into node attributes, edge attributes, and other attributes.

    Args:
        data: PyTorch Geometric Data object.

    Returns:
        tuple: (node_keys, edge_keys, other_keys) where each is a list of attribute names.
    """
    node_keys = []
    edge_keys = []
    other_keys = []

    num_nodes = graph.num_nodes
    num_edges = graph.num_edges

    for key in graph.keys():
        value = graph[key]
        if key == "edge_index":
            other_keys.append(key)
            continue
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            first_dim = value.size(0)
            if first_dim == num_nodes:
                node_keys.append(key)
            elif first_dim == num_edges:
                edge_keys.append(key)
            else:
                other_keys.append(key)
        else:
            other_keys.append(key)

    return node_keys, edge_keys, other_keys


def create_line_graph(
    g: Data,
    threebody_cutoff: float,
    directed: bool = False,
    error_handling: bool = False,
    numerical_noise: float = 1e-6,
) -> Data:
    """
    Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph
        threebody_cutoff (float): cutoff for three-body interactions
        directed (bool): Whether to create a directed line graph, or an M3gnet 3body line graph
            Default = False (M3Gnet)
        error_handling: whether to handle exception due to numerical error
            Default = False
        numerical_noise: a tiny noise added to lg construction to avoid numerical error
            Default = 1e-7

    Returns:
        l_g: DGL graph containing three body information from graph
    """
    if error_handling:
        graph_with_three_body = prune_edges_by_features(
            g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff
        )
        try:
            lg = (
                _create_directed_line_graph(graph_with_three_body)
                if directed
                else _compute_3body(graph_with_three_body)
            )
            return lg
        except Exception as e:
            # Print a warning if the first attempt fails
            warnings.warn(
                f"Initial line graph creation failed with error: {e}. "
                f"Adding numerical noise ({numerical_noise}) to threebody_cutoff and retrying.",
                RuntimeWarning,
                stacklevel=2,
            )
            graph_with_three_body = prune_edges_by_features(
                g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff + numerical_noise
            )
            lg = (
                _create_directed_line_graph(graph_with_three_body)
                if directed
                else _compute_3body(graph_with_three_body)
            )
            return lg
    else:
        graph_with_three_body = prune_edges_by_features(
            g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff
        )
        lg = _create_directed_line_graph(graph_with_three_body) if directed else _compute_3body(graph_with_three_body)
        return lg


def ensure_line_graph_compatibility(
    graph: Data,
    line_graph: Data,
    threebody_cutoff: float,
    directed: bool = False,
    tol: float = 5e-6,
) -> Data:
    """Ensure that line graph is compatible with graph.

    Sets edge data in line graph to be consistent with graph. The line graph is updated in place.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
        directed (bool): Whether to create a directed line graph, or an m3gnet 3body line graph (default: False, m3gnet)
        tol: numerical tolerance for cutoff
    """
    if directed:
        line_graph = _ensure_directed_line_graph_compatibility(graph, line_graph, threebody_cutoff, tol)
    else:
        line_graph = _ensure_3body_line_graph_compatibility(graph, line_graph, threebody_cutoff)

    return line_graph


def prune_edges_by_features(
    graph: Data,
    feat_name: str,
    condition: Callable[[torch.Tensor], torch.Tensor],
    keep_ndata: bool = False,
    keep_edata: bool = True,
    return_keys: bool = False,
    *args,
    **kwargs,
) -> Data:
    """Removes edges graph that do satisfy given condition based on a specified feature value.

    Returns a new graph with edges removed.

    Args:
        graph: DGL graph
        feat_name: edge field name
        condition: condition function. Must be a function where the first is the value
            of the edge field data and returns a Tensor of boolean values.
        keep_ndata: whether to keep node features
        keep_edata: whether to keep edge features
        *args: additional arguments to pass to condition function
        **kwargs: additional keyword arguments to pass to condition function

    Returns: dgl.Graph with removed edges.
    """
    if feat_name not in graph.keys():
        raise ValueError(f"Edge field {feat_name} not an edge feature in given graph.")

    valid_edges = torch.logical_not(condition(graph[feat_name], *args, **kwargs))

    edge_index = graph.edge_index[:, valid_edges]
    e_ids = valid_edges.nonzero(as_tuple=False).squeeze()

    new_graph = Data(edge_index=edge_index, num_nodes=graph.num_nodes, device=edge_index.device)
    new_graph.edge_ids = e_ids

    node_keys, edge_keys, _ = separate_node_edge_keys(graph)

    if keep_ndata:
        for key in node_keys:
            new_graph[key] = graph[key]

    if keep_edata:
        for key in edge_keys:
            new_graph[key] = graph[key][valid_edges]
    if return_keys:
        return new_graph, node_keys, edge_keys
    return new_graph


def _compute_3body(graph: Data) -> Data:
    """
    Calculate the three-body indices from pair atom indices using a PyTorch Geometric Data object.

    Args:
        graph: PyTorch Geometric Data object containing edge_index, bond_dist, bond_vec, and pbc_offset.

    Returns:
        Tuple containing:
        - l_graph: PyTorch Geometric Data object containing three-body information (line graph).
        - triple_bond_indices: NumPy array of bond indices that form three-body interactions.
        - n_triple_ij: NumPy array of number of three-body angles for each bond.
        - n_triple_i: NumPy array of number of three-body angles for each atom.
        - n_triple_s: NumPy array of number of three-body angles for each structure.
    """
    # Number of atoms (nodes)
    n_atoms = graph.num_nodes

    # Get source node indices from edge_index
    first_col = graph.edge_index[0].cpu().numpy()
    # Count bonds per atom
    n_bond_per_atom = np.bincount(first_col, minlength=n_atoms)

    # Compute three-body statistics
    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)  # Three-body angles per atom
    n_triple = n_triple_i.sum()  # Total three-body angles
    n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)  # Three-body angles per bond
    n_triple_s = np.array([n_triple])  # Total three-body angles per structure (single graph)

    # Compute triple bond indices
    triple_bond_indices = np.empty((n_triple, 2), dtype=np.int64)
    start = 0
    cs = 0
    for n in n_bond_per_atom:
        if n > 0:
            r = np.arange(n)
            x, y = np.meshgrid(r, r, indexing="xy")
            final = np.stack([y.ravel(), x.ravel()], axis=1)
            mask = final[:, 0] != final[:, 1]  # Exclude self-loops
            final = final[mask]
            triple_bond_indices[start : start + n * (n - 1)] = final + cs
            start += n * (n - 1)
            cs += n

    # Create line graph edge_index
    src_id = torch.tensor(triple_bond_indices[:, 0], dtype=torch.long, device=graph.edge_index.device)
    dst_id = torch.tensor(triple_bond_indices[:, 1], dtype=torch.long, device=graph.edge_index.device)
    l_edge_index = torch.stack([src_id, dst_id], dim=0)

    # Create line graph Data object
    l_graph = Data(
        edge_index=l_edge_index,
        num_nodes=max(src_id.max().item(), dst_id.max().item()) + 1 if src_id.numel() > 0 else 0,
    )

    # Transfer attributes from original graph to line graph
    three_body_id = l_edge_index.view(-1)
    max_three_body_id = three_body_id.max().item() + 1 if three_body_id.numel() > 0 else 0

    l_graph.bond_dist = graph.bond_dist[:max_three_body_id]
    l_graph.bond_vec = graph.bond_vec[:max_three_body_id]
    l_graph.pbc_offset = graph.pbc_offset[:max_three_body_id] if hasattr(graph, "pbc_offset") else None
    l_graph.n_triple_ij = torch.tensor(
        n_triple_ij[:max_three_body_id], dtype=torch.long, device=graph.edge_index.device
    )

    return l_graph


def _create_directed_line_graph(
    graph: Data,
) -> Data:
    """Creates a line graph from a graph, considers periodic boundary conditions.

    Args:
        graph: DGL graph representing atom graph

    Returns:
        line_graph: DGL line graph of pruned graph to three body cutoff
    """
    with torch.no_grad():
        src_indices, dst_indices = graph.edges()
        images = graph.edata["pbc_offset"]
        all_indices = torch.arange(graph.number_of_nodes(), device=graph.device).unsqueeze(dim=0)
        num_bonds_per_atom = torch.count_nonzero(src_indices.unsqueeze(dim=1) == all_indices, dim=0)
        num_edges_per_bond = (num_bonds_per_atom - 1).repeat_interleave(num_bonds_per_atom)
        lg_src = torch.empty(num_edges_per_bond.sum(), dtype=matgl.int_th, device=graph.device)  # type:ignore[call-overload]
        lg_dst = torch.empty(num_edges_per_bond.sum(), dtype=matgl.int_th, device=graph.device)  # type:ignore[call-overload]

        incoming_edges = src_indices.unsqueeze(1) == dst_indices
        is_self_edge = src_indices == dst_indices
        not_self_edge = ~is_self_edge

        n = 0
        # create line graph edges for bonds that are self edges in atom graph
        if is_self_edge.any():
            edge_inds_s = is_self_edge.nonzero()
            lg_dst_s = edge_inds_s.repeat_interleave(num_edges_per_bond[is_self_edge] + 1)
            lg_src_s = incoming_edges[is_self_edge].nonzero()[:, 1].squeeze()
            lg_src_s = lg_src_s[lg_src_s != lg_dst_s]
            lg_dst_s = edge_inds_s.repeat_interleave(num_edges_per_bond[is_self_edge])
            n = len(lg_dst_s)
            lg_src[:n], lg_dst[:n] = lg_src_s, lg_dst_s

        # create line graph edges for bonds that are not self edges in atom graph
        shared_src = src_indices.unsqueeze(1) == src_indices
        back_tracking = (dst_indices.unsqueeze(1) == src_indices) & torch.all(-images.unsqueeze(1) == images, axis=2)  # type:ignore[call-overload]
        incoming = incoming_edges & (shared_src | ~back_tracking)

        edge_inds_ns = not_self_edge.nonzero().squeeze()
        lg_src_ns = incoming[not_self_edge].nonzero()[:, 1].squeeze()
        lg_dst_ns = edge_inds_ns.repeat_interleave(num_edges_per_bond[not_self_edge])
        lg_src[n:], lg_dst[n:] = lg_src_ns, lg_dst_ns
        lg = dgl.graph((lg_src, lg_dst))

        for key in graph.edata:
            lg.ndata[key] = graph.edata[key][: lg.number_of_nodes()]

        # we need to store the sign of bond vector when a bond is a src node in the line
        # graph in order to appropriately calculate angles when self edges are involved
        lg.ndata["src_bond_sign"] = torch.ones(
            (lg.number_of_nodes(), 1), dtype=lg.ndata["bond_vec"].dtype, device=lg.device
        )
        # if we flip self edges then we need to correct computed angles by pi - angle
        # lg.ndata["src_bond_sign"][edge_inds_s] = -lg.ndata["src_bond_sign"][edge_ind_s]
        # find the intersection for the rare cases where not all edges end up as nodes in the line graph
        all_ns, counts = torch.cat([torch.arange(lg.number_of_nodes(), device=graph.device), edge_inds_ns]).unique(
            return_counts=True
        )
        lg_inds_ns = all_ns[torch.where(counts > 1)]
        lg.ndata["src_bond_sign"][lg_inds_ns] = -lg.ndata["src_bond_sign"][lg_inds_ns]

    return lg


def _ensure_3body_line_graph_compatibility(graph: Data, line_graph: Data, threebody_cutoff: float):
    """Ensure that 3body line graph is compatible with a given graph.

    Sets edge data in line graph to be consistent with graph. The line graph is updated in place.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
    """
    valid_three_body = graph.edata["bond_dist"] <= threebody_cutoff
    if line_graph.num_nodes() == graph.edata["bond_vec"][valid_three_body].shape[0]:
        line_graph.ndata["bond_vec"] = graph.edata["bond_vec"][valid_three_body]
        line_graph.ndata["bond_dist"] = graph.edata["bond_dist"][valid_three_body]
        line_graph.ndata["pbc_offset"] = graph.edata["pbc_offset"][valid_three_body]
    else:
        three_body_id = torch.concatenate(line_graph.edges())
        max_three_body_id = torch.max(three_body_id) + 1 if three_body_id.numel() > 0 else 0
        line_graph.ndata["bond_vec"] = graph.edata["bond_vec"][:max_three_body_id]
        line_graph.ndata["bond_dist"] = graph.edata["bond_dist"][:max_three_body_id]
        line_graph.ndata["pbc_offset"] = graph.edata["pbc_offset"][:max_three_body_id]

    return line_graph


def _ensure_directed_line_graph_compatibility(
    graph: Data, line_graph: Data, threebody_cutoff: float, tol: float = 5e-6
) -> Data:
    """Ensure that line graph is compatible with graph.

    Sets edge data in line graph to be consistent with graph. The line graph is updated in place.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
        tol: numerical tolerance for cutoff
    """
    valid_edges = graph.edata["bond_dist"] <= threebody_cutoff

    # this means there probably is a bond that is just at the cutoff
    # this should only really occur when batching graphs
    if line_graph.number_of_nodes() > sum(valid_edges):
        valid_edges = graph.edata["bond_dist"] <= threebody_cutoff + tol

    # check again and raise if invalid
    if line_graph.number_of_nodes() > sum(valid_edges):
        raise RuntimeError("Line graph is not compatible with graph.")

    edge_ids = valid_edges.nonzero().squeeze()[: line_graph.number_of_nodes()]
    line_graph.ndata["edge_ids"] = edge_ids

    for key in graph.edata:
        line_graph.ndata[key] = graph.edata[key][edge_ids]

    src_indices, dst_indices = graph.edges()
    ns_edge_ids = (src_indices[edge_ids] != dst_indices[edge_ids]).nonzero().squeeze()
    line_graph.ndata["src_bond_sign"] = torch.ones(
        (line_graph.number_of_nodes(), 1), dtype=graph.edata["bond_vec"].dtype, device=line_graph.device
    )
    line_graph.ndata["src_bond_sign"][ns_edge_ids] = -line_graph.ndata["src_bond_sign"][ns_edge_ids]

    return line_graph
