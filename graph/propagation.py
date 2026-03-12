"""
Propagation Graph Analysis for Fake News Detection.

Models how a news article spreads through a social network.
Fake news tends to exhibit: wide-shallow trees, rapid early spread,
low-credibility seed users, and high bot-to-human ratios.

Graph format expected:
    nodes: [{"id": str, "is_bot": bool, "followers": int, "verified": bool, ...}]
    edges: [{"from": str, "to": str, "timestamp": int, "type": "retweet"|"reply"|"quote"}]

Compatible with:
    - FakeNewsNet social context data
    - Twitter Academic API v2
    - Reddit comment trees (PHEME dataset)
"""

import json
import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logger.warning("networkx not installed. Install with: pip install networkx")

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GraphFeatures:
    """Feature vector extracted from a propagation tree."""
    # Structural
    num_nodes: int = 0
    num_edges: int = 0
    max_depth: int = 0
    avg_depth: float = 0.0
    max_breadth: int = 0
    branching_factor: float = 0.0
    # Temporal
    time_to_peak_spread_min: float = 0.0   # minutes from root to max retweets/hour
    total_spread_duration_min: float = 0.0
    early_spread_ratio: float = 0.0        # % of shares in first 30 min
    # User credibility
    bot_ratio: float = 0.0
    verified_ratio: float = 0.0
    avg_follower_count: float = 0.0
    # Graph topology
    density: float = 0.0
    clustering_coeff: float = 0.0
    # Derived score
    fake_propagation_score: float = 0.0    # 0=likely real, 1=likely fake


@dataclass
class PropagationGraph:
    """In-memory representation of a news propagation tree."""
    root_id: str
    nodes: dict = field(default_factory=dict)   # id → attrs
    children: dict = field(default_factory=lambda: defaultdict(list))  # parent → [children]
    edges: list = field(default_factory=list)   # [(from, to, attrs)]
    timestamps: dict = field(default_factory=dict)  # node_id → unix timestamp


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Constructs PropagationGraph from raw node/edge data."""

    @staticmethod
    def from_dict(data: dict) -> PropagationGraph:
        """
        Args:
            data: {
                "root_id": str,
                "nodes": [{"id": str, "is_bot": bool, "followers": int,
                            "verified": bool, "timestamp": int}],
                "edges": [{"from": str, "to": str, "timestamp": int,
                            "type": "retweet"|"reply"|"quote"}]
            }
        """
        g = PropagationGraph(root_id=data["root_id"])

        for node in data.get("nodes", []):
            nid = str(node["id"])
            g.nodes[nid] = node
            if "timestamp" in node:
                g.timestamps[nid] = node["timestamp"]

        for edge in data.get("edges", []):
            src, dst = str(edge["from"]), str(edge["to"])
            g.children[src].append(dst)
            g.edges.append((src, dst, edge))

        return g

    @staticmethod
    def from_jsonl(path: str) -> list[PropagationGraph]:
        graphs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                graphs.append(GraphBuilder.from_dict(obj))
        return graphs

    @staticmethod
    def from_reddit_thread(thread: dict) -> PropagationGraph:
        """
        Parse Reddit API thread format into a PropagationGraph.
        thread["data"]["children"] = top-level comments
        """
        root_id = thread["data"].get("id", "root")
        nodes, edges = [], []

        def _recurse(comment, parent_id):
            cid = comment["data"].get("id", "")
            nodes.append({
                "id": cid,
                "is_bot": False,
                "followers": comment["data"].get("link_karma", 0),
                "verified": False,
                "timestamp": comment["data"].get("created_utc", 0),
                "score": comment["data"].get("score", 0),
            })
            if parent_id:
                edges.append({"from": parent_id, "to": cid, "type": "reply",
                              "timestamp": comment["data"].get("created_utc", 0)})
            for reply in comment["data"].get("replies", {}).get("data", {}).get("children", []):
                if reply.get("kind") == "t1":
                    _recurse(reply, cid)

        for child in thread["data"].get("children", []):
            if child.get("kind") == "t1":
                _recurse(child, root_id)

        return GraphBuilder.from_dict({
            "root_id": root_id,
            "nodes": [{"id": root_id, "is_bot": False, "followers": 0, "verified": False}] + nodes,
            "edges": edges,
        })


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class PropagationAnalyzer:
    """
    Extracts structural, temporal, and credibility features from
    a PropagationGraph and returns a GraphFeatures object.
    """

    def analyze(self, graph: PropagationGraph) -> GraphFeatures:
        feat = GraphFeatures()
        if not graph.nodes:
            return feat

        feat.num_nodes = len(graph.nodes)
        feat.num_edges = len(graph.edges)

        # BFS for depth/breadth stats
        depths = self._bfs_depths(graph)
        if depths:
            depth_vals = list(depths.values())
            feat.max_depth = max(depth_vals)
            feat.avg_depth = sum(depth_vals) / len(depth_vals)
            # Breadth = max nodes at any depth level
            depth_counts = defaultdict(int)
            for d in depth_vals:
                depth_counts[d] += 1
            feat.max_breadth = max(depth_counts.values())

        # Branching factor
        if feat.num_nodes > 1:
            feat.branching_factor = feat.num_edges / (feat.num_nodes - 1)

        # Temporal features
        self._compute_temporal(graph, feat)

        # User credibility
        self._compute_credibility(graph, feat)

        # Graph topology (requires networkx)
        if HAS_NX:
            self._compute_topology(graph, feat)

        # Fake propagation score heuristic
        feat.fake_propagation_score = self._score(feat)

        return feat

    def _bfs_depths(self, graph: PropagationGraph) -> dict:
        depths = {graph.root_id: 0}
        queue = deque([graph.root_id])
        while queue:
            node = queue.popleft()
            for child in graph.children.get(node, []):
                if child not in depths:
                    depths[child] = depths[node] + 1
                    queue.append(child)
        return depths

    def _compute_temporal(self, graph: PropagationGraph, feat: GraphFeatures):
        ts = graph.timestamps
        if len(ts) < 2:
            return
        root_ts = ts.get(graph.root_id, min(ts.values()))
        times = sorted([t for t in ts.values()])
        total_duration = (times[-1] - times[0]) / 60.0  # minutes
        feat.total_spread_duration_min = total_duration

        if total_duration > 0:
            early_cutoff = root_ts + 30 * 60  # 30 minutes
            early_count = sum(1 for t in ts.values() if t <= early_cutoff and t > root_ts)
            feat.early_spread_ratio = early_count / max(len(ts) - 1, 1)

            # Peak spread: bin into 5-min windows, find max activity window
            bins = defaultdict(int)
            for t in ts.values():
                window = int((t - root_ts) / 300)  # 5-min bins
                bins[window] += 1
            peak_window = max(bins, key=bins.__getitem__)
            feat.time_to_peak_spread_min = peak_window * 5.0

    def _compute_credibility(self, graph: PropagationGraph, feat: GraphFeatures):
        nodes = list(graph.nodes.values())
        if not nodes:
            return
        bots = sum(1 for n in nodes if n.get("is_bot", False))
        verified = sum(1 for n in nodes if n.get("verified", False))
        followers = [n.get("followers", 0) for n in nodes]
        feat.bot_ratio = bots / len(nodes)
        feat.verified_ratio = verified / len(nodes)
        feat.avg_follower_count = sum(followers) / len(followers) if followers else 0

    def _compute_topology(self, graph: PropagationGraph, feat: GraphFeatures):
        G = nx.DiGraph()
        for nid in graph.nodes:
            G.add_node(nid)
        for src, dst, _ in graph.edges:
            G.add_edge(src, dst)
        feat.density = nx.density(G)
        # Clustering on undirected version
        ug = G.to_undirected()
        if ug.number_of_nodes() > 0:
            feat.clustering_coeff = nx.average_clustering(ug)

    def _score(self, feat: GraphFeatures) -> float:
        """
        Heuristic fake-propagation score in [0, 1].
        Based on empirical findings from Vosoughi et al. (Science 2018)
        and Shu et al. (SIGKDD 2020):
        - Fake news spreads faster, wider, reaches greater depth
        - Higher bot ratio, lower verified ratio
        - Early burst pattern
        """
        score = 0.0
        weights = {
            "early_spread": (feat.early_spread_ratio, 0.25),
            "bot_ratio": (feat.bot_ratio, 0.25),
            "shallow_wide": (
                min(feat.max_breadth / max(feat.max_depth + 1, 1) / 10, 1.0),
                0.20,
            ),
            "low_verified": (1.0 - feat.verified_ratio, 0.15),
            "low_followers": (
                max(0, 1.0 - math.log10(max(feat.avg_follower_count, 1)) / 6),
                0.15,
            ),
        }
        for _, (value, weight) in weights.items():
            score += value * weight
        return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# GNN-based classifier (PyTorch Geometric) — optional
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True

    class PropagationGNN(torch.nn.Module):
        """
        2-layer GCN that classifies a propagation graph as fake/real.
        Node features: [depth_norm, is_bot, verified, log_followers, timestamp_norm]
        """

        def __init__(self, in_channels: int = 5, hidden: int = 64, num_classes: int = 2):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.classifier = torch.nn.Linear(hidden, num_classes)

        def forward(self, data: "Data"):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.classifier(x)

    def graph_to_pyg(
        graph: PropagationGraph,
        analyzer: PropagationAnalyzer,
        label: Optional[int] = None,
    ) -> "Data":
        """Convert PropagationGraph → PyG Data object."""
        depths = analyzer._bfs_depths(graph)
        max_depth = max(depths.values()) if depths else 1
        node_ids = list(graph.nodes.keys())
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        ts_vals = list(graph.timestamps.values())
        ts_min = min(ts_vals) if ts_vals else 0
        ts_max = max(ts_vals) if ts_vals else 1

        rows = []
        for nid in node_ids:
            n = graph.nodes[nid]
            d = depths.get(nid, 0) / max(max_depth, 1)
            b = float(n.get("is_bot", False))
            v = float(n.get("verified", False))
            f = math.log10(max(n.get("followers", 1), 1)) / 6.0
            t_raw = graph.timestamps.get(nid, ts_min)
            t = (t_raw - ts_min) / max(ts_max - ts_min, 1)
            rows.append([d, b, v, f, t])

        import torch
        x = torch.tensor(rows, dtype=torch.float)
        src = [id_to_idx[e[0]] for e in graph.edges if e[0] in id_to_idx and e[1] in id_to_idx]
        dst = [id_to_idx[e[1]] for e in graph.edges if e[0] in id_to_idx and e[1] in id_to_idx]
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        return data

except ImportError:
    HAS_PYG = False
    logger.info("torch_geometric not installed. GNN classifier unavailable. "
                "Install: pip install torch-geometric")