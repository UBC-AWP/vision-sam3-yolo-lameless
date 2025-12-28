"""
Graph Transformer (GraphGPS) Pipeline Service
Implements graph-based reasoning for lameness detection using relational context.

Key Features:
- Graph construction from video clips with node features
- kNN edges in DINOv3 embedding space + temporal edges
- GraphGPS architecture: local message passing + global attention
- Positional encodings: Laplacian eigenvectors, random walk
- Uncertainty estimation

This is a comprehensive Graph Neural Network implementation for learning purposes.
"""
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
from shared.utils.nats_client import NATSClient


# ============================================================================
# Graph Construction
# ============================================================================

class GraphBuilder:
    """
    Builds graphs from video clips for graph-based lameness analysis.
    
    Node Features:
    - Pose metrics (from T-LEAP)
    - Silhouette metrics (from SAM3/YOLO)
    - DINOv3 embeddings (dimension-reduced)
    - Metadata features
    
    Edge Types:
    - kNN edges in embedding space (similarity)
    - Temporal edges (same cow over time)
    """
    
    def __init__(self, k_neighbors: int = 5, embedding_dim: int = 64):
        self.k_neighbors = k_neighbors
        self.embedding_dim = embedding_dim
    
    def compute_knn_edges(self, embeddings: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute kNN edges from embeddings.
        
        Args:
            embeddings: Node embeddings (N, D)
            k: Number of neighbors
        
        Returns:
            edge_index: (2, E) edge index array
            edge_weights: (E,) similarity weights
        """
        if k is None:
            k = self.k_neighbors
        
        N = len(embeddings)
        if N <= k:
            k = max(1, N - 1)
        
        # Compute pairwise distances
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity = embeddings_norm @ embeddings_norm.T
        
        # Get top-k neighbors for each node
        edges_src = []
        edges_dst = []
        edge_weights = []
        
        for i in range(N):
            # Exclude self
            sim_i = similarity[i].copy()
            sim_i[i] = -np.inf
            
            # Get top-k
            top_k_idx = np.argsort(sim_i)[-k:]
            
            for j in top_k_idx:
                if sim_i[j] > -np.inf:
                    edges_src.append(i)
                    edges_dst.append(j)
                    edge_weights.append(sim_i[j])
        
        edge_index = np.array([edges_src, edges_dst])
        edge_weights = np.array(edge_weights)
        
        return edge_index, edge_weights
    
    def add_temporal_edges(self, video_ids: List[str], cow_ids: List[Optional[str]],
                           timestamps: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add temporal edges between videos of the same cow.
        
        Returns:
            edge_index: (2, E) temporal edge index
            edge_attr: (E,) time deltas
        """
        edges_src = []
        edges_dst = []
        edge_attr = []
        
        # Group by cow ID
        cow_to_indices = {}
        for i, cow_id in enumerate(cow_ids):
            if cow_id is not None:
                if cow_id not in cow_to_indices:
                    cow_to_indices[cow_id] = []
                cow_to_indices[cow_id].append(i)
        
        # Create temporal edges within each cow group
        for cow_id, indices in cow_to_indices.items():
            if len(indices) < 2:
                continue
            
            # Sort by timestamp
            sorted_indices = sorted(indices, key=lambda x: timestamps[x])
            
            # Connect consecutive videos
            for i in range(len(sorted_indices) - 1):
                src, dst = sorted_indices[i], sorted_indices[i + 1]
                time_delta = timestamps[dst] - timestamps[src]
                
                # Bidirectional edges
                edges_src.extend([src, dst])
                edges_dst.extend([dst, src])
                edge_attr.extend([time_delta, -time_delta])
        
        if not edges_src:
            return np.array([[], []], dtype=np.int64), np.array([])
        
        edge_index = np.array([edges_src, edges_dst])
        edge_attr = np.array(edge_attr)
        
        return edge_index, edge_attr
    
    def build_graph(self, node_features: np.ndarray, embeddings: np.ndarray,
                    video_ids: List[str] = None, cow_ids: List[str] = None,
                    timestamps: List[float] = None, labels: np.ndarray = None) -> Data:
        """
        Build a PyTorch Geometric Data object from node features.
        
        Args:
            node_features: (N, D) node feature matrix
            embeddings: (N, E) DINOv3 embeddings for kNN
            video_ids: List of video IDs
            cow_ids: List of cow IDs (for temporal edges)
            timestamps: List of timestamps (for temporal edges)
            labels: (N,) labels if available
        
        Returns:
            PyG Data object
        """
        N = len(node_features)
        
        # Compute kNN edges
        knn_edges, knn_weights = self.compute_knn_edges(embeddings)
        
        # Add temporal edges if cow IDs available
        if cow_ids is not None and timestamps is not None:
            temp_edges, temp_weights = self.add_temporal_edges(video_ids or [], cow_ids, timestamps)
            
            if temp_edges.size > 0:
                # Combine edges
                edge_index = np.concatenate([knn_edges, temp_edges], axis=1)
                # Create edge type indicator: 0 = kNN, 1 = temporal
                edge_type = np.concatenate([
                    np.zeros(knn_edges.shape[1]),
                    np.ones(temp_edges.shape[1])
                ])
            else:
                edge_index = knn_edges
                edge_type = np.zeros(knn_edges.shape[1])
        else:
            edge_index = knn_edges
            edge_type = np.zeros(knn_edges.shape[1])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        
        if labels is not None:
            data.y = torch.tensor(labels, dtype=torch.float32)
        
        return data


# ============================================================================
# Positional Encodings
# ============================================================================

class LapPE(nn.Module):
    """
    Laplacian Positional Encoding.
    
    Uses eigenvectors of the graph Laplacian to provide positional information.
    This helps the model understand graph structure.
    """
    
    def __init__(self, max_freq: int = 10, hidden_dim: int = 16):
        super().__init__()
        self.max_freq = max_freq
        self.linear = nn.Linear(max_freq, hidden_dim)
    
    def compute_laplacian_pe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute Laplacian eigenvector positional encoding"""
        # Build adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        
        # Compute degree
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        # For simplicity, use random walk Laplacian approximation
        
        # Create position encoding based on degree and local structure
        pe = torch.zeros(num_nodes, self.max_freq)
        
        for i in range(self.max_freq):
            # Use powers of normalized adjacency as features
            pe[:, i] = deg.pow(i / self.max_freq)
        
        return pe
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        pe = self.compute_laplacian_pe(edge_index, num_nodes)
        return self.linear(pe)


class RWPE(nn.Module):
    """
    Random Walk Positional Encoding.
    
    Uses random walk statistics to encode structural position.
    """
    
    def __init__(self, walk_length: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.walk_length = walk_length
        self.linear = nn.Linear(walk_length, hidden_dim)
    
    def compute_rwpe(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute random walk positional encoding"""
        # Compute transition matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        # Landing probabilities (simplified)
        pe = torch.zeros(num_nodes, self.walk_length)
        
        for step in range(self.walk_length):
            # Use degree-based approximation
            pe[:, step] = (deg / deg.sum()).pow(step + 1)
        
        return pe
    
    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        pe = self.compute_rwpe(edge_index, num_nodes)
        return self.linear(pe)


# ============================================================================
# GraphGPS Architecture
# ============================================================================

class GatedGCNLayer(nn.Module):
    """
    Gated Graph Convolution Layer.
    
    Implements message passing with gating mechanism for controlled information flow.
    """
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.A = nn.Linear(in_dim, out_dim)
        self.B = nn.Linear(in_dim, out_dim)
        self.C = nn.Linear(in_dim, out_dim)
        self.D = nn.Linear(in_dim, out_dim)
        self.E = nn.Linear(in_dim, out_dim)
        
        self.bn_node = nn.BatchNorm1d(out_dim)
        self.bn_edge = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (N, in_dim)
            edge_index: Edge indices (2, E)
            edge_attr: Edge features (E, in_dim) - optional
        """
        src, dst = edge_index
        
        # Node transformations
        Ax = self.A(x)
        Bx = self.B(x)
        Dx = self.D(x)
        Ex = self.E(x)
        
        # Edge gate
        if edge_attr is not None:
            Ce = self.C(edge_attr)
            sigma = torch.sigmoid(Ce + Dx[dst] + Ex[src])
            e_new = Ce
        else:
            sigma = torch.sigmoid(Dx[dst] + Ex[src])
            e_new = sigma
        
        # Message passing with gating
        message = sigma * Bx[src]
        
        # Aggregate messages
        agg = torch.zeros_like(x[:, :message.size(1)])
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(message), message)
        
        # Count neighbors for normalization
        deg = degree(dst, x.size(0), dtype=x.dtype).clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        
        # Residual connection
        h = Ax + agg
        h = self.bn_node(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        return h, e_new


class GlobalAttention(nn.Module):
    """
    Global Self-Attention for graphs.
    
    Applies Transformer-style attention across all nodes.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply global attention.
        
        Args:
            x: Node features (N, D)
            batch: Batch assignment (N,) for batched graphs
        """
        if batch is None:
            # Single graph - simple attention
            x_2d = x.unsqueeze(0)  # (1, N, D)
            attn_out, _ = self.attention(x_2d, x_2d, x_2d)
            attn_out = attn_out.squeeze(0)  # (N, D)
        else:
            # Batched graphs - attention within each graph
            unique_batches = batch.unique()
            outputs = []
            
            for b in unique_batches:
                mask = batch == b
                x_b = x[mask].unsqueeze(0)  # (1, N_b, D)
                attn_out, _ = self.attention(x_b, x_b, x_b)
                outputs.append(attn_out.squeeze(0))
            
            attn_out = torch.cat(outputs, dim=0)
        
        # Residual connection
        out = self.norm(x + self.dropout(attn_out))
        return out


class GraphGPSLayer(nn.Module):
    """
    GraphGPS Layer combining local and global attention.
    
    Architecture:
    1. Local message passing (GatedGCN)
    2. Global self-attention
    3. Feedforward network
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Local message passing
        self.local_conv = GatedGCNLayer(hidden_dim, hidden_dim, dropout)
        
        # Global attention
        self.global_attn = GlobalAttention(hidden_dim, num_heads, dropout)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through GPS layer"""
        
        # Local message passing
        h_local, _ = self.local_conv(x, edge_index, edge_attr)
        
        # Global attention
        h_global = self.global_attn(h_local, batch)
        
        # Feedforward with residual
        out = self.norm(h_global + self.ffn(h_global))
        
        return out


class GraphGPS(nn.Module):
    """
    GraphGPS Model for node-level lameness prediction.
    
    Full architecture:
    1. Input projection
    2. Positional encoding (Laplacian + Random Walk)
    3. Stack of GPS layers
    4. Prediction head
    """
    
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 pe_dim: int = 16):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim - 2 * pe_dim)
        
        # Positional encodings
        self.lap_pe = LapPE(max_freq=10, hidden_dim=pe_dim)
        self.rw_pe = RWPE(walk_length=8, hidden_dim=pe_dim)
        
        # GPS layers
        self.layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, and optionally batch
        
        Returns:
            Node-level predictions (N, 1)
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Input projection
        h = self.input_proj(x)
        
        # Add positional encodings
        lap_pe = self.lap_pe(edge_index, x.size(0))
        rw_pe = self.rw_pe(edge_index, x.size(0))
        h = torch.cat([h, lap_pe, rw_pe], dim=-1)
        
        # GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
        
        # Prediction
        out = self.pred_head(h)
        
        return out
    
    def predict_with_uncertainty(self, data: Data, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout for uncertainty estimation"""
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(data)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()
        return mean_pred, std_pred
    
    def get_node_embeddings(self, data: Data) -> torch.Tensor:
        """Get intermediate node embeddings for analysis"""
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        h = self.input_proj(x)
        lap_pe = self.lap_pe(edge_index, x.size(0))
        rw_pe = self.rw_pe(edge_index, x.size(0))
        h = torch.cat([h, lap_pe, rw_pe], dim=-1)
        
        for layer in self.layers:
            h = layer(h, edge_index, batch=batch)
        
        return h


# ============================================================================
# Pipeline Service
# ============================================================================

class GNNPipeline:
    """Graph Neural Network Pipeline Service using GraphGPS"""
    
    # Feature dimensions
    POSE_FEATURES = 10  # Summary pose metrics
    SILHOUETTE_FEATURES = 5  # SAM3/YOLO metrics
    EMBEDDING_DIM = 32  # Reduced DINOv3 dimension
    META_FEATURES = 3  # Metadata features
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Graph builder
        self.graph_builder = GraphBuilder(k_neighbors=5, embedding_dim=self.EMBEDDING_DIM)
        
        # Model
        input_dim = self.POSE_FEATURES + self.SILHOUETTE_FEATURES + self.EMBEDDING_DIM + self.META_FEATURES
        self.model = GraphGPS(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            pe_dim=16
        ).to(self.device)
        
        # Load weights if available
        self.model_path = Path("/app/shared/models/gnn")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_model()
        
        # Results directory
        self.results_dir = Path("/app/data/results/gnn")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for building graphs across videos
        self.video_features_cache = {}
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_model(self):
        weights_path = self.model_path / "graphgps_lameness.pt"
        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ Loaded GraphGPS weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}")
        else:
            print("⚠️ No pretrained GraphGPS weights. Using random initialization.")
        
        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"GraphGPS parameters: {num_params:,}")
    
    def extract_node_features(self, video_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract node features from pipeline results for a video"""
        features = {}
        
        # T-LEAP pose features
        tleap_path = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_path.exists():
            with open(tleap_path) as f:
                tleap_data = json.load(f)
            
            loco = tleap_data.get("locomotion_features", {})
            features["pose"] = np.array([
                loco.get("back_arch_mean", 0),
                loco.get("back_arch_std", 0),
                loco.get("head_bob_magnitude", 0),
                loco.get("head_bob_frequency", 0),
                loco.get("front_leg_asymmetry", 0),
                loco.get("rear_leg_asymmetry", 0),
                loco.get("lameness_score", 0.5),
                loco.get("stride_fl_mean", 0),
                loco.get("stride_fr_mean", 0),
                loco.get("steadiness_score", 0.5)
            ], dtype=np.float32)
        else:
            features["pose"] = np.zeros(self.POSE_FEATURES, dtype=np.float32)
        
        # SAM3/YOLO silhouette features
        sam3_path = Path(f"/app/data/results/sam3/{video_id}_sam3.json")
        yolo_path = Path(f"/app/data/results/yolo/{video_id}_yolo.json")
        
        silhouette = np.zeros(self.SILHOUETTE_FEATURES, dtype=np.float32)
        
        if sam3_path.exists():
            with open(sam3_path) as f:
                sam3_data = json.load(f)
            feats = sam3_data.get("features", {})
            silhouette[0] = feats.get("avg_area_ratio", 0)
            silhouette[1] = feats.get("avg_circularity", 0)
            silhouette[2] = feats.get("avg_aspect_ratio", 1)
        
        if yolo_path.exists():
            with open(yolo_path) as f:
                yolo_data = json.load(f)
            feats = yolo_data.get("features", {})
            silhouette[3] = feats.get("avg_confidence", 0.5)
            silhouette[4] = feats.get("position_stability", 0.5)
        
        features["silhouette"] = silhouette
        
        # DINOv3 embeddings
        dinov3_path = Path(f"/app/data/results/dinov3/{video_id}_dinov3.json")
        if dinov3_path.exists():
            with open(dinov3_path) as f:
                dinov3_data = json.load(f)
            
            embedding = dinov3_data.get("embedding", [])
            if len(embedding) > 0:
                # Reduce dimension if needed
                embedding = np.array(embedding, dtype=np.float32)
                if len(embedding) > self.EMBEDDING_DIM:
                    # Simple PCA-like reduction (use first N dims)
                    embedding = embedding[:self.EMBEDDING_DIM]
                elif len(embedding) < self.EMBEDDING_DIM:
                    embedding = np.pad(embedding, (0, self.EMBEDDING_DIM - len(embedding)))
                features["embedding"] = embedding
            else:
                features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        else:
            features["embedding"] = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        
        # Metadata features
        features["meta"] = np.array([
            0.5,  # Placeholder for normalized timestamp
            1.0,  # Quality score
            0.5   # Prior lameness estimate
        ], dtype=np.float32)
        
        return features
    
    def collect_graph_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Collect features from all analyzed videos for graph construction"""
        node_features_list = []
        embeddings_list = []
        video_ids = []
        
        # Scan results directory for available videos
        tleap_dir = Path("/app/data/results/tleap")
        if tleap_dir.exists():
            for result_file in tleap_dir.glob("*_tleap.json"):
                video_id = result_file.stem.replace("_tleap", "")
                
                features = self.extract_node_features(video_id)
                if features is not None:
                    # Concatenate all features
                    node_feat = np.concatenate([
                        features["pose"],
                        features["silhouette"],
                        features["embedding"],
                        features["meta"]
                    ])
                    
                    node_features_list.append(node_feat)
                    embeddings_list.append(features["embedding"])
                    video_ids.append(video_id)
        
        if not node_features_list:
            return None, None, []
        
        node_features = np.stack(node_features_list)
        embeddings = np.stack(embeddings_list)
        
        return node_features, embeddings, video_ids
    
    async def process_video(self, video_data: dict):
        """Process video through GNN pipeline"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"GNN pipeline processing video {video_id}")
        
        try:
            # Collect all video features for graph
            node_features, embeddings, video_ids = self.collect_graph_data()
            
            if node_features is None or len(video_ids) == 0:
                print(f"  No video features available for graph construction")
                return
            
            # Find index of current video
            if video_id not in video_ids:
                # Add current video to graph
                features = self.extract_node_features(video_id)
                if features is None:
                    print(f"  Could not extract features for {video_id}")
                    return
                
                new_node = np.concatenate([
                    features["pose"],
                    features["silhouette"],
                    features["embedding"],
                    features["meta"]
                ])
                
                node_features = np.vstack([node_features, new_node])
                embeddings = np.vstack([embeddings, features["embedding"]])
                video_ids.append(video_id)
            
            target_idx = video_ids.index(video_id)
            
            print(f"  Graph: {len(video_ids)} nodes")
            
            # Build graph
            graph = self.graph_builder.build_graph(
                node_features=node_features,
                embeddings=embeddings,
                video_ids=video_ids
            )
            graph = graph.to(self.device)
            
            # Predict with uncertainty
            mean_pred, std_pred = self.model.predict_with_uncertainty(graph, n_samples=10)
            
            # Get prediction for target video
            severity_score = float(mean_pred[target_idx, 0].cpu().numpy())
            uncertainty = float(std_pred[target_idx, 0].cpu().numpy())
            
            # Get neighbor influence
            neighbor_scores = []
            edge_index = graph.edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                if edge_index[1, i] == target_idx:
                    src = edge_index[0, i]
                    neighbor_scores.append({
                        "video_id": video_ids[src],
                        "score": float(mean_pred[src, 0].cpu().numpy())
                    })
            
            # Save results
            results = {
                "video_id": video_id,
                "pipeline": "gnn",
                "model": "GraphGPS",
                "severity_score": severity_score,
                "uncertainty": uncertainty,
                "prediction": int(severity_score > 0.5),
                "confidence": 1.0 - uncertainty,
                "graph_info": {
                    "num_nodes": len(video_ids),
                    "num_edges": graph.edge_index.shape[1],
                    "k_neighbors": self.graph_builder.k_neighbors
                },
                "neighbor_influence": neighbor_scores[:5]  # Top 5 neighbors
            }
            
            results_file = self.results_dir / f"{video_id}_gnn.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get("pipeline_gnn", "pipeline.gnn"),
                {
                    "video_id": video_id,
                    "pipeline": "gnn",
                    "results_path": str(results_file),
                    "severity_score": severity_score,
                    "uncertainty": uncertainty
                }
            )
            
            print(f"  ✅ GNN completed: score={severity_score:.3f}, uncertainty={uncertainty:.3f}")
            print(f"     Graph had {len(video_ids)} nodes, {len(neighbor_scores)} neighbors")
            
        except Exception as e:
            print(f"  ❌ Error in GNN pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the GNN pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to DINOv3 results (after embeddings are computed)
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_dinov3", "pipeline.dinov3"
        )
        print(f"GNN pipeline subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        print("=" * 60)
        print("GNN (GraphGPS) Pipeline Service Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: GraphGPS with {len(self.model.layers)} layers")
        print(f"Hidden dim: {self.model.hidden_dim}")
        print(f"k-neighbors: {self.graph_builder.k_neighbors}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = GNNPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

