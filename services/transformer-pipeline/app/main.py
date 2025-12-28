"""
Transformer Pipeline Service
Encoder-only Transformer for temporal gait analysis.

Key Features:
- Self-attention for capturing long-range temporal dependencies
- Positional encoding for sequence awareness
- Attention masking for missing/low-confidence pose points
- MC Dropout for uncertainty estimation
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
from shared.utils.nats_client import NATSClient


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness"""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer encoder layer with pre-norm architecture.
    
    Pre-norm tends to train more stably than post-norm.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            src_key_padding_mask: Mask for padding positions (batch, seq_len)
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, 
                                      key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feedforward
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


class GaitTransformer(nn.Module):
    """
    Transformer Encoder for gait-based lameness detection.
    
    Architecture:
    - Input projection
    - Positional encoding
    - Stack of transformer encoder layers
    - Global pooling
    - Classification head
    """
    
    def __init__(self,
                 input_dim: int = 44,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 150):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head with global pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional padding mask (batch, seq_len), True = masked
        
        Returns:
            Lameness probability (batch, 1)
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Global average pooling (ignoring masked positions)
        if mask is not None:
            # Create inverse mask for averaging
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Classification
        out = self.classifier(x)
        
        return out
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                  mask: Optional[torch.Tensor] = None,
                                  n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with MC Dropout for uncertainty estimation.
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, mask)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()
        return mean_pred, std_pred
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all layers for interpretability.
        
        Returns list of attention weight tensors, one per layer.
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        attention_weights = []
        
        for layer in self.encoder_layers:
            x_norm = layer.norm1(x)
            _, attn_weights = layer.self_attn(x_norm, x_norm, x_norm, 
                                               need_weights=True, average_attn_heads=False)
            attention_weights.append(attn_weights.detach())
            x = layer(x)
        
        return attention_weights


class TransformerPipeline:
    """Transformer Pipeline Service for lameness prediction"""
    
    # Feature configuration (same as TCN for consistency)
    NUM_KEYPOINTS = 20
    FEATURES_PER_KEYPOINT = 2
    EXTRA_FEATURES = 4
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = Path("/app/shared/models/transformer")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
        
        # Directories
        self.results_dir = Path("/app/data/results/transformer")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_model(self):
        """Load or initialize Transformer model"""
        input_dim = self.NUM_KEYPOINTS * self.FEATURES_PER_KEYPOINT + self.EXTRA_FEATURES
        
        self.model = GaitTransformer(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_len=150
        ).to(self.device)
        
        # Try to load pretrained weights
        weights_path = self.model_path / "transformer_lameness.pt"
        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ Loaded Transformer weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}")
        else:
            print("⚠️ No pretrained Transformer weights found. Using random initialization.")
        
        self.model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Transformer parameters: {num_params:,}")
    
    def extract_features_from_tleap(self, tleap_data: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract features and confidence mask from T-LEAP data.
        
        Returns:
            features: Feature array (time, features)
            mask: Confidence mask (time,) - True for low-confidence frames
        """
        pose_sequences = tleap_data.get("pose_sequences", [])
        
        if not pose_sequences:
            return None, None
        
        features = []
        confidences = []
        
        for frame_data in pose_sequences:
            frame_features = []
            frame_confidence = []
            
            keypoints = frame_data.get("keypoints", [])
            bbox = frame_data.get("bbox", [0, 0, 100, 100])
            detection_conf = frame_data.get("detection_confidence", 1.0)
            
            bbox_x, bbox_y = bbox[0], bbox[1]
            bbox_w = bbox[2] - bbox[0] if len(bbox) > 2 else 100
            bbox_h = bbox[3] - bbox[1] if len(bbox) > 3 else 100
            
            for kp in keypoints[:self.NUM_KEYPOINTS]:
                x = (kp.get("x", 0) - bbox_x) / max(bbox_w, 1)
                y = (kp.get("y", 0) - bbox_y) / max(bbox_h, 1)
                conf = kp.get("confidence", 0.5)
                frame_features.extend([x, y])
                frame_confidence.append(conf)
            
            # Pad if fewer keypoints
            while len(frame_features) < self.NUM_KEYPOINTS * self.FEATURES_PER_KEYPOINT:
                frame_features.extend([0.0, 0.0])
                frame_confidence.append(0.0)
            
            # Extra features
            centroid_x = (bbox[0] + bbox[2]) / 2 if len(bbox) > 2 else 0
            centroid_y = (bbox[1] + bbox[3]) / 2 if len(bbox) > 3 else 0
            bbox_area = bbox_w * bbox_h
            
            frame_features.append(centroid_x / 1280)
            frame_features.append(centroid_y / 720)
            frame_features.append(bbox_area / (1280 * 720))
            frame_features.append(0.0)  # Velocity placeholder
            
            features.append(frame_features)
            
            # Average confidence for masking
            avg_conf = np.mean(frame_confidence) * detection_conf
            confidences.append(avg_conf)
        
        features = np.array(features, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        # Compute velocity
        if len(features) > 1:
            centroid_x = features[:, -4]
            velocities = np.zeros(len(features))
            velocities[1:] = np.diff(centroid_x)
            features[:, -1] = velocities
        
        # Create mask: True for low-confidence frames (to be masked)
        mask = confidences < 0.3
        
        return features, mask
    
    def pad_or_truncate(self, features: np.ndarray, mask: np.ndarray,
                        target_length: int = 125) -> Tuple[np.ndarray, np.ndarray]:
        """Pad or truncate to fixed length"""
        current_length = features.shape[0]
        
        if current_length >= target_length:
            start = (current_length - target_length) // 2
            return (features[start:start + target_length], 
                    mask[start:start + target_length])
        else:
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            
            features_padded = np.pad(features, ((pad_before, pad_after), (0, 0)), 
                                      mode='constant')
            mask_padded = np.pad(mask, (pad_before, pad_after), 
                                 mode='constant', constant_values=True)
            
            return features_padded, mask_padded
    
    async def process_video(self, video_data: dict):
        """Process video through Transformer pipeline"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"Transformer pipeline processing video {video_id}")
        
        try:
            # Load T-LEAP results
            tleap_path = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
            if not tleap_path.exists():
                print(f"  No T-LEAP results found for {video_id}")
                return
            
            with open(tleap_path) as f:
                tleap_data = json.load(f)
            
            # Extract features and mask
            features, mask = self.extract_features_from_tleap(tleap_data)
            if features is None or len(features) == 0:
                print(f"  No features extracted for {video_id}")
                return
            
            # Pad/truncate
            features, mask = self.pad_or_truncate(features, mask, target_length=125)
            
            # Convert to tensors
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            
            # Predict with uncertainty
            mean_pred, std_pred = self.model.predict_with_uncertainty(
                x, mask_tensor, n_samples=10
            )
            
            severity_score = float(mean_pred[0, 0].cpu().numpy())
            uncertainty = float(std_pred[0, 0].cpu().numpy())
            
            # Get attention weights for interpretability
            self.model.eval()
            attention_weights = self.model.get_attention_weights(x)
            
            # Compute attention-based saliency (average attention to each timestep)
            avg_attention = attention_weights[-1].mean(dim=(0, 1)).cpu().numpy()
            temporal_saliency = avg_attention.mean(axis=0).tolist()
            
            # Save results
            results = {
                "video_id": video_id,
                "pipeline": "transformer",
                "severity_score": severity_score,
                "uncertainty": uncertainty,
                "prediction": int(severity_score > 0.5),
                "confidence": 1.0 - uncertainty,
                "input_frames": features.shape[0],
                "input_features": features.shape[1],
                "masked_frames": int(mask.sum()),
                "temporal_saliency": temporal_saliency[:20],  # First 20 for brevity
                "model_info": {
                    "d_model": self.model.d_model,
                    "num_layers": len(self.model.encoder_layers),
                    "nhead": 4
                }
            }
            
            results_file = self.results_dir / f"{video_id}_transformer.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get(
                    "pipeline_transformer", "pipeline.transformer"
                ),
                {
                    "video_id": video_id,
                    "pipeline": "transformer",
                    "results_path": str(results_file),
                    "severity_score": severity_score,
                    "uncertainty": uncertainty
                }
            )
            
            print(f"  ✅ Transformer completed: score={severity_score:.3f}, uncertainty={uncertainty:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error in Transformer pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the Transformer pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to T-LEAP results
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_tleap", "pipeline.tleap"
        )
        print(f"Transformer pipeline subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        print("=" * 60)
        print("Transformer Pipeline Service Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: GaitTransformer")
        print(f"d_model: {self.model.d_model}")
        print(f"Layers: {len(self.model.encoder_layers)}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = TransformerPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

