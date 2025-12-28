"""
TCN (Temporal Convolutional Network) Pipeline Service
Performs temporal analysis of cow gait patterns for lameness detection.

Key Features:
- Dilated 1D convolutions for large receptive field
- Processes pose trajectories and silhouette signals
- Outputs severity score with uncertainty estimation via MC Dropout
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.utils.nats_client import NATSClient


class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding for temporal causality"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    Temporal block with residual connection.
    
    Architecture:
    - Two causal convolutions with dilation
    - Weight normalization
    - ReLU activation
    - Dropout for regularization
    - Residual connection
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channels differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        residual = self.residual(x)
        return self.relu(out + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for gait analysis.
    
    Features:
    - Stack of temporal blocks with exponentially increasing dilation
    - Large receptive field to capture 5s of gait patterns
    - MC Dropout for uncertainty estimation
    """
    
    def __init__(self, 
                 input_dim: int = 40,  # Number of input features per timestep
                 hidden_channels: List[int] = [64, 64, 64, 64],
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 num_classes: int = 1):  # Binary: lameness score
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        
        # Build temporal blocks
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_channels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, ...
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
            in_channels = out_channels
        
        self.network = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid()  # Output probability
        )
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field(kernel_size, len(hidden_channels))
    
    def _calculate_receptive_field(self, kernel_size: int, num_layers: int) -> int:
        """Calculate the receptive field of the TCN"""
        # For each layer: receptive_field += (kernel_size - 1) * dilation
        # Total: 1 + sum((k-1) * 2^i for i in range(num_layers * 2))
        rf = 1
        for i in range(num_layers):
            dilation = 2 ** i
            rf += 2 * (kernel_size - 1) * dilation
        return rf
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, time, features)
        
        Returns:
            Lameness probability (batch, 1)
        """
        # Transpose to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply temporal blocks
        out = self.network(x)
        
        # Classification
        out = self.classifier(out)
        
        return out
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                  n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with MC Dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes for MC Dropout
        
        Returns:
            mean_prediction: Mean of predictions
            std_prediction: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()
        return mean_pred, std_pred


class TCNPipeline:
    """TCN Pipeline Service for lameness prediction from temporal data"""
    
    # Feature configuration
    NUM_KEYPOINTS = 20  # From T-LEAP
    FEATURES_PER_KEYPOINT = 2  # x, y normalized coordinates
    EXTRA_FEATURES = 4  # centroid_x, centroid_y, bbox_area, velocity
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = Path("/app/shared/models/tcn")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self._load_model()
        
        # Directories
        self.results_dir = Path("/app/data/results/tcn")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_model(self):
        """Load or initialize TCN model"""
        input_dim = self.NUM_KEYPOINTS * self.FEATURES_PER_KEYPOINT + self.EXTRA_FEATURES
        
        self.model = TCN(
            input_dim=input_dim,
            hidden_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.2
        ).to(self.device)
        
        # Try to load pretrained weights
        weights_path = self.model_path / "tcn_lameness.pt"
        if weights_path.exists():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"✅ Loaded TCN weights from {weights_path}")
            except Exception as e:
                print(f"⚠️ Failed to load weights: {e}")
        else:
            print("⚠️ No pretrained TCN weights found. Using random initialization.")
        
        self.model.eval()
        print(f"TCN receptive field: {self.model.receptive_field} timesteps")
    
    def extract_features_from_tleap(self, tleap_data: Dict) -> Optional[np.ndarray]:
        """
        Extract time-series features from T-LEAP pose sequences.
        
        Returns:
            Feature array of shape (time, features)
        """
        pose_sequences = tleap_data.get("pose_sequences", [])
        
        if not pose_sequences:
            return None
        
        features = []
        
        for frame_data in pose_sequences:
            frame_features = []
            
            # Keypoint positions (normalized to [0, 1])
            keypoints = frame_data.get("keypoints", [])
            bbox = frame_data.get("bbox", [0, 0, 100, 100])
            
            # Normalize coordinates relative to bbox
            bbox_x, bbox_y = bbox[0], bbox[1]
            bbox_w = bbox[2] - bbox[0] if len(bbox) > 2 else 100
            bbox_h = bbox[3] - bbox[1] if len(bbox) > 3 else 100
            
            for kp in keypoints[:self.NUM_KEYPOINTS]:
                x = (kp.get("x", 0) - bbox_x) / max(bbox_w, 1)
                y = (kp.get("y", 0) - bbox_y) / max(bbox_h, 1)
                frame_features.extend([x, y])
            
            # Pad if fewer keypoints
            while len(frame_features) < self.NUM_KEYPOINTS * self.FEATURES_PER_KEYPOINT:
                frame_features.extend([0.0, 0.0])
            
            # Extra features
            centroid_x = (bbox[0] + bbox[2]) / 2 if len(bbox) > 2 else 0
            centroid_y = (bbox[1] + bbox[3]) / 2 if len(bbox) > 3 else 0
            bbox_area = bbox_w * bbox_h
            
            # Normalize extra features
            frame_features.append(centroid_x / 1280)  # Assume 1280 width
            frame_features.append(centroid_y / 720)   # Assume 720 height
            frame_features.append(bbox_area / (1280 * 720))
            
            # Velocity (will compute after)
            frame_features.append(0.0)
            
            features.append(frame_features)
        
        features = np.array(features, dtype=np.float32)
        
        # Compute velocity from centroid positions
        if len(features) > 1:
            centroid_x = features[:, -4]
            velocities = np.zeros(len(features))
            velocities[1:] = np.diff(centroid_x)
            features[:, -1] = velocities
        
        return features
    
    def pad_or_truncate(self, features: np.ndarray, target_length: int = 125) -> np.ndarray:
        """Pad or truncate features to fixed length (5s at 25 FPS = 125 frames)"""
        current_length = features.shape[0]
        
        if current_length >= target_length:
            # Center crop
            start = (current_length - target_length) // 2
            return features[start:start + target_length]
        else:
            # Pad with zeros
            pad_before = (target_length - current_length) // 2
            pad_after = target_length - current_length - pad_before
            return np.pad(features, ((pad_before, pad_after), (0, 0)), mode='constant')
    
    async def process_video(self, video_data: dict):
        """Process video through TCN pipeline"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"TCN pipeline processing video {video_id}")
        
        try:
            # Load T-LEAP results
            tleap_path = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
            if not tleap_path.exists():
                print(f"  No T-LEAP results found for {video_id}")
                return
            
            with open(tleap_path) as f:
                tleap_data = json.load(f)
            
            # Extract features
            features = self.extract_features_from_tleap(tleap_data)
            if features is None or len(features) == 0:
                print(f"  No features extracted for {video_id}")
                return
            
            # Pad/truncate to fixed length
            features = self.pad_or_truncate(features, target_length=125)
            
            # Convert to tensor
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predict with uncertainty
            mean_pred, std_pred = self.model.predict_with_uncertainty(x, n_samples=10)
            
            severity_score = float(mean_pred[0, 0].cpu().numpy())
            uncertainty = float(std_pred[0, 0].cpu().numpy())
            
            # Save results
            results = {
                "video_id": video_id,
                "pipeline": "tcn",
                "severity_score": severity_score,
                "uncertainty": uncertainty,
                "prediction": int(severity_score > 0.5),
                "confidence": 1.0 - uncertainty,
                "input_frames": features.shape[0],
                "input_features": features.shape[1],
                "model_receptive_field": self.model.receptive_field
            }
            
            results_file = self.results_dir / f"{video_id}_tcn.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            await self.nats_client.publish(
                self.config.get("nats", {}).get("subjects", {}).get("pipeline_tcn", "pipeline.tcn"),
                {
                    "video_id": video_id,
                    "pipeline": "tcn",
                    "results_path": str(results_file),
                    "severity_score": severity_score,
                    "uncertainty": uncertainty
                }
            )
            
            print(f"  ✅ TCN completed: score={severity_score:.3f}, uncertainty={uncertainty:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error in TCN pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the TCN pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to T-LEAP results
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_tleap", "pipeline.tleap"
        )
        print(f"TCN pipeline subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        print("=" * 60)
        print("TCN Pipeline Service Started")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: TCN with {len(self.model.hidden_channels)} layers")
        print(f"Receptive field: {self.model.receptive_field} timesteps")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = TCNPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

