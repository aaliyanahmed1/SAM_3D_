"""Configuration management for SAM 3D."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import torch


@dataclass
class ModelConfig:
    """Model configuration."""
    
    model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    checkpoint_path: Optional[str] = None
    device: str = "auto"  # auto, cuda, cpu
    precision: str = "fp32"  # fp32, fp16, bf16
    
    def __post_init__(self):
        """Validate and auto-detect device."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SegmentationConfig:
    """Segmentation configuration."""
    
    multimask_output: bool = True
    stability_score_threshold: float = 0.95
    box_nms_threshold: float = 0.7
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 1
    min_mask_region_area: int = 0


@dataclass
class ReconstructionConfig:
    """3D reconstruction configuration."""
    
    depth_estimation_model: str = "dpt_large"
    point_cloud_density: int = 10000
    mesh_simplification_factor: float = 0.9
    texture_resolution: int = 1024
    enable_refinement: bool = True


@dataclass
class TrackingConfig:
    """Video tracking configuration."""
    
    max_frames: Optional[int] = None
    fps: Optional[int] = None
    propagation_window: int = 8
    memory_bank_size: int = 16
    confidence_threshold: float = 0.5


@dataclass
class APIConfig:
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB


@dataclass
class Config:
    """Main configuration class."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # Logging
    log_level: str = "INFO"
    enable_tensorboard: bool = False
    enable_wandb: bool = False
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.data_dir, self.output_dir, self.checkpoint_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        seg_config = SegmentationConfig(**config_dict.get("segmentation", {}))
        rec_config = ReconstructionConfig(**config_dict.get("reconstruction", {}))
        track_config = TrackingConfig(**config_dict.get("tracking", {}))
        api_config = APIConfig(**config_dict.get("api", {}))
        
        return cls(
            model=model_config,
            segmentation=seg_config,
            reconstruction=rec_config,
            tracking=track_config,
            api=api_config,
            **{k: v for k, v in config_dict.items() 
               if k not in ["model", "segmentation", "reconstruction", "tracking", "api"]}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "segmentation": self.segmentation.__dict__,
            "reconstruction": self.reconstruction.__dict__,
            "tracking": self.tracking.__dict__,
            "api": self.api.__dict__,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "log_dir": str(self.log_dir),
            "log_level": self.log_level,
            "enable_tensorboard": self.enable_tensorboard,
            "enable_wandb": self.enable_wandb,
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
default_config = Config()

