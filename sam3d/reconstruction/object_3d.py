"""3D object reconstruction from single images."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any
from loguru import logger

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("open3d not available for 3D visualization")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available for mesh processing")


class Object3DReconstructor:
    """
    Reconstruct 3D objects from single images using SAM 3D.
    
    Features:
    - Depth estimation from single image
    - Point cloud generation
    - Mesh reconstruction
    - Texture mapping
    - Multi-view fusion
    """
    
    def __init__(
        self,
        depth_model: str = "dpt_large",
        device: Optional[str] = None
    ):
        """
        Initialize 3D reconstructor.
        
        Args:
            depth_model: Depth estimation model (dpt_large, midas, etc.)
            device: Device to run on (cuda, cpu, or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model_name = depth_model
        self.depth_model = None
        self.depth_transform = None
        
        logger.info(f"Initialized Object3DReconstructor with {depth_model} on {self.device}")
    
    def load_depth_model(self):
        """Load depth estimation model."""
        logger.info(f"Loading depth model: {self.depth_model_name}")
        
        try:
            if self.depth_model_name.startswith("dpt"):
                # Intel DPT model
                model = torch.hub.load(
                    "intel-isl/MiDaS",
                    self.depth_model_name,
                    pretrained=True
                )
            else:
                # MiDaS model
                model = torch.hub.load(
                    "intel-isl/MiDaS",
                    "MiDaS_small",
                    pretrained=True
                )
            
            model.to(self.device)
            model.eval()
            
            self.depth_model = model
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.depth_model_name == "dpt_large":
                self.depth_transform = midas_transforms.dpt_transform
            else:
                self.depth_transform = midas_transforms.small_transform
            
            logger.success("Depth model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
            raise
    
    def estimate_depth(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Estimate depth map from image.
        
        Args:
            image: Input image
            
        Returns:
            Depth map (numpy array)
        """
        if self.depth_model is None:
            self.load_depth_model()
        
        # Load and prepare image
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image
        
        # Transform image
        input_batch = self.depth_transform(img_pil).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        logger.debug(f"Estimated depth map: {depth_map.shape}")
        
        return depth_map
    
    def reconstruct_from_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D from single image.
        
        Args:
            image: Input image
            mask: Optional segmentation mask
            
        Returns:
            Dictionary with point_cloud, mesh, and metadata
        """
        logger.info("Starting 3D reconstruction from image")
        
        # Load image
        if isinstance(image, (str, Path)):
            img_pil = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            img_pil = image
        
        img_array = np.array(img_pil)
        
        # Estimate depth
        depth_map = self.estimate_depth(img_array)
        
        # Generate point cloud
        point_cloud = self.depth_to_point_cloud(
            depth_map, img_array, mask
        )
        
        # Generate mesh
        mesh = self.point_cloud_to_mesh(point_cloud)
        
        result = {
            'point_cloud': point_cloud,
            'mesh': mesh,
            'depth_map': depth_map,
            'image': img_array,
            'mask': mask
        }
        
        logger.success("3D reconstruction complete")
        
        return result
    
    def reconstruct_from_mask(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D from image and segmentation mask.
        
        Args:
            image: Input image
            mask: Segmentation mask
            
        Returns:
            3D reconstruction result
        """
        return self.reconstruct_from_image(image, mask)
    
    def depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None
    ) -> Union[o3d.geometry.PointCloud, np.ndarray]:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map
            image: RGB image
            mask: Optional mask to filter points
            focal_length: Camera focal length (estimated if None)
            
        Returns:
            Point cloud object or numpy array
        """
        height, width = depth_map.shape
        
        # Estimate focal length if not provided
        if focal_length is None:
            focal_length = width / 2.0
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to 3D coordinates
        z = depth_map
        x = (x - width / 2.0) * z / focal_length
        y = (y - height / 2.0) * z / focal_length
        
        # Stack into points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        colors = image.reshape(-1, 3) / 255.0
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            points = points[mask_flat > 0]
            colors = colors[mask_flat > 0]
        
        # Create Open3D point cloud if available
        if OPEN3D_AVAILABLE:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
        else:
            return points
    
    def point_cloud_to_mesh(
        self,
        point_cloud: Union[o3d.geometry.PointCloud, np.ndarray],
        method: str = "poisson"
    ) -> Union[o3d.geometry.TriangleMesh, trimesh.Trimesh, None]:
        """
        Convert point cloud to mesh.
        
        Args:
            point_cloud: Input point cloud
            method: Reconstruction method (poisson, ball_pivoting, alpha_shape)
            
        Returns:
            Mesh object or None
        """
        if not OPEN3D_AVAILABLE and not isinstance(point_cloud, np.ndarray):
            logger.warning("open3d not available for mesh generation")
            return None
        
        if isinstance(point_cloud, np.ndarray):
            if not OPEN3D_AVAILABLE:
                return None
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            pcd = point_cloud
        
        logger.info(f"Generating mesh using {method} method")
        
        try:
            if method == "poisson":
                # Poisson surface reconstruction
                pcd.estimate_normals()
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9
                )
                
                # Remove low density vertices
                vertices_to_remove = densities < np.quantile(densities, 0.1)
                mesh.remove_vertices_by_mask(vertices_to_remove)
                
            elif method == "ball_pivoting":
                # Ball pivoting algorithm
                pcd.estimate_normals()
                radii = [0.005, 0.01, 0.02, 0.04]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                
            elif method == "alpha_shape":
                # Alpha shapes
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                    pcd, alpha=0.03
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            logger.success(f"Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
            
            return mesh
            
        except Exception as e:
            logger.error(f"Mesh generation failed: {e}")
            return None
    
    def save_point_cloud(
        self,
        point_cloud: Union[o3d.geometry.PointCloud, np.ndarray],
        output_path: Union[str, Path],
        format: str = "ply"
    ):
        """
        Save point cloud to file.
        
        Args:
            point_cloud: Point cloud to save
            output_path: Output file path
            format: File format (ply, pcd, xyz)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(point_cloud, np.ndarray):
            if not OPEN3D_AVAILABLE:
                # Save as numpy
                np.save(str(output_path.with_suffix('.npy')), point_cloud)
                logger.info(f"Saved point cloud as numpy: {output_path.with_suffix('.npy')}")
                return
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        else:
            pcd = point_cloud
        
        o3d.io.write_point_cloud(str(output_path), pcd)
        logger.success(f"Saved point cloud: {output_path}")
    
    def save_mesh(
        self,
        mesh: Union[o3d.geometry.TriangleMesh, trimesh.Trimesh],
        output_path: Union[str, Path],
        format: str = "obj"
    ):
        """
        Save mesh to file.
        
        Args:
            mesh: Mesh to save
            output_path: Output file path
            format: File format (obj, ply, stl, gltf)
        """
        if mesh is None:
            logger.warning("No mesh to save")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(output_path), mesh)
        elif TRIMESH_AVAILABLE and isinstance(mesh, trimesh.Trimesh):
            mesh.export(str(output_path))
        else:
            logger.warning("Cannot save mesh: unsupported type")
            return
        
        logger.success(f"Saved mesh: {output_path}")
    
    def visualize_3d(
        self,
        geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
        window_name: str = "3D Visualization"
    ):
        """
        Visualize 3D geometry.
        
        Args:
            geometry: Point cloud or mesh to visualize
            window_name: Window title
        """
        if not OPEN3D_AVAILABLE:
            logger.warning("open3d not available for visualization")
            return
        
        o3d.visualization.draw_geometries([geometry], window_name=window_name)

