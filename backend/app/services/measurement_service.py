from __future__ import annotations

import numpy as np

from app.core.constants import REGION_NAMES
from app.schemas.results import MeasurementsInfo, RegionStats


class MeasurementService:
    """Calculates geometric measurements from a semantic segmentation mask."""

    def compute_measurements(
        self,
        segmentation: np.ndarray,
        spacing: tuple[float, float, float],
        is_synthetic: bool = False,
        description: str | None = None,
    ) -> MeasurementsInfo:
        """Compute volumes and centroids for BraTS segmentation labels."""
        voxel_volume_mm3 = float(np.prod(spacing))
        
        # BraTS labels: 1=NCR, 2=ED, 4=ET
        labels_of_interest = [1, 2, 4]
        
        regions = []
        total_foreground_voxels = 0
        
        for label in labels_of_interest:
            mask = (segmentation == label)
            voxel_count = int(np.sum(mask))
            total_foreground_voxels += voxel_count
            
            volume_mm3 = float(voxel_count * voxel_volume_mm3)
            volume_cm3 = volume_mm3 / 1000.0
            
            present = voxel_count > 0
            
            bounding_box = None
            centroid_mm = None
            
            if present:
                # Calculate bounding box
                coords = np.argwhere(mask)
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                bounding_box = {
                    "min": min_coords.tolist(),
                    "max": max_coords.tolist()
                }
                
                # Calculate centroid in mm
                centroid_vox = coords.mean(axis=0)
                centroid_mm = (centroid_vox * np.array(spacing)).tolist()

            region_name = REGION_NAMES.get(label, f"Label {label}")
            
            regions.append(
                RegionStats(
                    region=region_name,
                    label=label,
                    voxel_count=voxel_count,
                    volume_mm3=volume_mm3,
                    volume_cm3=volume_cm3,
                    present=present,
                    bounding_box=bounding_box,
                    centroid_mm=centroid_mm,
                )
            )

        foreground_volume_mm3 = float(total_foreground_voxels * voxel_volume_mm3)
        
        return MeasurementsInfo(
            synthetic=is_synthetic,
            description=description,
            voxel_spacing_mm=list(spacing),
            voxel_volume_mm3=voxel_volume_mm3,
            foreground_voxels=total_foreground_voxels,
            foreground_volume_mm3=foreground_volume_mm3,
            foreground_volume_cm3=foreground_volume_mm3 / 1000.0,
            regions=regions,
        )
