from __future__ import annotations

import logging
import numpy as np
from scipy.ndimage import binary_erosion

from app.core.constants import REGION_NAMES
from app.schemas.results import DiceMetric, EvaluationInfo, HD95Metric, PerClassMetrics


logger = logging.getLogger(__name__)


class MetricsService:
    """Calculates Dice and HD95 metrics between prediction and ground truth."""

    def compute_evaluation(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> EvaluationInfo:
        """Compute metrics for BraTS segmentation labels."""
        # BraTS labels: 1=NCR, 2=ED, 4=ET
        labels_of_interest = [1, 2, 4]
        
        per_class = []
        dice_values = []
        hd95_values = []
        
        for label in labels_of_interest:
            pred_mask = (prediction == label)
            gt_mask = (ground_truth == label)
            
            dice = self._compute_dice(pred_mask, gt_mask)
            hd95 = self._compute_hd95(pred_mask, gt_mask, spacing)
            
            class_name = REGION_NAMES.get(label, f"Label {label}")
            
            per_class.append(
                PerClassMetrics(
                    class_name=class_name,
                    label=label,
                    dice=dice,
                    hd95=hd95,
                )
            )
            
            if dice is not None and not dice.both_empty:
                dice_values.append(dice.value)
                
            if hd95 is not None and hd95.defined:
                hd95_values.append(hd95.value_mm)

        mean_dice = None
        if dice_values:
            mean_dice = DiceMetric(value=float(np.mean(dice_values)), both_empty=False)
            
        mean_hd95 = None
        if hd95_values:
            mean_hd95 = HD95Metric(value_mm=float(np.mean(hd95_values)), defined=True)
            
        return EvaluationInfo(
            available=True,
            ground_truth_available=True,
            per_class=per_class,
            mean_dice=mean_dice,
            mean_hd95=mean_hd95,
        )

    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray) -> DiceMetric:
        pred_empty = not np.any(pred)
        gt_empty = not np.any(gt)
        
        if pred_empty and gt_empty:
            return DiceMetric(value=1.0, both_empty=True)
        if pred_empty or gt_empty:
            return DiceMetric(value=0.0, both_empty=False)
            
        intersection = np.logical_and(pred, gt)
        dice = 2.0 * intersection.sum() / (pred.sum() + gt.sum())
        return DiceMetric(value=float(dice), both_empty=False)

    def _compute_hd95(
        self, pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
    ) -> HD95Metric:
        pred_empty = not np.any(pred)
        gt_empty = not np.any(gt)
        
        if pred_empty and gt_empty:
            return HD95Metric(value_mm=0.0, defined=True, reason="both_empty")
        if pred_empty or gt_empty:
            return HD95Metric(value_mm=None, defined=False, reason="one_sided_empty")
            
        # Get boundaries
        pred_border = self._get_border(pred)
        gt_border = self._get_border(gt)
        
        # Calculate surface distances
        pred_dist = self._surface_distances(pred_border, gt_border, spacing)
        gt_dist = self._surface_distances(gt_border, pred_border, spacing)
        
        if len(pred_dist) == 0 or len(gt_dist) == 0:
            return HD95Metric(value_mm=None, defined=False, reason="no_boundary")
            
        # Combine distances and calculate 95th percentile
        all_dist = np.concatenate([pred_dist, gt_dist])
        hd95 = np.percentile(all_dist, 95)
        
        return HD95Metric(value_mm=float(hd95), defined=True)

    def _get_border(self, mask: np.ndarray) -> np.ndarray:
        """Extract the border of a boolean mask."""
        eroded = binary_erosion(mask)
        return np.logical_and(mask, np.logical_not(eroded))

    def _surface_distances(
        self, mask1: np.ndarray, mask2: np.ndarray, spacing: tuple[float, float, float]
    ) -> np.ndarray:
        """Calculate distances from surface voxels in mask1 to nearest in mask2."""
        coords1 = np.argwhere(mask1)
        coords2 = np.argwhere(mask2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            return np.array([])
            
        # Physical coordinates
        pts1 = coords1 * np.array(spacing)
        pts2 = coords2 * np.array(spacing)
        
        # We can use scipy.spatial.cKDTree for efficient nearest neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(pts2)
        distances, _ = tree.query(pts1)
        
        return distances
