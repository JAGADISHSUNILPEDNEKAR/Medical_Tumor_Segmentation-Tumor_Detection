import numpy as np

from app.services.metrics_service import MetricsService


def test_metrics_service_empty_both():
    svc = MetricsService()
    pred = np.zeros((10, 10, 10), dtype=np.int16)
    gt = np.zeros((10, 10, 10), dtype=np.int16)
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_evaluation(pred, gt, spacing)
    
    assert res.available is True
    assert res.ground_truth_available is True
    assert res.mean_dice is None
    assert res.mean_hd95.value_mm == 0.0
    assert res.mean_hd95.defined is True
    
    for cls in res.per_class:
        assert cls.dice.value == 1.0
        assert cls.dice.both_empty is True
        assert cls.hd95.value_mm == 0.0
        assert cls.hd95.defined is True
        assert cls.hd95.reason == "both_empty"


def test_metrics_service_perfect_match():
    svc = MetricsService()
    pred = np.zeros((10, 10, 10), dtype=np.int16)
    pred[2:5, 2:5, 2:5] = 1
    
    gt = np.zeros((10, 10, 10), dtype=np.int16)
    gt[2:5, 2:5, 2:5] = 1
    
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_evaluation(pred, gt, spacing)
    
    ncr = next(c for c in res.per_class if c.label == 1)
    
    assert ncr.dice.value == 1.0
    assert ncr.dice.both_empty is False
    assert ncr.hd95.value_mm == 0.0
    assert ncr.hd95.defined is True


def test_metrics_service_one_sided_empty():
    svc = MetricsService()
    pred = np.zeros((10, 10, 10), dtype=np.int16)
    pred[2:5, 2:5, 2:5] = 1
    
    gt = np.zeros((10, 10, 10), dtype=np.int16)
    
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_evaluation(pred, gt, spacing)
    
    ncr = next(c for c in res.per_class if c.label == 1)
    
    assert ncr.dice.value == 0.0
    assert ncr.dice.both_empty is False
    assert ncr.hd95.value_mm is None
    assert ncr.hd95.defined is False
    assert ncr.hd95.reason == "one_sided_empty"


def test_metrics_service_partial_overlap():
    svc = MetricsService()
    pred = np.zeros((10, 10, 10), dtype=np.int16)
    pred[0, 0, 0] = 1
    pred[0, 0, 1] = 1
    
    gt = np.zeros((10, 10, 10), dtype=np.int16)
    gt[0, 0, 1] = 1
    gt[0, 0, 2] = 1
    
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_evaluation(pred, gt, spacing)
    
    ncr = next(c for c in res.per_class if c.label == 1)
    
    # Intersection is 1 voxel [0,0,1]
    # Sum is 2 + 2 = 4
    # Dice = 2 * 1 / 4 = 0.5
    assert ncr.dice.value == 0.5
    assert ncr.hd95.defined is True
    # HD95 should be positive distance
    assert ncr.hd95.value_mm > 0
