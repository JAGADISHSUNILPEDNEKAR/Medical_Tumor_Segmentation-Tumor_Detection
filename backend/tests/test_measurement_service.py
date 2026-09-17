import numpy as np

from app.schemas.results import MeasurementsInfo
from app.services.measurement_service import MeasurementService


def test_measurement_service_empty():
    svc = MeasurementService()
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_measurements(seg, spacing, is_synthetic=True)
    
    assert res.foreground_voxels == 0
    assert res.foreground_volume_mm3 == 0.0
    assert res.foreground_volume_cm3 == 0.0
    
    for region in res.regions:
        assert region.voxel_count == 0
        assert region.present is False
        assert region.bounding_box is None
        assert region.centroid_mm is None


def test_measurement_service_single_region():
    svc = MeasurementService()
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    seg[4:6, 4:6, 4:6] = 1  # 2x2x2 = 8 voxels of NCR
    spacing = (2.0, 2.0, 2.0)  # Voxel volume = 8.0 mm3

    res = svc.compute_measurements(seg, spacing, is_synthetic=True)
    
    assert res.voxel_volume_mm3 == 8.0
    assert res.foreground_voxels == 8
    assert res.foreground_volume_mm3 == 64.0
    assert res.foreground_volume_cm3 == 0.064
    
    ncr_region = next(r for r in res.regions if r.label == 1)
    assert ncr_region.voxel_count == 8
    assert ncr_region.present is True
    assert ncr_region.volume_mm3 == 64.0
    
    assert ncr_region.bounding_box is not None
    assert ncr_region.bounding_box["min"] == [4, 4, 4]
    assert ncr_region.bounding_box["max"] == [5, 5, 5]
    
    # centroid vox is 4.5, 4.5, 4.5
    # spacing is 2.0, so mm is 9.0, 9.0, 9.0
    assert ncr_region.centroid_mm == [9.0, 9.0, 9.0]


def test_measurement_service_multiple_regions():
    svc = MeasurementService()
    seg = np.zeros((10, 10, 10), dtype=np.int16)
    seg[0, 0, 0] = 1 # 1 NCR
    seg[1, 1, 1] = 2 # 1 ED
    seg[2, 2, 2] = 4 # 1 ET
    spacing = (1.0, 1.0, 1.0)

    res = svc.compute_measurements(seg, spacing)
    
    assert res.foreground_voxels == 3
    assert res.foreground_volume_mm3 == 3.0
    
    for r in res.regions:
        assert r.present is True
        assert r.voxel_count == 1
        assert r.volume_mm3 == 1.0
