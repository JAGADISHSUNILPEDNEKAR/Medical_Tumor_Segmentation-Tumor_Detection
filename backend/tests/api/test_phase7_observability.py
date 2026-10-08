from app.core.logging import _LOGGED_EXTRA_FIELDS

def test_phase7_log_allowlist_contains_observability_fields():
    expected_fields = [
        "input_shape",
        "patch_size",
        "preprocess_seconds",
        "postprocess_seconds",
        "total_seconds",
        "status",
        "count",
        "timeout",
    ]
    for field in expected_fields:
        assert field in _LOGGED_EXTRA_FIELDS
