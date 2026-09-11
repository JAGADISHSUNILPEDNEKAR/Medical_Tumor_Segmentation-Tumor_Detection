"""Production inference boundary.

The web application must call this interface only. Concrete implementations
(MockInferenceService, RealBraTSInferenceService) are added in later phases.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceService(Protocol):
    def predict(self, case: Any) -> Any:
        """Run segmentation for a validated case.

        Not implemented in Phase 1. Do not attach a fake trained model.
        """
        ...
