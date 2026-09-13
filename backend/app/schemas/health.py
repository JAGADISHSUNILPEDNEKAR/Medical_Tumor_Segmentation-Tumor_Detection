from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool = Field(
        description="True only when a real inference implementation has loaded a checkpoint."
    )
    inference_source: str = Field(
        description="unavailable | mock | pytorch."
    )


class ModelInfoResponse(BaseModel):
    checkpoint_id: str | None = None
    dataset_version: str | None = None
    trained_on_split: str | None = None
    headline_metrics: dict[str, float] | None = Field(
        default=None,
        description="Registered checkpoint metrics only. Null when no checkpoint is loaded.",
    )
    num_classes: int = 4
    input_modalities: list[str] = Field(default_factory=lambda: ["T1", "T1ce", "T2", "FLAIR"])
    model_loaded: bool
    inference_source: str
    model_version: str | None = None
    message: str
