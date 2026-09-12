from pydantic import BaseModel


class APIError(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
    case_id: str | None = None
