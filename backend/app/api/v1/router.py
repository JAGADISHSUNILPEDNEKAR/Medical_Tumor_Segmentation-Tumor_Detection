from fastapi import APIRouter

from app.api.v1.cases import router as cases_router
from app.api.v1.health import router as health_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.predict import router as predict_router
from app.api.v1.results import router as results_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(cases_router)
api_router.include_router(predict_router)
api_router.include_router(jobs_router)
api_router.include_router(results_router)
