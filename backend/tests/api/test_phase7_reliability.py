from app.services.job_service import JobService
from app.db.session import get_session
from app.db.models import JobRecord
from app.core.constants import JobStatus, JobType

def test_phase7_stale_job_recovery(app):
    # Create stale jobs
    db_session = next(get_session())
    job_svc = JobService(db_session)
    job1 = job_svc.create_job("case_123", JobType.PREDICT)
    job_svc.transition(job1.job_id, JobStatus.QUEUED, JobStatus.RUNNING)
    
    job2 = job_svc.create_job("case_456", JobType.PREDICT)
    # job2 stays QUEUED
    db_session.commit()
    
    recovered = job_svc.recover_stale_jobs()
    assert recovered == 2
    
    # Check status
    assert job1.status == JobStatus.FAILED.value
    assert job2.status == JobStatus.FAILED.value
    assert job1.error_code == "INVALID_JOB_STATE"
    assert "abandoned" in job1.error_message
