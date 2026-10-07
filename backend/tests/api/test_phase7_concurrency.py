from app.services.job_queue import JobQueue
from app.inference.mock import MockInferenceService
from pathlib import Path

def test_phase7_queue_depth_limit():
    service = MockInferenceService()
    # The maxsize is 100 now
    queue = JobQueue(service, Path("/tmp"))
    assert queue._queue.maxsize == 100
    
def test_phase7_single_worker_lock():
    service = MockInferenceService()
    queue = JobQueue(service, Path("/tmp"))
    assert hasattr(queue, "_inference_lock")
