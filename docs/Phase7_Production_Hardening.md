# Phase 7: Production Hardening & Deployment

Phase 7 hardens the verified machine learning architecture of Phase 6 into a production-ready system. The primary goal is infrastructure security, memory safety, logging, and concurrency control without altering the verified parameters or algorithms of the 3D U-Net.

## Configuration & Fail-Fast Validation

The environment configuration via `.env` now imposes strict startup validation:
- **PyTorch Backend Verification**: `INFERENCE_BACKEND="pytorch"` requires `MODEL_PATH` to be explicitly set and resolvable. If missing, the API throws a `ValidationError` and aborts startup rather than silently failing to mock outputs or crashing mid-inference.
- **Worker Configuration Check**: `INFERENCE_WORKERS` must be greater than or equal to `MAX_CONCURRENT_JOBS`.

## Concurrency and Memory Safety

Inference workloads (specifically full 128³ patch processing) are highly resource-intensive:
- **Single-worker Locking**: The `JobQueue` now implements a strict `threading.Lock()` enforcing that only one active ML job accesses the inference device at a time, preventing Out-Of-Memory (OOM) crashes on the GPU or CPU host.
- **Queue Limits**: `JobQueue` imposes a strict size limit (`maxsize=100`) to prevent unbound queuing.
- **CPU Garbage Collection**: Explicit `gc.collect()` calls are strategically placed after job completion and inside sliding-window operations to clear dangling memory buffers, crucial for long-running worker stability.
- **Job Recovery**: On API startup, `JobService.recover_stale_jobs()` identifies any jobs left in `RUNNING` state due to a host crash and transitions them cleanly to `FAILED`.

## API Security

The artifact layer was hardened:
- **Path Traversal Protection**: `CaseStorage` aggressively blocks requests for files containing `../` or resolving outside the designated artifact root directory.
- **Upload Restrictions**: Uploads strictly adhere to `MAX_UPLOAD_SIZE_MB`, validated by scanning the incoming buffers before disk IO. NIfTI `.gz` payloads are checked via magic bytes (`\x1f\x8b`) to prevent rogue binary execution.
- **Headers**: Artifact downloads via `FileResponse` now inject `X-Content-Type-Options: nosniff`.
- **Temp Cleanup**: `cleanup_temp_files()` ensures residual processing files are swept from the queue after a job concludes, even if it failed.

## Observability

Structured logging using `JsonFormatter` was extended to include comprehensive ML diagnostics:
- Provenance logs contain `device`, `parameter_count`, `backend`, `patch_size`, and `compat_fingerprint` alongside basic job telemetry.
- Timing metrics natively record `preprocess_seconds`, `inference_seconds`, and `postprocess_seconds` directly to stdout for ingestion by SIEM tooling.
- Application initialization logs the selected configuration parameters at startup.

## Docker Profiles

Two explicit deployment paths are provided:
- **CPU/General `docker-compose.yml`**: Uses `Dockerfile.backend` featuring a non-root `appuser`, structured host volume mounts, healthchecks, memory limits, and a built-in restart policy.
- **GPU Template `docker-compose.gpu.yml`**: Uses `Dockerfile.backend.gpu` which provides an untested NVIDIA CUDA base-image layout for deployment onto hardware with the `nvidia-container-toolkit`.
