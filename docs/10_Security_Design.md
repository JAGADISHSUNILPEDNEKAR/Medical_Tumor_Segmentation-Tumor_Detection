# Document 10 — Security Design

## 1. Authentication & Authorization
* **Authentication:** MVP ships without gated auth (single-user/small-class demo); V1 adds basic session-based authentication.
* **Role-Based Access Control (RBAC), V1:**
  * `Researcher` / `Radiologist Reviewer`: Can upload cases, run predict/evaluate jobs, and view their own case results.
  * `Admin`: Full checkpoint/config management and cross-session case visibility for debugging.
* Even in a demo/academic deployment, one session must not be able to browse another's cases by guessing IDs — enforced via opaque, non-guessable, server-generated case/job/result identifiers (never sequential integers).

## 2. Input Validation & Media Security
* Every uploaded file is validated server-side (NIfTI header parse, 4-modality shape/affine consistency check) before it reaches any parsing library — client-side validation is never trusted alone.
* Upload size limits enforced at the web-server layer, not only in application code (illustrative cap: 500 MB per case, tuned once real case sizes are confirmed against the chosen BraTS release).
* Uploaded cases and results are stored in a location that is not publicly web-accessible.

## 3. Data Privacy
* No real patient data is used in development — only public, de-identified BraTS research data; no additional real clinical data may be introduced without institutional review and explicit legal clearance (out of scope for this project).
* Logs never contain uploaded file contents, voxel data, or any field that could re-identify a case beyond its internal system-generated ID.
* A data-deletion path (endpoint or admin script) exists to remove an uploaded case and its derived results.
* HTTPS required for any non-local deployment; secrets (session secrets, DB credentials) supplied via environment variables, never committed to the repository.

## 4. Compliance Disclaimer
This project is an academic/research prototype and is **not automatically HIPAA/GDPR compliant** merely because these controls exist. Achieving actual regulatory compliance would require a formal risk assessment, a designated compliance officer, signed data-processing agreements, audited infrastructure, and — for HIPAA specifically — a Business Associate Agreement with any cloud vendor used; none of that is in scope here.
