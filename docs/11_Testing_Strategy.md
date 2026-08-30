# Document 11 — Testing Strategy

## Test Cases

| Test ID | Category | Scenario | Input | Expected Outcome | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TEST-01** | Unit | Dice/HD95 computed on synthetic arrays with known, hand-computed values | Two small synthetic masks with a known overlap | Computed Dice/HD95 match the hand-computed expected values, including empty-mask edge cases. | High |
| **TEST-02** | Unit | Resampling preserves the label set on ground-truth masks | A label mask with classes `{0,1,2,4}` resampled via nearest-neighbor | Resampled mask contains only the original label set — no fractional/interpolated classes introduced. | High |
| **TEST-03** | Integration | Model forward pass produces the expected tensor shape | Dummy input tensor `(1, 4, 128, 128, 128)` | Output tensor of shape `(1, C, 128, 128, 128)` where `C` = number of classes. | High |
| **TEST-04** | API | Upload a case missing one modality | 3 of 4 required modality files | API returns HTTP 400 with a machine-readable error naming the missing modality. | Medium |
| **TEST-05** | System E2E | End-to-end upload → mask → metrics → mesh pipeline | One fixture 4-modality case with ground truth | Job reaches `done` status; response includes a valid `mask_url`, computed metrics, and a `mesh_url`. | High |
