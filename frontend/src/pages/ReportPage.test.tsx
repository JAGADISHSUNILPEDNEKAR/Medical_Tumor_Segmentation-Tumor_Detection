import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../lib/api", () => ({
  fetchJob: vi.fn(),
  fetchReport: vi.fn(),
}));

import { ReportPage } from "./ReportPage";
import { fetchJob, fetchReport } from "../lib/api";
import { ApiError, type JobStatusResponse, type ReportResponse } from "../types/api";
import { REGION_CONFIG } from "../types/viewer";

const CASE_ID = "case-123";
const JOB_ID = "job-456";
const RESULT_ID = "result-789";

function mockJob(): JobStatusResponse {
  return {
    job_id: JOB_ID,
    case_id: CASE_ID,
    job_type: "PREDICT",
    status: "COMPLETED",
    progress: 100,
    inference_source: "mock",
    model_version: null,
    result_id: RESULT_ID,
    error_code: null,
    error_message: null,
    created_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:01Z",
    completed_at: "2026-01-01T00:00:02Z",
  };
}

function mockReport(): ReportResponse {
  return {
    result_id: RESULT_ID,
    job_id: JOB_ID,
    case_id: CASE_ID,
    status: "COMPLETED",
    measurements: {
      synthetic: true,
      description: "Geometric measurements of the synthetic segmentation mask.",
      voxel_spacing_mm: [1, 1, 1],
      voxel_volume_mm3: 1.0,
      foreground_voxels: 1037,
      foreground_volume_mm3: 1037.0,
      foreground_volume_cm3: 1.037,
      regions: [
        {
          region: "NCR",
          label: 1,
          voxel_count: 202,
          volume_mm3: 202.0,
          volume_cm3: 0.202,
          present: true,
          bounding_box: null,
          centroid_mm: null,
        },
      ],
    },
    evaluation: {
        available: true,
        ground_truth_available: true,
        per_class: [
            { class_name: "NCR", label: 1, dice: { value: 0.95, both_empty: false }, hd95: { value_mm: 1.5, defined: true, reason: null } }
        ],
        mean_dice: { value: 0.95, both_empty: false },
        mean_hd95: { value_mm: 1.5, defined: true, reason: null },
    },
    provenance: {
        inference_source: "mock",
        model_version: "v1.0",
        checkpoint_id: "ckpt-1",
        synthetic: true,
        inference_timestamp: "2026-01-01T00:00:01Z",
        description: "Test report",
    },
    created_at: "2026-01-01T00:00:02Z",
  };
}

function renderReport() {
  return render(
    <MemoryRouter initialEntries={[`/cases/${CASE_ID}/jobs/${JOB_ID}/report`]}>
      <Routes>
        <Route path="/cases/:caseId/jobs/:jobId/report" element={<ReportPage />} />
      </Routes>
    </MemoryRouter>
  );
}

describe("ReportPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the report data successfully", async () => {
    vi.mocked(fetchJob).mockResolvedValue(mockJob());
    vi.mocked(fetchReport).mockResolvedValue(mockReport());
    
    renderReport();

    // Verify header
    expect(await screen.findByText("Segmentation Report")).toBeInTheDocument();
    
    // Verify provenance
    expect(screen.getByText("mock")).toBeInTheDocument();
    expect(screen.getByText("v1.0")).toBeInTheDocument();
    expect(screen.getByText("Synthetic")).toBeInTheDocument();
    expect(screen.getByText("ckpt-1")).toBeInTheDocument();

    // Verify measurements
    expect(screen.getByText("Volumetric Measurements")).toBeInTheDocument();
    expect(screen.getByText("0.20")).toBeInTheDocument(); // NCR cm3
    expect(screen.getByText("1.04")).toBeInTheDocument(); // Foreground cm3

    // Verify evaluation
    expect(screen.getByText("Evaluation Metrics")).toBeInTheDocument();
    expect(screen.getAllByText("0.9500")).toHaveLength(2);
    expect(screen.getAllByText("1.50")).toHaveLength(2);
  });

  it("handles missing job result", async () => {
    vi.mocked(fetchJob).mockResolvedValue({ ...mockJob(), result_id: null });
    renderReport();

    expect(await screen.findByText("Error loading report")).toBeInTheDocument();
    expect(screen.getByText("No result available for this job yet.")).toBeInTheDocument();
  });

  it("handles api errors", async () => {
    vi.mocked(fetchJob).mockRejectedValue(new ApiError("Not found", 404, "JOB_NOT_FOUND"));
    renderReport();

    expect(await screen.findByText("Error loading report")).toBeInTheDocument();
    expect(screen.getByText("Not found")).toBeInTheDocument();
  });
});
