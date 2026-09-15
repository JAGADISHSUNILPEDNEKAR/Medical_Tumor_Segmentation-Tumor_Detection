/**
 * ResultsPage tests.
 *
 * Covers the job polling contract, the mock-result disclaimer, measurement
 * rendering, and the regression that coloured every region identically
 * because it matched on the display string rather than the BraTS label.
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../lib/api", () => ({
  fetchJob: vi.fn(),
  fetchResult: vi.fn(),
}));

import { ResultsPage } from "./ResultsPage";
import { fetchJob, fetchResult } from "../lib/api";
import { ApiError, type JobStatusResponse, type ResultResponse } from "../types/api";
import { REGION_CONFIG } from "../types/viewer";

const CASE_ID = "case-123";
const JOB_ID = "job-456";

function job(overrides: Partial<JobStatusResponse> = {}): JobStatusResponse {
  return {
    job_id: JOB_ID,
    case_id: CASE_ID,
    job_type: "PREDICT",
    status: "COMPLETED",
    progress: 100,
    inference_source: "mock",
    model_version: null,
    result_id: "result-789",
    error_code: null,
    error_message: null,
    created_at: "2026-01-01T00:00:00Z",
    started_at: "2026-01-01T00:00:01Z",
    completed_at: "2026-01-01T00:00:02Z",
    ...overrides,
  };
}

/**
 * Region keys match the backend's real response shape, which is
 * "NCR (Necrotic Core)" rather than a bare "NCR".
 */
function result(): ResultResponse {
  return {
    result_id: "result-789",
    job_id: JOB_ID,
    case_id: CASE_ID,
    status: "COMPLETED",
    inference_source: "mock",
    model_version: null,
    segmentation: { available: true },
    measurements: {
      synthetic: true,
      description: "Geometric measurements of the synthetic segmentation mask.",
      foreground_voxels: 1037,
      foreground_volume_mm3: 2074,
      foreground_volume_cm3: 2.074,
      regions: {
        "NCR (Necrotic Core)": {
          label: 1,
          voxel_count: 202,
          volume_mm3: 404,
          volume_cm3: 0.404,
        },
        "ED (Peritumoral Edema)": {
          label: 2,
          voxel_count: 804,
          volume_mm3: 1608,
          volume_cm3: 1.608,
        },
        "ET (Enhancing Tumor)": {
          label: 4,
          voxel_count: 31,
          volume_mm3: 62,
          volume_cm3: 0.062,
        },
      },
    },
    evaluation: { available: false, dice: null, hd95_mm: null },
    created_at: "2026-01-01T00:00:02Z",
  };
}

function renderResults() {
  return render(
    <MemoryRouter initialEntries={[`/cases/${CASE_ID}/jobs/${JOB_ID}`]}>
      <Routes>
        <Route path="/cases/:caseId/jobs/:jobId" element={<ResultsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ResultsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows the synthetic demo disclaimer", async () => {
    vi.mocked(fetchJob).mockResolvedValue(job());
    vi.mocked(fetchResult).mockResolvedValue(result());
    renderResults();
    expect(await screen.findByText(/synthetic demo mode/i)).toBeInTheDocument();
    expect(
      screen.getByText(/not by a trained medical model/i),
    ).toBeInTheDocument();
  });

  it("renders total and per-region volumes once the job completes", async () => {
    vi.mocked(fetchJob).mockResolvedValue(job());
    vi.mocked(fetchResult).mockResolvedValue(result());
    renderResults();

    expect(await screen.findByText("2.07")).toBeInTheDocument();
    expect(screen.getByText("0.40", { exact: false })).toBeInTheDocument();
    expect(screen.getByText("1.61", { exact: false })).toBeInTheDocument();
  });

  it("colours each region by its BraTS label, not by the display string", async () => {
    vi.mocked(fetchJob).mockResolvedValue(job());
    vi.mocked(fetchResult).mockResolvedValue(result());
    const { container } = renderResults();

    await screen.findByText("2.07");

    const swatches = Array.from(
      container.querySelectorAll<HTMLElement>("li span[style*='background-color']"),
    ).map((el) => el.style.backgroundColor);

    expect(swatches).toHaveLength(3);
    // Distinct colours, drawn from the shared viewer configuration.
    expect(new Set(swatches).size).toBe(3);
    for (const label of [1, 2, 4]) {
      expect(screen.getByText(REGION_CONFIG[label]!.fullName)).toBeInTheDocument();
    }
  });

  it("keeps polling while the job is still running", async () => {
    vi.mocked(fetchJob)
      .mockResolvedValueOnce(job({ status: "RUNNING", progress: 30, result_id: null }))
      .mockResolvedValue(job());
    vi.mocked(fetchResult).mockResolvedValue(result());
    renderResults();

    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
    await waitFor(() => expect(vi.mocked(fetchJob).mock.calls.length).toBeGreaterThan(1), {
      timeout: 3000,
    });
    expect(await screen.findByText("COMPLETED")).toBeInTheDocument();
  });

  it("surfaces a failed job's error without fetching a result", async () => {
    vi.mocked(fetchJob).mockResolvedValue(
      job({
        status: "FAILED",
        progress: 40,
        result_id: null,
        error_code: "INFERENCE_FAILED",
        error_message: "Mock inference failed. This is not a clinical result.",
      }),
    );
    renderResults();

    expect(await screen.findByText(/INFERENCE_FAILED/)).toBeInTheDocument();
    expect(fetchResult).not.toHaveBeenCalled();
  });

  it("shows a readable message when the API is unreachable", async () => {
    vi.mocked(fetchJob).mockRejectedValue(new ApiError("Job not found.", 404, "JOB_NOT_FOUND"));
    renderResults();
    expect(await screen.findByText("Job not found.")).toBeInTheDocument();
  });
});
