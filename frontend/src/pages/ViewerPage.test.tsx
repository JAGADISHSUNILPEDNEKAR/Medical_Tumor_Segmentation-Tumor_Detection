/**
 * ViewerPage integration tests.
 *
 * Tests component rendering, loading states, disclaimer visibility,
 * and control presence. Does NOT test canvas rendering or WebGL
 * (those are tested through the unit tests on sliceExtraction and meshGeneration).
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, within as getWithin } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { ViewerPage } from "./ViewerPage";

// Mock the nifti loader to avoid real network requests
vi.mock("../lib/nifti", () => ({
  loadNiftiFromUrl: vi.fn(),
}));

// Metadata panel calls the API client; keep it off the network.
vi.mock("../lib/api", () => ({
  fetchCase: vi.fn(),
  fetchJob: vi.fn(),
}));

// Mock react-three-fiber since jsdom doesn't support WebGL
vi.mock("@react-three/fiber", () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="three-canvas">{children}</div>
  ),
}));

vi.mock("@react-three/drei", () => ({
  OrbitControls: () => null,
}));

import userEventDefault from "@testing-library/user-event";
import { loadNiftiFromUrl } from "../lib/nifti";
import { fetchCase, fetchJob } from "../lib/api";
import type { CaseResponse } from "../types/api";
import type { NiftiVolume } from "../types/viewer";

const mockCase: CaseResponse = {
  case_id: "test-case-123",
  status: "READY",
  created_at: "2026-01-01T09:30:00Z",
  modalities_present: ["flair", "t1", "t1ce", "t2"],
  has_ground_truth: false,
  total_bytes: 1024,
  files: [],
  validation: null,
  error: null,
  detail: null,
  inference: "not_started",
};

const mockVolume: NiftiVolume = {
  data: new Float32Array(8 * 8 * 8),
  shape: [8, 8, 8],
  affine: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
  spacing: [1, 1, 1],
};

function renderViewer(caseId = "test-case-123") {
  return render(
    <MemoryRouter initialEntries={[`/cases/${caseId}/viewer`]}>
      <Routes>
        <Route path="/cases/:caseId/viewer" element={<ViewerPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("ViewerPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchCase).mockResolvedValue(mockCase);
    vi.mocked(fetchJob).mockResolvedValue(
      undefined as never, // no ?job= in the test URL, so this is never called
    );
  });

  it("shows loading state initially", () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockReturnValue(new Promise(() => {})); // Never resolves

    renderViewer();

    expect(screen.getByText(/loading mri volumes/i)).toBeInTheDocument();
  });

  it("shows synthetic demo disclaimer", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(
      await screen.findByText(/synthetic demo mode/i),
    ).toBeInTheDocument();
  });

  it("shows modality selector after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    const selector = await screen.findByLabelText(/modality/i);
    expect(selector).toBeInTheDocument();
  });

  it("shows error state on load failure", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockRejectedValue(new Error("Network error"));

    renderViewer();

    expect(
      await screen.findByText(/failed to load viewer/i),
    ).toBeInTheDocument();
  });

  it("renders three plane views after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(await screen.findByText("AXIAL")).toBeInTheDocument();
    expect(screen.getByText("CORONAL")).toBeInTheDocument();
    expect(screen.getByText("SAGITTAL")).toBeInTheDocument();
  });

  it("renders slice controls after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(
      await screen.findByText(/slice navigation/i),
    ).toBeInTheDocument();
  });

  it("renders overlay toggle after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(
      await screen.findByRole("switch", { name: /toggle segmentation overlay/i }),
    ).toBeInTheDocument();
  });

  it("renders region visibility controls after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(
      await screen.findByText(/region visibility/i),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/toggle ncr/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/toggle ed/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/toggle et/i)).toBeInTheDocument();
  });

  it("renders open 3D view button after loading", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer();

    expect(
      await screen.findByRole("button", { name: /open 3d/i }),
    ).toBeInTheDocument();
  });

  it("shows case ID in toolbar", async () => {
    const mockedLoad = vi.mocked(loadNiftiFromUrl);
    mockedLoad.mockResolvedValue(mockVolume);

    renderViewer("abcdef12-3456-7890-abcd-ef1234567890");

    expect(
      await screen.findByText("abcdef12…"),
    ).toBeInTheDocument();
  });
});

describe("ViewerPage case metadata", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loadNiftiFromUrl).mockResolvedValue(mockVolume);
  });

  it("lists the modalities present on the case", async () => {
    vi.mocked(fetchCase).mockResolvedValue(mockCase);
    renderViewer();

    // Scoped to the panel: the modality selector renders these labels too.
    const panel = await screen.findByRole("region", { name: /case metadata/i });
    const within = getWithin(panel);
    for (const label of ["T1", "T1ce", "T2", "FLAIR"]) {
      expect(within.getByText(label)).toBeInTheDocument();
    }
    expect(within.getByText("test-case-123")).toBeInTheDocument();
    expect(within.getByText(/not started/i)).toBeInTheDocument();
  });

  it("keeps the viewer usable when case metadata fails to load", async () => {
    vi.mocked(fetchCase).mockRejectedValue(new Error("boom"));
    renderViewer();

    expect(
      await screen.findByText(/case details are unavailable/i),
    ).toBeInTheDocument();
    // Viewer itself still renders.
    expect(await screen.findByText(/slice navigation/i)).toBeInTheDocument();
  });
});

describe("ViewerPage keyboard slice navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchCase).mockResolvedValue(mockCase);
    vi.mocked(loadNiftiFromUrl).mockResolvedValue(mockVolume);
  });

  /** The axial slider is the authoritative readout of the axial index. */
  function axialSlider(): HTMLInputElement {
    return screen.getByLabelText(/axial slice slider/i) as HTMLInputElement;
  }

  it("steps the active plane forward with KeyL and back with KeyJ", async () => {
    const user = userEventDefault.setup();
    renderViewer();
    await screen.findByText(/slice navigation/i);

    // 8-voxel volume: initial axial index is the centre, 4.
    expect(axialSlider().value).toBe("4");

    await user.keyboard("{l>}{/l}");
    expect(axialSlider().value).toBe("5");

    await user.keyboard("{j>}{/j}");
    await user.keyboard("{j>}{/j}");
    expect(axialSlider().value).toBe("3");
  });

  it("clamps at the volume bounds instead of wrapping", async () => {
    const user = userEventDefault.setup();
    renderViewer();
    await screen.findByText(/slice navigation/i);

    for (let i = 0; i < 10; i += 1) await user.keyboard("{j>}{/j}");
    expect(axialSlider().value).toBe("0");

    for (let i = 0; i < 20; i += 1) await user.keyboard("{l>}{/l}");
    expect(axialSlider().value).toBe("7");
  });

  it("cycles the active plane with KeyK so all three are reachable", async () => {
    const user = userEventDefault.setup();
    renderViewer();
    await screen.findByText(/slice navigation/i);

    const coronal = screen.getByLabelText(/coronal slice slider/i) as HTMLInputElement;
    expect(coronal.value).toBe("4");

    // Axial -> coronal, then step coronal rather than axial.
    await user.keyboard("{k>}{/k}");
    await user.keyboard("{l>}{/l}");

    expect(coronal.value).toBe("5");
    expect(axialSlider().value).toBe("4");
  });
});
