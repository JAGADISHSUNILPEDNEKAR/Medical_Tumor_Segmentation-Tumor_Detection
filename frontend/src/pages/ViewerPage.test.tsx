/**
 * ViewerPage integration tests.
 *
 * Tests component rendering, loading states, disclaimer visibility,
 * and control presence. Does NOT test canvas rendering or WebGL
 * (those are tested through the unit tests on sliceExtraction and meshGeneration).
 */

import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import { ViewerPage } from "./ViewerPage";

// Mock the nifti loader to avoid real network requests
vi.mock("../lib/nifti", () => ({
  loadNiftiFromUrl: vi.fn(),
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

import { loadNiftiFromUrl } from "../lib/nifti";
import type { NiftiVolume } from "../types/viewer";

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
