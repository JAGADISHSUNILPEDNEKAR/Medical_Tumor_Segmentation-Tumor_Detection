import { useCallback, useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Box, Loader2, AlertTriangle } from "lucide-react";

import type {
  Modality,
  NiftiVolume,
  PlaneId,
  SliceState,
  ViewerState,
} from "../types/viewer";
import { defaultRegionVisibility } from "../types/viewer";
import { loadNiftiFromUrl } from "../lib/nifti";
import { getSliceCount, PLANE_AXIS } from "../lib/sliceExtraction";

import { ViewerToolbar } from "../components/viewer/ViewerToolbar";
import { ViewerDisclaimer } from "../components/viewer/ViewerDisclaimer";
import { ModalitySelector } from "../components/viewer/ModalitySelector";
import { PlaneView } from "../components/viewer/PlaneView";
import { SliceControls } from "../components/viewer/SliceControls";
import { OverlayControls } from "../components/viewer/OverlayControls";
import { RegionVisibility } from "../components/viewer/RegionVisibility";
import { Tumor3DViewer } from "../components/viewer/Tumor3DViewer";

/** Default slice state when volume is unknown. */
function defaultSlices(): Record<PlaneId, SliceState> {
  return {
    axial: { index: 0, total: 1 },
    coronal: { index: 0, total: 1 },
    sagittal: { index: 0, total: 1 },
  };
}

/** Compute initial slice indices at the center of each axis. */
function computeSlices(volume: NiftiVolume): Record<PlaneId, SliceState> {
  return {
    axial: {
      index: Math.floor(getSliceCount(volume, PLANE_AXIS.axial) / 2),
      total: getSliceCount(volume, PLANE_AXIS.axial),
    },
    coronal: {
      index: Math.floor(getSliceCount(volume, PLANE_AXIS.coronal) / 2),
      total: getSliceCount(volume, PLANE_AXIS.coronal),
    },
    sagittal: {
      index: Math.floor(getSliceCount(volume, PLANE_AXIS.sagittal) / 2),
      total: getSliceCount(volume, PLANE_AXIS.sagittal),
    },
  };
}

export function ViewerPage() {
  const { caseId } = useParams<{ caseId: string }>();

  const [state, setState] = useState<ViewerState>({
    status: "loading",
    errorMessage: null,
    modality: "flair",
    mriVolume: null,
    segVolume: null,
    slices: defaultSlices(),
    overlayVisible: true,
    overlayOpacity: 0.6,
    regionVisibility: defaultRegionVisibility(),
    show3D: false,
  });

  const [loadingModality, setLoadingModality] = useState(false);

  // Load segmentation once
  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;

    async function loadSegmentation() {
      try {
        const seg = await loadNiftiFromUrl(caseId!, "segmentation");
        if (!cancelled) {
          setState((prev) => ({ ...prev, segVolume: seg }));
        }
      } catch (err) {
        console.warn("Could not load segmentation:", err);
        // Non-fatal: viewer works without overlay
      }
    }

    loadSegmentation();
    return () => { cancelled = true; };
  }, [caseId]);

  // Load MRI volume when modality changes
  useEffect(() => {
    if (!caseId) return;
    let cancelled = false;

    async function loadMRI() {
      setLoadingModality(true);
      try {
        const volume = await loadNiftiFromUrl(caseId!, state.modality);
        if (!cancelled) {
          const slices = computeSlices(volume);
          setState((prev) => ({
            ...prev,
            status: "loaded",
            errorMessage: null,
            mriVolume: volume,
            slices,
          }));
        }
      } catch (err) {
        if (!cancelled) {
          setState((prev) => ({
            ...prev,
            status: "error",
            errorMessage:
              err instanceof Error ? err.message : "Failed to load MRI volume",
          }));
        }
      } finally {
        if (!cancelled) setLoadingModality(false);
      }
    }

    loadMRI();
    return () => { cancelled = true; };
  }, [caseId, state.modality]);

  const handleModalityChange = useCallback((modality: Modality) => {
    setState((prev) => ({ ...prev, modality, mriVolume: null }));
  }, []);

  const handleSliceChange = useCallback((plane: PlaneId, index: number) => {
    setState((prev) => ({
      ...prev,
      slices: {
        ...prev.slices,
        [plane]: { ...prev.slices[plane], index },
      },
    }));
  }, []);

  const handleToggleOverlay = useCallback(() => {
    setState((prev) => ({
      ...prev,
      overlayVisible: !prev.overlayVisible,
    }));
  }, []);

  const handleOpacityChange = useCallback((opacity: number) => {
    setState((prev) => ({ ...prev, overlayOpacity: opacity }));
  }, []);

  const handleToggleRegion = useCallback((label: number) => {
    setState((prev) => ({
      ...prev,
      regionVisibility: {
        ...prev.regionVisibility,
        [label]: !prev.regionVisibility[label],
      },
    }));
  }, []);

  const handleShowAll = useCallback(() => {
    setState((prev) => ({
      ...prev,
      regionVisibility: defaultRegionVisibility(),
    }));
  }, []);

  const handleHideAll = useCallback(() => {
    setState((prev) => ({
      ...prev,
      regionVisibility: { 1: false, 2: false, 4: false },
    }));
  }, []);

  const handleToggle3D = useCallback(() => {
    setState((prev) => ({ ...prev, show3D: !prev.show3D }));
  }, []);

  /**
   * Crosshair synchronization: clicking on one plane updates the
   * other two planes' slice indices.
   */
  const handleCrosshairClick = useCallback(
    (plane: PlaneId, xFrac: number, yFrac: number) => {
      if (!state.mriVolume) return;

      setState((prev) => {
        const newSlices = { ...prev.slices };

        // Map click coordinates to slice indices on the other planes
        // Axial (axis 2): displays dim0 (x-axis) × dim1 (y-axis)
        // Coronal (axis 1): displays dim0 (x-axis) × dim2 (y-axis)
        // Sagittal (axis 0): displays dim1 (x-axis) × dim2 (y-axis)
        const vol = prev.mriVolume!;
        const [d0, d1, d2] = vol.shape;

        switch (plane) {
          case "axial": {
            // Click on axial → x maps to sagittal, y maps to coronal
            const sagIdx = Math.floor(xFrac * d0);
            const corIdx = Math.floor(yFrac * d1);
            newSlices.sagittal = {
              ...newSlices.sagittal,
              index: Math.max(0, Math.min(sagIdx, d0 - 1)),
            };
            newSlices.coronal = {
              ...newSlices.coronal,
              index: Math.max(0, Math.min(corIdx, d1 - 1)),
            };
            break;
          }
          case "coronal": {
            // Click on coronal → x maps to sagittal, y maps to axial
            const sagIdx = Math.floor(xFrac * d0);
            const axIdx = Math.floor(yFrac * d2);
            newSlices.sagittal = {
              ...newSlices.sagittal,
              index: Math.max(0, Math.min(sagIdx, d0 - 1)),
            };
            newSlices.axial = {
              ...newSlices.axial,
              index: Math.max(0, Math.min(axIdx, d2 - 1)),
            };
            break;
          }
          case "sagittal": {
            // Click on sagittal → x maps to coronal, y maps to axial
            const corIdx = Math.floor(xFrac * d1);
            const axIdx = Math.floor(yFrac * d2);
            newSlices.coronal = {
              ...newSlices.coronal,
              index: Math.max(0, Math.min(corIdx, d1 - 1)),
            };
            newSlices.axial = {
              ...newSlices.axial,
              index: Math.max(0, Math.min(axIdx, d2 - 1)),
            };
            break;
          }
        }

        return { ...prev, slices: newSlices };
      });
    },
    [state.mriVolume],
  );

  if (!caseId) {
    return (
      <div className="flex h-96 items-center justify-center">
        <p className="text-ink-500">No case ID specified.</p>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-paper-50">
      <ViewerToolbar caseId={caseId} />

      <div className="mx-auto w-full max-w-7xl space-y-6 px-4 py-6">
        {/* Back link */}
        <Link
          to={`/`}
          className="inline-flex items-center gap-1.5 text-sm text-accent-700 transition-colors hover:text-accent-600"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to home
        </Link>

        {/* Disclaimer */}
        <ViewerDisclaimer />

        {/* Controls row: modality + overlay */}
        <div className="grid gap-6 rounded-lg border border-slate-200 bg-white p-4 md:grid-cols-2">
          <ModalitySelector
            value={state.modality}
            onChange={handleModalityChange}
            disabled={loadingModality}
          />
          <OverlayControls
            overlayVisible={state.overlayVisible}
            onToggleOverlay={handleToggleOverlay}
            overlayOpacity={state.overlayOpacity}
            onOpacityChange={handleOpacityChange}
          />
        </div>

        {/* Error state */}
        {state.status === "error" && state.errorMessage && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4">
            <AlertTriangle className="h-5 w-5 flex-shrink-0 text-red-500" />
            <div>
              <p className="text-sm font-medium text-red-800">
                Failed to load viewer
              </p>
              <p className="text-xs text-red-600">{state.errorMessage}</p>
            </div>
          </div>
        )}

        {/* Loading state */}
        {state.status === "loading" && !state.mriVolume && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-accent-600" />
            <p className="mt-3 text-sm text-ink-500">Loading MRI volumes…</p>
          </div>
        )}

        {/* Three planes */}
        {state.mriVolume && (
          <>
            <div className="grid gap-4 md:grid-cols-3">
              {(["axial", "coronal", "sagittal"] as PlaneId[]).map((plane) => (
                <PlaneView
                  key={plane}
                  plane={plane}
                  mriVolume={state.mriVolume}
                  segVolume={state.segVolume}
                  sliceIndex={state.slices[plane].index}
                  totalSlices={state.slices[plane].total}
                  loading={loadingModality}
                  error={null}
                  overlayVisible={state.overlayVisible}
                  overlayOpacity={state.overlayOpacity}
                  regionVisibility={state.regionVisibility}
                  onCrosshairClick={handleCrosshairClick}
                />
              ))}
            </div>

            {/* Slice controls */}
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <SliceControls
                slices={state.slices}
                onSliceChange={handleSliceChange}
              />
            </div>

            {/* Region visibility */}
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <RegionVisibility
                visibility={state.regionVisibility}
                onToggle={handleToggleRegion}
                onShowAll={handleShowAll}
                onHideAll={handleHideAll}
              />
            </div>

            {/* 3D viewer toggle */}
            <div className="flex justify-center">
              <button
                onClick={handleToggle3D}
                className="inline-flex items-center gap-2 rounded-lg border border-accent-700 bg-accent-700 px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-accent-600"
                aria-label={state.show3D ? "Close 3D viewer" : "Open 3D viewer"}
              >
                <Box className="h-4 w-4" />
                {state.show3D ? "Close 3D View" : "Open 3D View"}
              </button>
            </div>

            {/* 3D Viewer */}
            {state.show3D && (
              <Tumor3DViewer
                segVolume={state.segVolume}
                regionVisibility={state.regionVisibility}
                loading={state.segVolume === null && state.status === "loading"}
                error={
                  state.segVolume === null && state.status !== "loading"
                    ? "No segmentation available for 3D visualization."
                    : null
                }
                onClose={handleToggle3D}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
