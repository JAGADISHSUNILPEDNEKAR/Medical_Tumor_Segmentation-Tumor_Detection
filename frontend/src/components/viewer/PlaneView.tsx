import { useEffect, useRef, useCallback } from "react";
import { Loader2, AlertTriangle } from "lucide-react";

import type { NiftiVolume, PlaneId, RegionVisibility } from "../../types/viewer";
import {
  extractSlice,
  normalizeToGrayscale,
  compositeSlice,
  PLANE_AXIS,
} from "../../lib/sliceExtraction";

interface PlaneViewProps {
  /** Which anatomical plane this viewport shows. */
  plane: PlaneId;
  /** Current MRI volume (null while loading). */
  mriVolume: NiftiVolume | null;
  /** Current segmentation volume (null if unavailable). */
  segVolume: NiftiVolume | null;
  /** Current slice index for this plane. */
  sliceIndex: number;
  /** Total slices for this plane. */
  totalSlices: number;
  /** Whether loading. */
  loading: boolean;
  /** Error message. */
  error: string | null;
  /** Whether segmentation overlay is visible. */
  overlayVisible: boolean;
  /** Overlay opacity (0-1). */
  overlayOpacity: number;
  /** Per-region visibility. */
  regionVisibility: RegionVisibility;
  /** Called when user clicks on the canvas to set crosshair. */
  onCrosshairClick?: (plane: PlaneId, xFrac: number, yFrac: number) => void;
  /** Whether this plane currently receives KeyJ / KeyL slice steps. */
  active?: boolean;
  /** Called when this plane should become the keyboard target. */
  onActivate?: (plane: PlaneId) => void;
}

const PLANE_LABELS: Record<PlaneId, string> = {
  axial: "AXIAL",
  coronal: "CORONAL",
  sagittal: "SAGITTAL",
};

export function PlaneView({
  plane,
  mriVolume,
  segVolume,
  sliceIndex,
  totalSlices,
  loading,
  error,
  overlayVisible,
  overlayOpacity,
  regionVisibility,
  onCrosshairClick,
  active = false,
  onActivate,
}: PlaneViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const renderSlice = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !mriVolume) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const axis = PLANE_AXIS[plane];
    const mriSlice = extractSlice(mriVolume, axis, sliceIndex);
    const grayPixels = normalizeToGrayscale(mriSlice.values);

    // Extract segmentation slice if available
    let segValues: Float32Array | null = null;
    if (segVolume) {
      const segSlice = extractSlice(segVolume, axis, sliceIndex);
      segValues = segSlice.values;
    }

    // Composite MRI + overlay
    const imageData = compositeSlice(
      grayPixels,
      segValues,
      mriSlice.width,
      mriSlice.height,
      regionVisibility,
      overlayOpacity,
      overlayVisible,
    );

    // Size canvas to match slice dimensions
    canvas.width = mriSlice.width;
    canvas.height = mriSlice.height;
    ctx.putImageData(imageData, 0, 0);
  }, [mriVolume, segVolume, plane, sliceIndex, overlayVisible, overlayOpacity, regionVisibility]);

  useEffect(() => {
    renderSlice();
  }, [renderSlice]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    onActivate?.(plane);
    if (!onCrosshairClick || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const xFrac = (e.clientX - rect.left) / rect.width;
    const yFrac = (e.clientY - rect.top) / rect.height;
    onCrosshairClick(plane, xFrac, yFrac);
  };

  return (
    <div
      className={`flex flex-col overflow-hidden rounded-lg border bg-white shadow-sm transition-colors ${
        active ? "border-accent-700" : "border-slate-200"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-3 py-2">
        <span className="text-xs font-bold uppercase tracking-wider text-accent-700">
          {PLANE_LABELS[plane]}
          {active && <span className="sr-only"> (active for keyboard navigation)</span>}
        </span>
        <span className="font-mono text-xs text-ink-500">
          Slice {sliceIndex + 1} / {totalSlices}
        </span>
      </div>

      {/* Canvas area */}
      <div className="relative flex aspect-square items-center justify-center bg-black">
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/80">
            <Loader2 className="h-6 w-6 animate-spin text-accent-600" />
            <p className="mt-2 text-xs text-slate-400">
              Loading {PLANE_LABELS[plane].toLowerCase()} view…
            </p>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/80 p-4 text-center">
            <AlertTriangle className="h-6 w-6 text-red-400" />
            <p className="mt-2 text-xs text-red-300">{error}</p>
          </div>
        )}

        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="h-full w-full cursor-crosshair object-contain"
          style={{ imageRendering: "pixelated" }}
          aria-label={`${PLANE_LABELS[plane]} MRI slice ${sliceIndex + 1} of ${totalSlices}`}
        />
      </div>
    </div>
  );
}
