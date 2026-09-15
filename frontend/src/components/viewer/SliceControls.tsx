import type { PlaneId, SliceState } from "../../types/viewer";

interface SliceControlsProps {
  slices: Record<PlaneId, SliceState>;
  onSliceChange: (plane: PlaneId, index: number) => void;
  /** Plane that KeyJ / KeyL currently step through. */
  activePlane: PlaneId;
  onActivePlaneChange: (plane: PlaneId) => void;
}

const PLANE_LABELS: Record<PlaneId, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};

const PLANES: PlaneId[] = ["axial", "coronal", "sagittal"];

export function SliceControls({
  slices,
  onSliceChange,
  activePlane,
  onActivePlaneChange,
}: SliceControlsProps) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-500">
          Slice Navigation
        </h3>
        <p className="text-xs text-ink-500">
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-mono">J</kbd>
          {" / "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-mono">L</kbd>
          {" step "}
          <span className="font-medium text-ink-700">{PLANE_LABELS[activePlane]}</span>
          {" · "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-mono">I</kbd>
          {" / "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-mono">K</kbd>
          {" change plane"}
        </p>
      </div>
      {PLANES.map((plane) => {
        const state = slices[plane];
        const isActive = plane === activePlane;
        return (
          <div
            key={plane}
            className={`flex items-center gap-3 rounded px-2 py-1 transition-colors ${
              isActive ? "bg-accent-700/5" : ""
            }`}
          >
            <label
              htmlFor={`slice-${plane}`}
              className="w-20 text-right text-sm font-medium text-ink-700"
            >
              {PLANE_LABELS[plane]}
            </label>
            <input
              id={`slice-${plane}`}
              type="range"
              min={0}
              max={state.total - 1}
              value={state.index}
              onChange={(e) => onSliceChange(plane, Number(e.target.value))}
              onFocus={() => onActivePlaneChange(plane)}
              className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-slate-200 accent-accent-700"
              aria-label={`${PLANE_LABELS[plane]} slice slider`}
              aria-valuemin={1}
              aria-valuemax={state.total}
              aria-valuenow={state.index + 1}
            />
            <span className="w-20 text-right font-mono text-xs tabular-nums text-ink-500">
              {state.index + 1} / {state.total}
            </span>
          </div>
        );
      })}
    </div>
  );
}
