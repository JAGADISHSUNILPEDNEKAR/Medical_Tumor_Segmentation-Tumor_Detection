interface OverlayControlsProps {
  overlayVisible: boolean;
  onToggleOverlay: () => void;
  overlayOpacity: number;
  onOpacityChange: (opacity: number) => void;
}

export function OverlayControls({
  overlayVisible,
  onToggleOverlay,
  overlayOpacity,
  onOpacityChange,
}: OverlayControlsProps) {
  const percentValue = Math.round(overlayOpacity * 100);

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-500">
        Segmentation Overlay
      </h3>

      <div className="flex items-center gap-3">
        <label htmlFor="overlay-toggle" className="text-sm font-medium text-ink-700">
          Overlay
        </label>
        <button
          id="overlay-toggle"
          onClick={onToggleOverlay}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            overlayVisible ? "bg-accent-700" : "bg-slate-300"
          }`}
          role="switch"
          aria-checked={overlayVisible}
          aria-label="Toggle segmentation overlay"
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white shadow transition-transform ${
              overlayVisible ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
        <span className="text-xs text-ink-500">
          {overlayVisible ? "ON" : "OFF"}
        </span>
      </div>

      <div className="flex items-center gap-3">
        <label htmlFor="overlay-opacity" className="w-20 text-right text-sm font-medium text-ink-700">
          Opacity
        </label>
        <input
          id="overlay-opacity"
          type="range"
          min={0}
          max={100}
          value={percentValue}
          onChange={(e) => onOpacityChange(Number(e.target.value) / 100)}
          disabled={!overlayVisible}
          className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-slate-200 accent-accent-700 disabled:opacity-40"
          aria-label="Overlay opacity"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percentValue}
        />
        <span className="w-12 text-right font-mono text-xs tabular-nums text-ink-500">
          {percentValue}%
        </span>
      </div>
    </div>
  );
}
