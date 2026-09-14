import {
  FOREGROUND_LABELS,
  REGION_CONFIG,
  type RegionVisibility as RegionVisibilityType,
} from "../../types/viewer";

interface RegionVisibilityProps {
  visibility: RegionVisibilityType;
  onToggle: (label: number) => void;
  onShowAll: () => void;
  onHideAll: () => void;
}

export function RegionVisibility({
  visibility,
  onToggle,
  onShowAll,
  onHideAll,
}: RegionVisibilityProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-ink-500">
          Region Visibility
        </h3>
        <div className="flex gap-2">
          <button
            onClick={onShowAll}
            className="rounded px-2 py-0.5 text-xs font-medium text-accent-700 transition-colors hover:bg-accent-700/10"
            aria-label="Show all regions"
          >
            Show All
          </button>
          <button
            onClick={onHideAll}
            className="rounded px-2 py-0.5 text-xs font-medium text-ink-500 transition-colors hover:bg-ink-950/5"
            aria-label="Hide all regions"
          >
            Hide All
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-4">
        {FOREGROUND_LABELS.map((label) => {
          const cfg = REGION_CONFIG[label];
          if (!cfg) return null;
          const checked = visibility[label] ?? false;

          return (
            <label
              key={label}
              className="flex cursor-pointer items-center gap-2 text-sm"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(label)}
                className="h-4 w-4 rounded border-slate-300 accent-accent-700"
                aria-label={`Toggle ${cfg.name} visibility`}
              />
              <span
                className="inline-block h-3 w-3 rounded-full"
                style={{ backgroundColor: cfg.color }}
                aria-hidden
              />
              <span className="font-medium text-ink-700">{cfg.name}</span>
              <span className="text-xs text-ink-500">({cfg.fullName})</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
