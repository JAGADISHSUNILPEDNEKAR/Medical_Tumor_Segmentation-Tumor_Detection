import { MODALITIES, type Modality } from "../../types/viewer";

interface ModalitySelectorProps {
  value: Modality;
  onChange: (modality: Modality) => void;
  disabled?: boolean;
}

const MODALITY_LABELS: Record<Modality, string> = {
  flair: "FLAIR",
  t1: "T1",
  t1ce: "T1ce",
  t2: "T2",
};

export function ModalitySelector({ value, onChange, disabled }: ModalitySelectorProps) {
  return (
    <div className="flex items-center gap-3">
      <label
        htmlFor="modality-selector"
        className="text-sm font-medium text-ink-700"
      >
        Modality
      </label>
      <select
        id="modality-selector"
        value={value}
        onChange={(e) => onChange(e.target.value as Modality)}
        disabled={disabled}
        className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm text-ink-950 shadow-sm transition-colors hover:border-accent-600 focus:border-accent-600 focus:outline-none focus:ring-2 focus:ring-accent-600/20 disabled:opacity-50"
      >
        {MODALITIES.map((m) => (
          <option key={m} value={m}>
            {MODALITY_LABELS[m]}
          </option>
        ))}
      </select>
      <span className="text-xs text-ink-500">
        Selected modality: {MODALITY_LABELS[value]}
      </span>
    </div>
  );
}
