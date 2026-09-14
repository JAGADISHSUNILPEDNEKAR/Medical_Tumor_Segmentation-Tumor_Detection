import { TestTube2 } from "lucide-react";

interface ViewerToolbarProps {
  caseId: string;
}

export function ViewerToolbar({ caseId }: ViewerToolbarProps) {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3">
      <div className="flex items-center gap-3">
        <h1 className="font-display text-lg text-ink-950">Medical Image Viewer</h1>
        <span className="rounded bg-ink-950/5 px-2 py-0.5 font-mono text-xs text-ink-500">
          {caseId.slice(0, 8)}…
        </span>
      </div>
      <div className="flex items-center gap-2 rounded-full border border-amber-400/30 bg-amber-50 px-3 py-1.5">
        <TestTube2 className="h-4 w-4 text-amber-600" />
        <span className="text-xs font-semibold uppercase tracking-wider text-amber-700">
          Synthetic Demo
        </span>
      </div>
    </div>
  );
}
