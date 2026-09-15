/**
 * Case metadata panel for the analysis workspace (PRD §3).
 *
 * Shows case ID, modalities present, upload timestamp, and the live
 * inference job status so the reviewer can see the provenance of the
 * segmentation they are looking at.
 */

import { AlertTriangle, Loader2 } from "lucide-react";

import { ALL_MODALITY_LABELS } from "../../types/viewer";
import type { CaseResponse, JobStatusResponse } from "../../types/api";

interface CaseMetadataPanelProps {
  caseId: string;
  caseData: CaseResponse | null;
  job: JobStatusResponse | null;
  loading: boolean;
  error: string | null;
}

function formatTimestamp(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function jobTone(status: string | undefined): string {
  if (status === "COMPLETED") return "text-emerald-700";
  if (status === "FAILED") return "text-red-700";
  if (status === undefined) return "text-ink-500";
  return "text-accent-700";
}

export function CaseMetadataPanel({
  caseId,
  caseData,
  job,
  loading,
  error,
}: CaseMetadataPanelProps) {
  return (
    <section
      aria-label="Case metadata"
      className="rounded-lg border border-slate-200 bg-white p-4"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-ink-500">
          Case
        </h2>
        {loading && (
          <Loader2 className="h-4 w-4 animate-spin text-accent-700" aria-hidden />
        )}
      </div>

      {error ? (
        <p className="mt-3 flex items-center gap-2 text-sm text-ink-500">
          <AlertTriangle className="h-4 w-4 flex-shrink-0 text-amber-600" aria-hidden />
          {error}
        </p>
      ) : (
        <dl className="mt-3 grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className="text-xs text-ink-500">Case ID</dt>
            <dd className="mt-0.5 truncate font-mono text-xs text-ink-700" title={caseId}>
              {caseId}
            </dd>
          </div>

          <div>
            <dt className="text-xs text-ink-500">Modalities present</dt>
            <dd className="mt-0.5 flex flex-wrap gap-1">
              {caseData ? (
                ALL_MODALITY_LABELS.map(({ key, label }) => {
                  const present = caseData.modalities_present.includes(key);
                  return (
                    <span
                      key={key}
                      className={
                        present
                          ? "rounded bg-accent-700/10 px-1.5 py-0.5 text-xs font-medium text-accent-700"
                          : "rounded bg-ink-950/5 px-1.5 py-0.5 text-xs text-ink-500 line-through"
                      }
                    >
                      <span className="sr-only">
                        {label} {present ? "present" : "missing"}
                      </span>
                      <span aria-hidden>{label}</span>
                    </span>
                  );
                })
              ) : (
                <span className="text-xs text-ink-500">—</span>
              )}
            </dd>
          </div>

          <div>
            <dt className="text-xs text-ink-500">Uploaded</dt>
            <dd className="mt-0.5 text-xs text-ink-700">
              {caseData ? formatTimestamp(caseData.created_at) : "—"}
            </dd>
          </div>

          <div>
            <dt className="text-xs text-ink-500">Inference job</dt>
            <dd
              className={`mt-0.5 text-xs font-semibold ${jobTone(job?.status)}`}
              role="status"
              aria-live="polite"
            >
              {job ? `${job.status}${job.status === "RUNNING" ? ` · ${job.progress}%` : ""}` : "Not started"}
            </dd>
          </div>
        </dl>
      )}
    </section>
  );
}
