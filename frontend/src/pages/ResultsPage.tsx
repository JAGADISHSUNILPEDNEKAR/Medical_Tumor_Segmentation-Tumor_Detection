import {
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  Eye,
  FileX,
  Loader2,
  TestTube2,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { fetchJob, fetchResult } from "../lib/api";
import { ApiError, type JobStatusResponse, type ResultResponse } from "../types/api";
import { REGION_CONFIG } from "../types/viewer";

/** Milliseconds between job-status polls while a job is QUEUED or RUNNING. */
const POLL_INTERVAL_MS = 1000;

/** Terminal job states: polling stops once one of these is reached. */
const TERMINAL_STATUSES = new Set(["COMPLETED", "FAILED"]);

function statusTone(status: string): string {
  if (status === "COMPLETED") return "text-emerald-700";
  if (status === "FAILED") return "text-red-700";
  return "text-accent-700";
}

function StatusIcon({ status }: { status: string }) {
  if (status === "FAILED") {
    return <AlertTriangle className="h-5 w-5 text-red-600" aria-hidden />;
  }
  if (status === "COMPLETED") {
    return <CheckCircle2 className="h-5 w-5 text-emerald-600" aria-hidden />;
  }
  return <Loader2 className="h-5 w-5 animate-spin text-accent-700" aria-hidden />;
}

export function ResultsPage() {
  const { caseId, jobId } = useParams<{ caseId: string; jobId: string }>();
  const navigate = useNavigate();
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<ResultResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    async function poll() {
      try {
        const data = await fetchJob(jobId!);
        if (cancelled) return;
        setJob(data);

        if (data.status === "COMPLETED" && data.result_id) {
          const resultData = await fetchResult(data.result_id);
          if (!cancelled) setResult(resultData);
          return;
        }
        if (TERMINAL_STATUSES.has(data.status)) return;

        timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not reach the analysis service. Check that the backend is running, then reload.",
        );
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [jobId]);

  if (!jobId) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16 text-center">
        <h1 className="font-display text-2xl text-ink-950">No job selected</h1>
        <p className="mt-2 text-sm text-ink-500">
          Start from an upload to run an analysis and see its results here.
        </p>
        <Link
          to="/upload"
          className="mt-6 inline-flex items-center gap-2 rounded-lg bg-accent-700 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-accent-600"
        >
          Go to upload
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 pb-20">
      <header className="space-y-2 pt-6">
        <nav
          aria-label="Breadcrumb"
          className="flex items-center gap-2 text-sm text-ink-500"
        >
          <span>Jobs</span>
          <ChevronRight className="h-4 w-4" aria-hidden />
          <span className="truncate font-mono text-xs">{jobId}</span>
        </nav>
        <h1 className="font-display text-3xl text-ink-950">Inference result</h1>
      </header>

      <div className="flex items-start gap-3 rounded-lg border border-amber-400/40 bg-caution-50 p-4">
        <TestTube2 className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-700" aria-hidden />
        <div className="space-y-1">
          <h2 className="text-sm font-semibold text-caution-800">Synthetic demo mode</h2>
          <p className="text-sm leading-relaxed text-caution-800">
            This is a non-clinical environment. The segmentation and measurements
            below are produced by a deterministic geometric mock, not by a trained
            medical model. They must not be used for diagnosis or treatment.
          </p>
        </div>
      </div>

      {error ? (
        <div
          role="alert"
          className="flex flex-col items-center justify-center space-y-4 rounded-lg border border-red-200 bg-red-50 p-8 text-center"
        >
          <FileX className="h-10 w-10 text-red-600" aria-hidden />
          <div>
            <h2 className="font-semibold text-red-800">Could not load this job</h2>
            <p className="mt-1 text-sm text-red-700">{error}</p>
          </div>
        </div>
      ) : !job ? (
        <div className="flex flex-col items-center justify-center space-y-3 rounded-lg border border-slate-200 bg-white p-12">
          <Loader2 className="h-7 w-7 animate-spin text-accent-700" aria-hidden />
          <p className="text-sm text-ink-500">Loading job status…</p>
        </div>
      ) : (
        <>
          <section
            aria-label="Execution status"
            className="space-y-6 rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-ink-950">Execution status</h2>
              <div className="flex items-center gap-2">
                <StatusIcon status={job.status} />
                <span
                  className={`text-sm font-semibold ${statusTone(job.status)}`}
                  role="status"
                  aria-live="polite"
                >
                  {job.status}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between font-mono text-xs text-ink-500">
                <span>Progress</span>
                <span className="tabular-nums">{job.progress}%</span>
              </div>
              <div
                className="h-2 overflow-hidden rounded-full bg-slate-200"
                role="progressbar"
                aria-valuenow={job.progress}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Inference progress"
              >
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-out ${
                    job.status === "FAILED" ? "bg-red-600" : "bg-accent-700"
                  }`}
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>

            <dl className="grid gap-x-6 gap-y-2 border-t border-slate-100 pt-4 text-sm sm:grid-cols-2">
              <div className="flex justify-between gap-4">
                <dt className="text-ink-500">Inference source</dt>
                <dd className="font-medium text-ink-700">{job.inference_source}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-ink-500">Model version</dt>
                <dd className="font-medium text-ink-700">
                  {job.model_version ?? "Not registered"}
                </dd>
              </div>
            </dl>

            {job.status === "FAILED" && job.error_message && (
              <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4">
                <p className="text-sm font-semibold text-red-800">
                  Error: {job.error_code}
                </p>
                <p className="mt-1 text-sm text-red-700">{job.error_message}</p>
              </div>
            )}
          </section>

          {result && (
            <section aria-label="Volume measurements" className="space-y-6">
              <div className="flex items-center gap-3">
                <h2 className="font-display text-2xl text-ink-950">
                  Volume measurements
                </h2>
                <span className="rounded-full border border-amber-400/40 bg-caution-50 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-caution-800">
                  Synthetic
                </span>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="flex flex-col justify-center rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                  <p className="text-sm font-medium text-ink-500">
                    Total foreground volume
                  </p>
                  <p className="mt-2 flex items-baseline gap-2">
                    <span className="font-display text-4xl tabular-nums text-ink-950">
                      {result.measurements.foreground_volume_cm3.toFixed(2)}
                    </span>
                    <span className="font-medium text-ink-500">cm³</span>
                  </p>
                  <p className="mt-1 text-xs tabular-nums text-ink-500">
                    {result.measurements.foreground_voxels.toLocaleString()} voxels ·{" "}
                    {result.measurements.foreground_volume_mm3.toFixed(0)} mm³
                  </p>
                </div>

                <div className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
                  <h3 className="text-sm font-medium text-ink-500">
                    Regional breakdown
                  </h3>
                  <ul className="space-y-3">
                    {Object.entries(result.measurements.regions).map(([name, region]) => {
                      // Colour is keyed off the BraTS label, never the display
                      // string, so it stays in step with the segmentation viewer.
                      const config = REGION_CONFIG[region.label];
                      return (
                        <li
                          key={name}
                          className="flex items-center justify-between gap-4"
                        >
                          <span className="flex items-center gap-3">
                            <span
                              className="inline-block h-3 w-3 flex-shrink-0 rounded-full"
                              style={{ backgroundColor: config?.color ?? "#64748b" }}
                              aria-hidden
                            />
                            <span className="font-medium text-ink-700">
                              {config?.name ?? name}
                            </span>
                            <span className="text-xs text-ink-500">
                              {config?.fullName ?? name}
                            </span>
                          </span>
                          <span className="whitespace-nowrap tabular-nums text-ink-700">
                            {region.volume_cm3.toFixed(2)}{" "}
                            <span className="text-xs text-ink-500">cm³</span>
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              </div>

              {result.segmentation.available && caseId && (
                <button
                  type="button"
                  onClick={() => navigate(`/cases/${caseId}/viewer`)}
                  className="flex w-full items-center justify-center gap-2 rounded-lg border border-accent-700 bg-accent-700 p-4 font-semibold text-white transition-colors hover:bg-accent-600"
                >
                  <Eye className="h-5 w-5" aria-hidden />
                  Open medical image viewer
                </button>
              )}
            </section>
          )}
        </>
      )}
    </div>
  );
}
