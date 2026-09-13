import { AlertTriangle, CheckCircle2, ChevronRight, FileX, Loader2, TestTube2 } from "lucide-react";
import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

interface JobStatus {
  job_id: string;
  case_id: string;
  status: string;
  progress: number;
  inference_source: string;
  result_id?: string;
  error_code?: string;
  error_message?: string;
}

interface ResultData {
  result_id: string;
  job_id: string;
  case_id: string;
  inference_source: string;
  measurements: {
    synthetic: boolean;
    description: string;
    foreground_volume_cm3: number;
    foreground_volume_mm3: number;
    regions: Record<
      string,
      {
        label: number;
        voxel_count: number;
        volume_mm3: number;
        volume_cm3: number;
      }
    >;
  };
  segmentation: {
    available: boolean;
  };
}

export function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [result, setResult] = useState<ResultData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let isPolling = true;

    const pollJob = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/jobs/${jobId}`);
        if (!response.ok) throw new Error("Failed to fetch job status");
        
        const data: JobStatus = await response.json();
        if (!isPolling) return;
        setJob(data);

        if (data.status === "COMPLETED" && data.result_id) {
          fetchResult(data.result_id);
          isPolling = false;
        } else if (data.status === "FAILED") {
          isPolling = false;
        } else {
          // Poll again
          setTimeout(pollJob, 1000);
        }
      } catch (err) {
        if (!isPolling) return;
        setError(err instanceof Error ? err.message : "An error occurred");
        isPolling = false;
      }
    };

    pollJob();

    return () => {
      isPolling = false;
    };
  }, [jobId]);

  const fetchResult = async (resultId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/results/${resultId}`);
      if (!response.ok) throw new Error("Failed to fetch result");
      const data: ResultData = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load results");
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      
      {/* HEADER */}
      <header className="space-y-2 pt-6">
        <div className="flex items-center space-x-2 text-sm text-slate-400">
          <span>Jobs</span>
          <ChevronRight className="w-4 h-4" />
          <span className="truncate font-mono">{jobId}</span>
        </div>
        <h1 className="text-3xl font-light text-slate-100 tracking-tight">
          Inference Result
        </h1>
      </header>

      {/* GLOBAL MOCK WARNING */}
      <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-2xl flex gap-4 items-start">
        <div className="bg-amber-500/20 p-2 rounded-xl text-amber-400 shrink-0">
          <TestTube2 className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-amber-400 font-medium tracking-wide">Synthetic Demo Mode</h3>
          <p className="text-amber-200/70 text-sm mt-1 leading-relaxed max-w-2xl">
            This is a non-clinical environment. The displayed segmentation and measurements are generated using deterministic geometric algorithms (ellipsoids) and are NOT the output of a trained medical model.
          </p>
        </div>
      </div>

      {error ? (
        <div className="p-6 bg-red-500/10 border border-red-500/20 rounded-2xl flex flex-col items-center justify-center text-center space-y-4">
          <FileX className="w-12 h-12 text-red-400" />
          <div>
            <h2 className="text-red-400 font-medium">Error Loading Job</h2>
            <p className="text-red-300/70 text-sm mt-1">{error}</p>
          </div>
        </div>
      ) : !job ? (
        <div className="p-12 border border-slate-800 rounded-2xl flex flex-col items-center justify-center space-y-4">
          <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          <p className="text-slate-400 animate-pulse">Initializing...</p>
        </div>
      ) : (
        <>
          {/* JOB STATUS CARD */}
          <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium text-slate-200">Execution Status</h2>
              <div className="flex items-center gap-2">
                {job.status === "FAILED" && <AlertTriangle className="w-5 h-5 text-red-400" />}
                {job.status === "COMPLETED" && <CheckCircle2 className="w-5 h-5 text-emerald-400" />}
                {(job.status === "QUEUED" || job.status === "RUNNING") && (
                  <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                )}
                <span className={`text-sm font-medium ${
                  job.status === "COMPLETED" ? "text-emerald-400" :
                  job.status === "FAILED" ? "text-red-400" :
                  "text-blue-400"
                }`}>
                  {job.status}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm text-slate-400 font-mono">
                <span>Progress</span>
                <span>{job.progress}%</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-500 ease-out rounded-full ${
                    job.status === "FAILED" ? "bg-red-500" : "bg-blue-500"
                  }`}
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            </div>

            {job.status === "FAILED" && job.error_message && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
                <p className="text-red-400 text-sm font-medium">Error: {job.error_code}</p>
                <p className="text-red-300/80 text-sm mt-1">{job.error_message}</p>
              </div>
            )}
          </div>

          {/* RESULTS VISUALIZATION (When COMPLETED) */}
          {result && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center gap-4">
                <h2 className="text-2xl font-light text-slate-200">Volume Measurements</h2>
                <span className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium rounded-full tracking-wide">
                  SYNTHETIC
                </span>
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                {/* Total Volume */}
                <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl flex flex-col justify-center shadow-xl">
                  <p className="text-sm font-medium text-slate-400">Total Foreground Volume</p>
                  <div className="mt-2 flex items-baseline gap-2">
                    <span className="text-4xl font-light text-slate-100 tabular-nums">
                      {result.measurements.foreground_volume_cm3.toFixed(2)}
                    </span>
                    <span className="text-slate-500 font-medium">cm³</span>
                  </div>
                </div>

                {/* Subregions */}
                <div className="p-6 bg-slate-900 border border-slate-800 rounded-2xl shadow-xl space-y-4">
                  <h3 className="text-sm font-medium text-slate-400">Regional Breakdown</h3>
                  <div className="space-y-3">
                    {Object.entries(result.measurements.regions).map(([name, r]) => (
                      <div key={name} className="flex justify-between items-center group">
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            name === "ED" ? "bg-emerald-400" :
                            name === "NCR" ? "bg-red-400" :
                            "bg-yellow-400"
                          }`} />
                          <span className="text-slate-300 font-medium group-hover:text-slate-100 transition-colors">
                            {name}
                          </span>
                        </div>
                        <div className="text-slate-400 group-hover:text-slate-200 transition-colors tabular-nums">
                          {r.volume_cm3.toFixed(2)} <span className="text-slate-600 text-xs">cm³</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="p-4 bg-slate-800/50 border border-slate-800 rounded-xl text-center">
                <p className="text-slate-400 text-sm">
                  Clinical 3D Viewer integration is scheduled for Phase 4.
                </p>
              </div>

            </div>
          )}
        </>
      )}
    </div>
  );
}
