import { ArrowLeft, CheckCircle2, FileText, Loader2, Printer } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchReport } from "../lib/api";
import { ApiError, type ReportResponse } from "../types/api";
import { REGION_CONFIG } from "../types/viewer";

export function ReportPage() {
  const { caseId, jobId } = useParams<{ caseId: string; jobId: string }>();
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    
    async function load() {
      try {
        // Find result ID from job
        const { fetchJob } = await import("../lib/api");
        const jobData = await fetchJob(jobId!);
        
        if (jobData.result_id) {
            const data = await fetchReport(jobData.result_id);
            setReport(data);
        } else {
            setError("No result available for this job yet.");
        }
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Failed to load report.");
      }
    }
    
    load();
  }, [jobId]);

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-16 text-center">
        <h1 className="font-display text-2xl text-red-700">Error loading report</h1>
        <p className="mt-2 text-ink-500">{error}</p>
        <Link to={`/cases/${caseId}/jobs/${jobId}`} className="mt-6 inline-flex text-accent-700 hover:underline">
          Return to results
        </Link>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center space-y-3 p-12">
        <Loader2 className="h-7 w-7 animate-spin text-accent-700" />
        <p className="text-sm text-ink-500">Generating report…</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 pb-20 print:p-0 print:max-w-none">
      <div className="flex items-center justify-between pt-6 print:hidden">
        <Link
          to={`/cases/${caseId}/jobs/${jobId}`}
          className="inline-flex items-center gap-2 text-sm font-medium text-ink-500 hover:text-ink-900 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to results
        </Link>
        <button
          type="button"
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded bg-ink-900 px-3 py-1.5 text-sm font-medium text-white shadow hover:bg-ink-800 transition-colors"
        >
          <Printer className="h-4 w-4" />
          Print report
        </button>
      </div>

      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-8 shadow-sm print:border-none print:shadow-none print:p-0">
        <header className="mb-8 flex items-start justify-between border-b border-slate-100 pb-6">
          <div>
            <h1 className="flex items-center gap-2 font-display text-3xl text-ink-950">
              <FileText className="h-8 w-8 text-accent-700" />
              Segmentation Report
            </h1>
            <p className="mt-2 text-sm text-ink-500">Case ID: {report.case_id}</p>
            <p className="text-sm text-ink-500">Report Date: {new Date(report.created_at).toLocaleString()}</p>
          </div>
          <div className="text-right">
             <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-sm font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
               <CheckCircle2 className="h-4 w-4" />
               Completed
             </span>
          </div>
        </header>

        <section className="mb-10 space-y-4">
          <h2 className="text-lg font-semibold text-ink-900">Provenance</h2>
          <div className="grid grid-cols-2 gap-4 rounded-lg bg-slate-50 p-4 sm:grid-cols-4">
             <div>
                <dt className="text-xs font-semibold uppercase text-ink-500">Source</dt>
                <dd className="mt-1 font-medium text-ink-900">{report.provenance.inference_source}</dd>
             </div>
             <div>
                <dt className="text-xs font-semibold uppercase text-ink-500">Model</dt>
                <dd className="mt-1 font-medium text-ink-900">{report.provenance.model_version ?? "N/A"}</dd>
             </div>
             <div>
                <dt className="text-xs font-semibold uppercase text-ink-500">Mode</dt>
                <dd className="mt-1 font-medium text-ink-900">{report.provenance.synthetic ? "Synthetic" : "Clinical"}</dd>
             </div>
             <div>
                <dt className="text-xs font-semibold uppercase text-ink-500">Checkpoint</dt>
                <dd className="mt-1 font-medium text-ink-900 truncate" title={report.provenance.checkpoint_id ?? ""}>
                    {report.provenance.checkpoint_id ?? "N/A"}
                </dd>
             </div>
          </div>
          {report.provenance.description && (
             <p className="text-sm text-ink-600 italic">"{report.provenance.description}"</p>
          )}
        </section>

        <section className="mb-10 space-y-4">
           <h2 className="text-lg font-semibold text-ink-900">Volumetric Measurements</h2>
           <div className="overflow-hidden rounded-lg border border-slate-200">
               <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                   <thead className="bg-slate-50 text-ink-500">
                       <tr>
                           <th scope="col" className="px-6 py-3 font-semibold uppercase">Region</th>
                           <th scope="col" className="px-6 py-3 font-semibold uppercase">Voxels</th>
                           <th scope="col" className="px-6 py-3 font-semibold uppercase">Volume (mm³)</th>
                           <th scope="col" className="px-6 py-3 font-semibold uppercase">Volume (cm³)</th>
                       </tr>
                   </thead>
                   <tbody className="divide-y divide-slate-200 bg-white font-mono text-ink-900">
                      {report.measurements.regions && report.measurements.regions.map(region => {
                          const config = REGION_CONFIG[region.label];
                          return (
                              <tr key={region.label}>
                                  <td className="px-6 py-4 font-sans font-medium">
                                      <div className="flex items-center gap-2">
                                          <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: config?.color ?? "#94a3b8" }} />
                                          {config?.name ?? region.region}
                                      </div>
                                  </td>
                                  <td className="px-6 py-4">{region.voxel_count.toLocaleString()}</td>
                                  <td className="px-6 py-4">{region.volume_mm3.toFixed(1)}</td>
                                  <td className="px-6 py-4 font-semibold text-accent-700">{region.volume_cm3.toFixed(2)}</td>
                              </tr>
                          );
                      })}
                      <tr className="bg-slate-50/50">
                          <td className="px-6 py-4 font-sans font-bold">Total Foreground</td>
                          <td className="px-6 py-4 font-bold">{report.measurements.foreground_voxels.toLocaleString()}</td>
                          <td className="px-6 py-4 font-bold">{report.measurements.foreground_volume_mm3.toFixed(1)}</td>
                          <td className="px-6 py-4 font-bold text-accent-700">{report.measurements.foreground_volume_cm3.toFixed(2)}</td>
                      </tr>
                   </tbody>
               </table>
           </div>
        </section>

        {report.evaluation?.available && report.evaluation.per_class && (
           <section className="mb-10 space-y-4">
              <h2 className="text-lg font-semibold text-ink-900">Evaluation Metrics</h2>
              <div className="overflow-hidden rounded-lg border border-slate-200">
                 <table className="min-w-full divide-y divide-slate-200 text-left text-sm">
                     <thead className="bg-slate-50 text-ink-500">
                         <tr>
                             <th scope="col" className="px-6 py-3 font-semibold uppercase">Region</th>
                             <th scope="col" className="px-6 py-3 font-semibold uppercase">Dice Score</th>
                             <th scope="col" className="px-6 py-3 font-semibold uppercase">95% Hausdorff (mm)</th>
                         </tr>
                     </thead>
                     <tbody className="divide-y divide-slate-200 bg-white font-mono text-ink-900">
                        {report.evaluation.per_class.map(cls => (
                            <tr key={cls.label}>
                                <td className="px-6 py-4 font-sans font-medium">{cls.class_name}</td>
                                <td className="px-6 py-4">{cls.dice ? cls.dice.value.toFixed(4) : "N/A"}</td>
                                <td className="px-6 py-4">{cls.hd95?.defined && cls.hd95.value_mm != null ? cls.hd95.value_mm.toFixed(2) : "N/A"}</td>
                            </tr>
                        ))}
                        <tr className="bg-slate-50/50">
                            <td className="px-6 py-4 font-sans font-bold">Mean</td>
                            <td className="px-6 py-4 font-bold text-accent-700">{report.evaluation.mean_dice ? report.evaluation.mean_dice.value.toFixed(4) : "N/A"}</td>
                            <td className="px-6 py-4 font-bold text-accent-700">{report.evaluation.mean_hd95 ? report.evaluation.mean_hd95.value_mm?.toFixed(2) : "N/A"}</td>
                        </tr>
                     </tbody>
                 </table>
              </div>
           </section>
        )}
      </div>
    </div>
  );
}
