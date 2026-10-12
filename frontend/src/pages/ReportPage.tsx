import { ArrowLeft, CheckCircle2, FileText, Loader2, Printer, AlertTriangle } from "lucide-react";
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

  const hasTumor = report.measurements.foreground_voxels > 0;
  const isSynthetic = report.provenance.synthetic;

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
            <p className="mt-2 text-sm text-ink-500">Multimodal MRI research</p>
          </div>
          <div className="text-right">
             <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-sm font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20">
               <CheckCircle2 className="h-4 w-4" />
               Completed
             </span>
          </div>
        </header>

        <section aria-labelledby="case-info-heading" className="mb-10 space-y-4">
          <h2 id="case-info-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">1. Case Information</h2>
          <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-3">
             <div>
                <dt className="text-ink-500 font-medium">Case ID</dt>
                <dd className="mt-1 font-mono text-ink-900">{report.case_id}</dd>
             </div>
             <div>
                <dt className="text-ink-500 font-medium">Job ID</dt>
                <dd className="mt-1 font-mono text-ink-900">{report.job_id}</dd>
             </div>
             <div>
                <dt className="text-ink-500 font-medium">Report Date</dt>
                <dd className="mt-1 font-medium text-ink-900">{new Date(report.created_at).toLocaleString()}</dd>
             </div>
          </dl>
        </section>

        <section aria-labelledby="imaging-info-heading" className="mb-10 space-y-4">
          <h2 id="imaging-info-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">2. Imaging Information</h2>
          <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-2">
             <div>
                <dt className="text-ink-500 font-medium">Modalities Analyzed</dt>
                <dd className="mt-1 font-medium text-ink-900">T1, T1ce, T2, FLAIR</dd>
             </div>
             <div>
                <dt className="text-ink-500 font-medium">Analysis Mode</dt>
                <dd className="mt-1 font-medium text-ink-900">{isSynthetic ? "Synthetic (Non-clinical)" : "Real PyTorch Model"}</dd>
             </div>
             <div>
                <dt className="text-ink-500 font-medium">Voxel Spacing (mm)</dt>
                <dd className="mt-1 font-medium text-ink-900">
                  {report.measurements.voxel_spacing_mm ? report.measurements.voxel_spacing_mm.map(v => v.toFixed(2)).join(" × ") : "Unknown"}
                </dd>
             </div>
          </dl>
        </section>

        <section aria-labelledby="findings-heading" className="mb-10 space-y-4">
          <h2 id="findings-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">3. Segmentation Findings</h2>
          <div className="text-sm text-ink-900 space-y-2">
            <p>
              This is a factual summary of the regions detected by the inference service.
              It does not constitute a medical diagnosis.
            </p>
            <p>
              Segmentation regions {hasTumor ? "were" : "were not"} detected in this case.
            </p>
            {hasTumor && (
              <ul className="list-disc pl-5 mt-2 space-y-1">
                {report.measurements.regions?.filter(r => r.present).map(r => (
                   <li key={r.label}>
                     The region corresponding to {REGION_CONFIG[r.label]?.name ?? r.region} contains voxels.
                   </li>
                ))}
              </ul>
            )}
            {report.evaluation?.available && (
              <p className="mt-2 text-ink-600 italic">
                Ground truth segmentation is available for this case, enabling quantitative evaluation.
              </p>
            )}
          </div>
        </section>

        <section aria-labelledby="tumor-regions-heading" className="mb-10 space-y-4">
           <h2 id="tumor-regions-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">4 & 5. Tumor Regions & Quantitative Measurements</h2>
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

        <section aria-labelledby="model-info-heading" className="mb-10 space-y-4">
          <h2 id="model-info-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">6. Model Information (Provenance)</h2>
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
                <dd className="mt-1 font-medium text-ink-900">{isSynthetic ? "Synthetic" : "Clinical"}</dd>
             </div>
             <div>
                <dt className="text-xs font-semibold uppercase text-ink-500">Checkpoint ID</dt>
                <dd className="mt-1 font-medium text-ink-900 truncate" title={report.provenance.checkpoint_id ?? ""}>
                    {report.provenance.checkpoint_id ?? "N/A"}
                </dd>
             </div>
          </div>
          {report.provenance.description && (
             <p className="text-sm text-ink-600 italic">"{report.provenance.description}"</p>
          )}
        </section>

        {report.evaluation?.available && report.evaluation.per_class ? (
           <section aria-labelledby="evaluation-heading" className="mb-10 space-y-4">
              <h2 id="evaluation-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">7. Evaluation Metrics</h2>
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
        ) : (
           <section aria-labelledby="evaluation-heading" className="mb-10 space-y-4">
              <h2 id="evaluation-heading" className="text-lg font-semibold text-ink-900 border-b border-slate-100 pb-2">7. Evaluation Metrics</h2>
              <p className="text-sm text-ink-600">Evaluation metrics are not available because no ground truth segmentation was provided for this case.</p>
           </section>
        )}

        <section aria-labelledby="limitations-heading" className="mb-10 space-y-4 border border-caution-800/20 bg-caution-50/50 p-6 rounded-lg print:border-none print:p-0">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="h-5 w-5 text-amber-600" aria-hidden />
            <h2 id="limitations-heading" className="text-lg font-semibold text-caution-800">8. Limitations & Disclaimers</h2>
          </div>
          <ul className="list-disc pl-5 text-sm text-caution-800 space-y-2">
            <li><strong>Research Prototype:</strong> This software is a research prototype and is <strong>not a medical device</strong>.</li>
            <li><strong>Not Clinically Validated:</strong> The outputs are not clinically validated and must not be used for diagnosis, treatment, or clinical decision-making. No clinical diagnosis is produced.</li>
            <li><strong>Model Dependence:</strong> Segmentation performance depends heavily on the training data distribution and the specific trained model loaded.</li>
            <li><strong>Metrics Availability:</strong> Quantitative evaluation metrics (such as Dice score) require the presence of a ground truth segmentation and are unavailable otherwise.</li>
            <li><strong>Performance Variables:</strong> Inference performance (speed) depends significantly on the underlying hardware (CPU vs. GPU) and the specific patch size configured.</li>
            {isSynthetic && (
              <li><strong>Synthetic Mode:</strong> The current result was produced by a synthetic geometric fixture. It is not a clinical prediction and should only be used to verify software pipeline connectivity.</li>
            )}
          </ul>
        </section>

      </div>
    </div>
  );
}
