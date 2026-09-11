import { useEffect, useState } from "react";

import { fetchHealth, fetchModelInfo } from "../lib/api";
import { ApiError, type HealthResponse, type ModelInfoResponse } from "../types/api";

type LoadState<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; message: string };

export function HomePage() {
  const [health, setHealth] = useState<LoadState<HealthResponse>>({ status: "loading" });
  const [model, setModel] = useState<LoadState<ModelInfoResponse>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchHealth();
        if (!cancelled) {
          setHealth({ status: "ready", data });
        }
      } catch (error) {
        if (!cancelled) {
          setHealth({
            status: "error",
            message: describeFetchError(error),
          });
        }
      }

      try {
        const data = await fetchModelInfo();
        if (!cancelled) {
          setModel({ status: "ready", data });
        }
      } catch (error) {
        if (!cancelled) {
          setModel({
            status: "error",
            message: describeFetchError(error),
          });
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <section className="max-w-3xl">
        <h1 className="font-display text-4xl leading-tight text-ink-950">
          Inspectable brain tumor sub-region segmentation for research review
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-ink-700">
          Upload a four-modality BraTS MRI case, run segmentation through a
          versioned inference service, and review 2D slices, a 3D mesh, and
          quantitative metrics. The trained model is developed separately and
          is not loaded in this Phase 1 shell.
        </p>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-2" aria-label="System status">
        <article className="border border-ink-900/10 bg-white p-6">
          <h2 className="text-sm font-semibold uppercase tracking-[0.12em] text-ink-500">
            Backend health
          </h2>
          <HealthPanel state={health} />
        </article>

        <article className="border border-ink-900/10 bg-white p-6">
          <h2 className="text-sm font-semibold uppercase tracking-[0.12em] text-ink-500">
            Model registry
          </h2>
          <ModelPanel state={model} />
        </article>
      </section>

      <section className="mt-10 border border-ink-900/10 bg-white p-6">
        <h2 className="font-display text-2xl text-ink-950">Case upload</h2>
        <p className="mt-2 max-w-2xl text-ink-700">
          Four-modality NIfTI upload (T1, T1ce, T2, FLAIR) is implemented in
          Phase 2. This screen does not accept files and does not run
          inference.
        </p>
        <button
          type="button"
          disabled
          aria-disabled="true"
          className="mt-5 cursor-not-allowed border border-ink-900/20 bg-paper-100 px-4 py-2 text-sm font-medium text-ink-500"
        >
          Upload a case
        </button>
        <p className="mt-2 text-sm text-ink-500">
          Next action after Phase 2: validate modalities, create a case, then
          queue inference.
        </p>
      </section>
    </div>
  );
}

function HealthPanel({ state }: { state: LoadState<HealthResponse> }) {
  if (state.status === "loading") {
    return (
      <p className="mt-3 text-ink-700" aria-live="polite">
        Checking API…
      </p>
    );
  }

  if (state.status === "error") {
    return (
      <p className="mt-3 text-ink-700" role="alert">
        Cannot reach the API. {state.message} Start the backend on port 8000,
        then reload this page.
      </p>
    );
  }

  const { data } = state;
  return (
    <dl className="mt-4 space-y-2 text-sm" aria-live="polite">
      <StatusRow label="API status" value={data.status} />
      <StatusRow label="Model loaded" value={data.model_loaded ? "yes" : "no"} />
      <StatusRow label="Inference source" value={String(data.inference_source)} />
    </dl>
  );
}

function ModelPanel({ state }: { state: LoadState<ModelInfoResponse> }) {
  if (state.status === "loading") {
    return (
      <p className="mt-3 text-ink-700" aria-live="polite">
        Loading model information…
      </p>
    );
  }

  if (state.status === "error") {
    return (
      <p className="mt-3 text-ink-700" role="alert">
        Model information is unavailable. {state.message}
      </p>
    );
  }

  const { data } = state;
  return (
    <div className="mt-4 space-y-3 text-sm" aria-live="polite">
      <p className="text-ink-700">{data.message}</p>
      <dl className="space-y-2">
        <StatusRow label="Checkpoint" value={data.checkpoint_id ?? "none registered"} />
        <StatusRow label="Headline metrics" value="not available" />
        <StatusRow label="Expected modalities" value={data.input_modalities.join(", ")} />
      </dl>
    </div>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 border-b border-ink-900/5 py-1">
      <dt className="text-ink-500">{label}</dt>
      <dd className="font-medium text-ink-950">{value}</dd>
    </div>
  );
}

function describeFetchError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Network request failed.";
}
