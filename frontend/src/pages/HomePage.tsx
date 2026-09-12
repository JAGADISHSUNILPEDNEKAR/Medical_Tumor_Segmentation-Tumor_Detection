import { Link } from "react-router-dom";
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
          Upload a four-modality BraTS MRI case, then inspect validation before
          any future inference step. The trained model is developed separately
          and is not loaded.
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
          Assign T1, T1ce, T2, and FLAIR NIfTI volumes to explicit modality
          slots. Optional ground-truth <code>seg</code> is stored for later
          evaluation and is not scored in this phase.
        </p>
        <Link
          to="/upload"
          className="mt-5 inline-block border border-accent-700 bg-accent-700 px-4 py-2 text-sm font-medium text-white hover:bg-accent-600"
        >
          Upload a case
        </Link>
        <p className="mt-2 text-sm text-ink-500">
          Next action: validate four modalities. Inference remains unavailable.
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
