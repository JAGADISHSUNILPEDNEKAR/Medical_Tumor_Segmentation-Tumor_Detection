import { useMemo, useState, type FormEvent } from "react";

import { completeCase, createCase, uploadCaseFile } from "../lib/api";
import {
  MODALITY_LABELS,
  OPTIONAL_MODALITIES,
  REQUIRED_MODALITIES,
  hasAllowedNiftiExtension,
  missingRequiredModalities,
  type Modality,
} from "../lib/uploadValidation";
import { ApiError, type CaseResponse } from "../types/api";

interface SlotState {
  file: File | null;
  clientError: string | null;
  progress: number | null;
  status: "empty" | "selected" | "uploading" | "uploaded" | "failed";
}

const EMPTY_SLOT: SlotState = {
  file: null,
  clientError: null,
  progress: null,
  status: "empty",
};

const SLOTS: Modality[] = [...REQUIRED_MODALITIES, ...OPTIONAL_MODALITIES];

export function UploadPage() {
  const [slots, setSlots] = useState<Record<Modality, SlotState>>({
    t1: { ...EMPTY_SLOT },
    t1ce: { ...EMPTY_SLOT },
    t2: { ...EMPTY_SLOT },
    flair: { ...EMPTY_SLOT },
    seg: { ...EMPTY_SLOT },
  });
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<CaseResponse | null>(null);

  const files = useMemo(() => {
    const selected: Partial<Record<Modality, File | null>> = {};
    for (const modality of SLOTS) {
      selected[modality] = slots[modality].file;
    }
    return selected;
  }, [slots]);

  const missing = missingRequiredModalities(files);

  function assignFile(modality: Modality, file: File | null) {
    setResult(null);
    setSubmitError(null);
    if (!file) {
      setSlots((current) => ({ ...current, [modality]: { ...EMPTY_SLOT } }));
      return;
    }
    if (!hasAllowedNiftiExtension(file.name)) {
      setSlots((current) => ({
        ...current,
        [modality]: {
          file: null,
          clientError: `${file.name} is not a NIfTI volume. Use .nii or .nii.gz.`,
          progress: null,
          status: "failed",
        },
      }));
      return;
    }
    setSlots((current) => ({
      ...current,
      [modality]: { file, clientError: null, progress: null, status: "selected" },
    }));
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitError(null);
    setResult(null);

    if (missing.length > 0) {
      const firstMissing = missing[0];
      if (firstMissing) {
        setSubmitError(`Required modality '${firstMissing.toUpperCase()}' is missing.`);
      }
      return;
    }

    const extensionError = SLOTS.find((modality) => slots[modality].clientError);
    if (extensionError) {
      setSubmitError(slots[extensionError].clientError);
      return;
    }

    setBusy(true);
    try {
      const created = await createCase();
      let latest: CaseResponse | null = null;
      for (const modality of SLOTS) {
        const file = slots[modality].file;
        if (!file) {
          continue;
        }
        setSlots((current) => ({
          ...current,
          [modality]: { ...current[modality], status: "uploading", progress: 0 },
        }));
        latest = await uploadCaseFile(created.case_id, modality, file, (percent) => {
          setSlots((current) => ({
            ...current,
            [modality]: { ...current[modality], progress: percent, status: "uploading" },
          }));
        });
        setSlots((current) => ({
          ...current,
          [modality]: { ...current[modality], status: "uploaded", progress: 100 },
        }));
      }
      latest = await completeCase(created.case_id);
      setResult(latest);
    } catch (error) {
      setSubmitError(describeError(error));
      if (error instanceof ApiError && error.code) {
        // Keep slot-level failure visible when a specific file is rejected.
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <header className="max-w-3xl">
        <h1 className="font-display text-4xl text-ink-950">Upload a BraTS MRI case</h1>
        <p className="mt-3 text-lg text-ink-700">
          Assign each NIfTI volume to its modality slot. PNG, JPEG, and DICOM
          files are not accepted. Inference is not run in this phase.
        </p>
      </header>

      <form className="mt-8 space-y-6" onSubmit={(event) => void onSubmit(event)}>
        <div className="grid gap-4 md:grid-cols-2">
          {SLOTS.map((modality) => (
            <ModalitySlot
              key={modality}
              modality={modality}
              required={modality !== "seg"}
              slot={slots[modality]}
              disabled={busy}
              onFile={(file) => assignFile(modality, file)}
            />
          ))}
        </div>

        {missing.length > 0 ? (
          <p className="text-sm text-ink-700" role="status">
            Missing required modalities:{" "}
            {missing.map((item) => item.toUpperCase()).join(", ")}
          </p>
        ) : null}

        {submitError ? (
          <p className="border border-caution-800/30 bg-caution-50 px-4 py-3 text-sm text-caution-800" role="alert">
            {submitError}
          </p>
        ) : null}

        <button
          type="submit"
          disabled={busy}
          className="border border-accent-700 bg-accent-700 px-4 py-2 text-sm font-medium text-white hover:bg-accent-600 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? "Validating case…" : "Submit case"}
        </button>
      </form>

      {result ? <ValidationPanel result={result} /> : null}
    </div>
  );
}

function ModalitySlot({
  modality,
  required,
  slot,
  disabled,
  onFile,
}: {
  modality: Modality;
  required: boolean;
  slot: SlotState;
  disabled: boolean;
  onFile: (file: File | null) => void;
}) {
  const inputId = `modality-${modality}`;
  return (
    <div className="border border-ink-900/10 bg-white p-4">
      <label htmlFor={inputId} className="block font-medium text-ink-950">
        {MODALITY_LABELS[modality]}
        {required ? <span className="text-caution-800"> *</span> : null}
      </label>
      <p className="mt-1 text-sm text-ink-500">Drop .nii / .nii.gz here</p>
      <input
        id={inputId}
        name={modality}
        type="file"
        accept=".nii,.nii.gz,application/gzip"
        disabled={disabled}
        className="mt-3 block w-full text-sm text-ink-700 file:mr-3 file:border file:border-ink-900/20 file:bg-paper-100 file:px-3 file:py-1.5 file:text-sm"
        onChange={(event) => {
          const file = event.target.files?.[0] ?? null;
          onFile(file);
        }}
      />
      {slot.file ? (
        <p className="mt-2 text-sm text-ink-700">
          {slot.file.name} · {(slot.file.size / (1024 * 1024)).toFixed(2)} MB
        </p>
      ) : null}
      {slot.status === "uploading" && slot.progress !== null ? (
        <p className="mt-2 text-sm text-ink-700" aria-live="polite">
          Uploading {MODALITY_LABELS[modality].split(" ")[0]} {slot.progress}%
        </p>
      ) : null}
      {slot.status === "uploaded" ? (
        <p className="mt-2 text-sm text-ink-700">Uploaded</p>
      ) : null}
      {slot.clientError ? (
        <p className="mt-2 text-sm text-caution-800" role="alert">
          {slot.clientError}
        </p>
      ) : null}
    </div>
  );
}

function ValidationPanel({ result }: { result: CaseResponse }) {
  const spatial = result.validation?.spatial;
  return (
    <section className="mt-10 border border-ink-900/10 bg-white p-6" aria-labelledby="validation-heading">
      <h2 id="validation-heading" className="font-display text-2xl text-ink-950">
        Case validation
      </h2>
      <p className="mt-1 text-sm text-ink-500">
        Case {result.case_id} · status {result.status} · inference {result.inference}
      </p>
      <ul className="mt-4 space-y-1 text-sm">
        {REQUIRED_MODALITIES.map((modality) => (
          <li key={modality}>
            {result.validation?.modalities[modality] ? "✓" : "✗"} {modality.toUpperCase()}
          </li>
        ))}
        <li>
          {result.validation?.modalities.seg ? "✓" : "–"} SEG{" "}
          {result.has_ground_truth ? "(present, stored only)" : "(not provided)"}
        </li>
      </ul>
      <h3 className="mt-6 text-sm font-semibold uppercase tracking-[0.12em] text-ink-500">
        Spatial consistency
      </h3>
      <ul className="mt-2 space-y-1 text-sm">
        <li>{mark(spatial?.shape_consistent)} Shape consistent</li>
        <li>{mark(spatial?.affine_consistent)} Affine consistent</li>
        <li>{mark(spatial?.spacing_consistent)} Spacing consistent</li>
      </ul>
      {result.detail ? (
        <p className="mt-4 text-sm text-caution-800" role="alert">
          {result.detail}
        </p>
      ) : null}
      {result.validation?.warnings.map((warning) => (
        <p key={warning} className="mt-2 text-sm text-ink-700">
          Warning: {warning}
        </p>
      ))}
    </section>
  );
}

function mark(value: boolean | null | undefined): string {
  if (value === true) {
    return "✓";
  }
  if (value === false) {
    return "✗";
  }
  return "–";
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Upload failed.";
}
