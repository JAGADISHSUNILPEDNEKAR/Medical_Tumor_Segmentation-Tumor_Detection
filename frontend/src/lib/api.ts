import { parseApiError, type CaseCreateResponse, type CaseResponse, type HealthResponse, type ModelInfoResponse } from "../types/api";

export function apiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_URL;
  if (configured === undefined || configured === "") {
    return "";
  }
  return configured.replace(/\/$/, "");
}

async function readError(response: Response): Promise<never> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  throw parseApiError(response.status, body);
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    await readError(response);
  }
  return (await response.json()) as T;
}

async function sendJson<T>(path: string, init: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...init.headers },
  });
  if (!response.ok) {
    await readError(response);
  }
  return (await response.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/v1/health");
}

export function fetchModelInfo(): Promise<ModelInfoResponse> {
  return getJson<ModelInfoResponse>("/api/v1/model/info");
}

export function createCase(): Promise<CaseCreateResponse> {
  return sendJson<CaseCreateResponse>("/api/v1/cases", { method: "POST" });
}

export function fetchCase(caseId: string): Promise<CaseResponse> {
  return getJson<CaseResponse>(`/api/v1/cases/${caseId}`);
}

export function completeCase(caseId: string): Promise<CaseResponse> {
  return sendJson<CaseResponse>(`/api/v1/cases/${caseId}/complete`, { method: "POST" });
}

export function uploadCaseFile(
  caseId: string,
  modality: string,
  file: File,
  onProgress: (percent: number) => void,
): Promise<CaseResponse> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file, file.name);

    xhr.open("POST", `${apiBaseUrl()}/api/v1/cases/${caseId}/files/${modality}`);
    xhr.responseType = "json";

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as CaseResponse);
        return;
      }
      reject(parseApiError(xhr.status, xhr.response));
    };
    xhr.onerror = () => {
      reject(parseApiError(0, { detail: "Network request failed.", error: "NETWORK_ERROR" }));
    };
    xhr.send(form);
  });
}
