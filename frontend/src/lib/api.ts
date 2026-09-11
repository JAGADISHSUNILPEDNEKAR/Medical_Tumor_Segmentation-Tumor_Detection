import { ApiError, type HealthResponse, type ModelInfoResponse } from "../types/api";

function apiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_URL;
  if (configured === undefined || configured === "") {
    return "";
  }
  return configured.replace(/\/$/, "");
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl()}${path}`, {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Keep the generic status message if the body is not JSON.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/v1/health");
}

export function fetchModelInfo(): Promise<ModelInfoResponse> {
  return getJson<ModelInfoResponse>("/api/v1/model/info");
}
