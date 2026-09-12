import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { UploadPage } from "./UploadPage";

function renderUpload() {
  return render(
    <MemoryRouter>
      <UploadPage />
    </MemoryRouter>,
  );
}

describe("UploadPage", () => {
  it("renders four required modality slots and optional seg", () => {
    renderUpload();
    expect(document.getElementById("modality-t1")).toBeTruthy();
    expect(document.getElementById("modality-t1ce")).toBeTruthy();
    expect(document.getElementById("modality-t2")).toBeTruthy();
    expect(document.getElementById("modality-flair")).toBeTruthy();
    expect(document.getElementById("modality-seg")).toBeTruthy();
  });

  it("shows missing modality error on submit", () => {
    renderUpload();
    expect(screen.getByText(/Missing required modalities/)).toHaveTextContent("T1");
    const form = document.querySelector("form");
    expect(form).toBeTruthy();
    fireEvent.submit(form as HTMLFormElement);
    expect(screen.getByRole("alert")).toHaveTextContent("Required modality 'T1' is missing.");
  });

  it("rejects an invalid extension in the T1 slot", () => {
    renderUpload();
    const input = document.getElementById("modality-t1") as HTMLInputElement;
    const file = new File([new Uint8Array([1, 2, 3])], "scan.png", { type: "image/png" });
    fireEvent.change(input, { target: { files: [file] } });
    expect(screen.getByRole("alert")).toHaveTextContent(".nii");
  });
});
