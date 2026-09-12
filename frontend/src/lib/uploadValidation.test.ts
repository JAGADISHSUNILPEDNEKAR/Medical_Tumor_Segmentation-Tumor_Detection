import { describe, expect, it } from "vitest";

import { hasAllowedNiftiExtension, missingRequiredModalities } from "./uploadValidation";

describe("uploadValidation", () => {
  it("accepts nii and nii.gz only", () => {
    expect(hasAllowedNiftiExtension("t1.nii")).toBe(true);
    expect(hasAllowedNiftiExtension("t1.nii.gz")).toBe(true);
    expect(hasAllowedNiftiExtension("t1.PNG")).toBe(false);
    expect(hasAllowedNiftiExtension("scan.jpg")).toBe(false);
  });

  it("names missing required modalities", () => {
    expect(missingRequiredModalities({ t1: new File([], "t1.nii.gz") })).toEqual([
      "t1ce",
      "t2",
      "flair",
    ]);
  });
});
