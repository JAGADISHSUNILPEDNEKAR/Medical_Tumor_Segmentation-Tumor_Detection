/**
 * Slice extraction tests.
 *
 * Uses a synthetic 3D volume where voxel[x][y][z] = x*100 + y*10 + z
 * to verify that each anatomical plane extracts the correct orientation.
 *
 * Coordinate convention (RAS+ — NIfTI default):
 *   Axis 0 (dim0) → Left-Right      → Sagittal slicing axis
 *   Axis 1 (dim1) → Posterior-Anterior → Coronal slicing axis
 *   Axis 2 (dim2) → Inferior-Superior  → Axial slicing axis
 */

import { describe, expect, it } from "vitest";

import type { NiftiVolume } from "../types/viewer";
import {
  extractSlice,
  getAxialSlice,
  getCoronalSlice,
  getSagittalSlice,
  normalizeToGrayscale,
  compositeSlice,
} from "./sliceExtraction";

/** Create a test volume where data[x + y*nx + z*nx*ny] = x*100 + y*10 + z */
function makeTestVolume(
  nx: number,
  ny: number,
  nz: number,
): NiftiVolume {
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        data[x + y * nx + z * nx * ny] = x * 100 + y * 10 + z;
      }
    }
  }
  return {
    data,
    shape: [nx, ny, nz],
    affine: new Float64Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    spacing: [1, 1, 1],
  };
}

describe("extractSlice", () => {
  const vol = makeTestVolume(4, 5, 6);

  it("extracts axial slice (axis 2) with correct dimensions", () => {
    const result = extractSlice(vol, 2, 3);
    expect(result.width).toBe(4); // dim0
    expect(result.height).toBe(5); // dim1
    // Check specific voxel: data[x=2, y=1, z=3] = 2*100 + 1*10 + 3 = 213
    expect(result.values[1 * 4 + 2]).toBe(213);
  });

  it("extracts coronal slice (axis 1) with correct dimensions", () => {
    const result = extractSlice(vol, 1, 2);
    expect(result.width).toBe(4); // dim0
    expect(result.height).toBe(6); // dim2
    // Check specific voxel: data[x=1, y=2, z=4] = 1*100 + 2*10 + 4 = 124
    // In output: height=z=4, width=x=1 → index = 4*4 + 1 = 17
    expect(result.values[4 * 4 + 1]).toBe(124);
  });

  it("extracts sagittal slice (axis 0) with correct dimensions", () => {
    const result = extractSlice(vol, 0, 3);
    expect(result.width).toBe(5); // dim1
    expect(result.height).toBe(6); // dim2
    // Check specific voxel: data[x=3, y=2, z=5] = 3*100 + 2*10 + 5 = 325
    // In output: height=z=5, width=y=2 → index = 5*5 + 2 = 27
    expect(result.values[5 * 5 + 2]).toBe(325);
  });

  it("clamps slice index to valid range", () => {
    const resultLow = extractSlice(vol, 2, -5);
    const resultHigh = extractSlice(vol, 2, 999);

    // Should not throw
    expect(resultLow.values.length).toBe(4 * 5);
    expect(resultHigh.values.length).toBe(4 * 5);
  });
});

describe("named plane extractors", () => {
  const vol = makeTestVolume(3, 4, 5);

  it("getAxialSlice returns axis-2 slice", () => {
    const s = getAxialSlice(vol, 2);
    expect(s.width).toBe(3);
    expect(s.height).toBe(4);
  });

  it("getCoronalSlice returns axis-1 slice", () => {
    const s = getCoronalSlice(vol, 1);
    expect(s.width).toBe(3);
    expect(s.height).toBe(5);
  });

  it("getSagittalSlice returns axis-0 slice", () => {
    const s = getSagittalSlice(vol, 2);
    expect(s.width).toBe(4);
    expect(s.height).toBe(5);
  });
});

describe("normalizeToGrayscale", () => {
  it("normalizes values to 0-255 range", () => {
    const values = new Float32Array([0, 50, 100, 150, 200]);
    const result = normalizeToGrayscale(values, 0, 100);
    expect(result[0]).toBe(0);
    expect(result[4]).toBe(255);
    // Middle value should be around 128
    expect(result[2]).toBeGreaterThan(100);
    expect(result[2]).toBeLessThan(160);
  });

  it("handles all-zero input", () => {
    const values = new Float32Array([0, 0, 0, 0]);
    const result = normalizeToGrayscale(values, 1, 99);
    // Should not crash, all zeros → all zeros
    expect(result.length).toBe(4);
  });

  it("handles NaN/Infinity in input", () => {
    const values = new Float32Array([NaN, 0, 50, Infinity, 100]);
    const result = normalizeToGrayscale(values);
    expect(result[0]).toBe(0); // NaN → 0
    expect(result[3]).toBe(0); // Infinity → 0
    expect(result.length).toBe(5);
  });

  it("handles single-value input", () => {
    const values = new Float32Array([42]);
    const result = normalizeToGrayscale(values);
    // Single value: range = 0, should normalize to some value without crash
    expect(result.length).toBe(1);
  });
});

describe("compositeSlice", () => {
  it("produces RGBA ImageData of correct dimensions", () => {
    const gray = new Uint8Array([128, 200, 50, 100]);
    const result = compositeSlice(gray, null, 2, 2, { 1: true, 2: true, 4: true }, 0.5, false);
    expect(result.width).toBe(2);
    expect(result.height).toBe(2);
    expect(result.data.length).toBe(2 * 2 * 4); // RGBA
  });

  it("renders grayscale when overlay is off", () => {
    const gray = new Uint8Array([128]);
    const seg = new Float32Array([2]); // ED label
    const result = compositeSlice(gray, seg, 1, 1, { 1: true, 2: true, 4: true }, 0.5, false);
    // Overlay off → pure grayscale
    expect(result.data[0]).toBe(128); // R
    expect(result.data[1]).toBe(128); // G
    expect(result.data[2]).toBe(128); // B
    expect(result.data[3]).toBe(255); // A
  });

  it("blends segmentation color when overlay is on", () => {
    const gray = new Uint8Array([128]);
    const seg = new Float32Array([2]); // ED label (green)
    const result = compositeSlice(gray, seg, 1, 1, { 1: true, 2: true, 4: true }, 0.5, true);
    // Should be blended: half gray, half green
    // ED green = [46, 204, 113]
    // Expected R ≈ (128*0.5 + 46*0.5) = 87
    expect(result.data[0]).toBeLessThan(128); // More towards the color
    expect(result.data[1]).toBeGreaterThan(128); // Green contribution
  });

  it("respects region visibility", () => {
    const gray = new Uint8Array([128]);
    const seg = new Float32Array([2]); // ED
    // ED disabled
    const result = compositeSlice(gray, seg, 1, 1, { 1: true, 2: false, 4: true }, 0.5, true);
    // Should be pure grayscale because ED is hidden
    expect(result.data[0]).toBe(128);
    expect(result.data[1]).toBe(128);
    expect(result.data[2]).toBe(128);
  });

  it("handles background label (0) without overlay", () => {
    const gray = new Uint8Array([200]);
    const seg = new Float32Array([0]); // Background
    const result = compositeSlice(gray, seg, 1, 1, { 1: true, 2: true, 4: true }, 0.5, true);
    // Background → no overlay
    expect(result.data[0]).toBe(200);
    expect(result.data[1]).toBe(200);
    expect(result.data[2]).toBe(200);
  });
});

describe("orientation consistency", () => {
  it("axial, coronal, sagittal slices at same point yield same voxel", () => {
    // Volume where value = x*100 + y*10 + z
    const vol = makeTestVolume(8, 10, 12);
    const x = 3, y = 5, z = 7;

    // Axial at z=7: pixels[y*width + x] = pixels[5*8 + 3]
    const axial = extractSlice(vol, 2, z);
    expect(axial.values[y * axial.width + x]).toBe(x * 100 + y * 10 + z);

    // Coronal at y=5: pixels[z*width + x] = pixels[7*8 + 3]
    const coronal = extractSlice(vol, 1, y);
    expect(coronal.values[z * coronal.width + x]).toBe(x * 100 + y * 10 + z);

    // Sagittal at x=3: pixels[z*width + y] = pixels[7*10 + 5]
    const sagittal = extractSlice(vol, 0, x);
    expect(sagittal.values[z * sagittal.width + y]).toBe(x * 100 + y * 10 + z);
  });
});
