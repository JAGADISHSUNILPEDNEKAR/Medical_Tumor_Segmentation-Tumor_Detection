/**
 * 2D slice extraction from 3D NIfTI volumes.
 *
 * Coordinate convention (RAS+ — NIfTI default):
 *   Axis 0 (dim[1]) → Left-Right      → Sagittal slicing axis
 *   Axis 1 (dim[2]) → Posterior-Anterior → Coronal slicing axis
 *   Axis 2 (dim[3]) → Inferior-Superior  → Axial slicing axis
 *
 * Each extraction function slices perpendicular to its named axis:
 *   getAxialSlice(v, k)    → slice at v.data[*, *, k]  → image width=dim0, height=dim1
 *   getCoronalSlice(v, k)  → slice at v.data[*, k, *]  → image width=dim0, height=dim2
 *   getSagittalSlice(v, k) → slice at v.data[k, *, *]  → image width=dim1, height=dim2
 *
 * All three functions delegate to extractSlice() to avoid duplicated logic.
 */

import type { NiftiVolume } from "../types/viewer";
import { type RegionConfig, REGION_CONFIG } from "../types/viewer";
import type { RegionVisibility } from "../types/viewer";

// ─── Core extraction ─────────────────────────────────────────────────────────

/**
 * Extract a 2D slice from a 3D volume along the given axis.
 *
 * @param volume - 3D NIfTI volume
 * @param axis   - 0 (sagittal), 1 (coronal), or 2 (axial)
 * @param index  - Slice index along the axis (clamped to valid range)
 * @returns Raw Float32Array of voxel intensities, width, height
 */
export function extractSlice(
  volume: NiftiVolume,
  axis: 0 | 1 | 2,
  index: number,
): { values: Float32Array; width: number; height: number } {
  const [d0, d1, d2] = volume.shape;
  const data = volume.data;

  let width: number;
  let height: number;
  let maxIndex: number;

  switch (axis) {
    case 2: // Axial: slice along axis 2
      width = d0;
      height = d1;
      maxIndex = d2;
      break;
    case 1: // Coronal: slice along axis 1
      width = d0;
      height = d2;
      maxIndex = d1;
      break;
    case 0: // Sagittal: slice along axis 0
      width = d1;
      height = d2;
      maxIndex = d0;
      break;
  }

  const clampedIndex = Math.max(0, Math.min(index, maxIndex - 1));
  const values = new Float32Array(width * height);

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      let voxelIndex: number;

      switch (axis) {
        case 2: // axial: data[w, h, clampedIndex]
          voxelIndex = w + h * d0 + clampedIndex * d0 * d1;
          break;
        case 1: // coronal: data[w, clampedIndex, h]
          voxelIndex = w + clampedIndex * d0 + h * d0 * d1;
          break;
        case 0: // sagittal: data[clampedIndex, w, h]
          voxelIndex = clampedIndex + w * d0 + h * d0 * d1;
          break;
      }

      values[h * width + w] = data[voxelIndex] ?? 0;
    }
  }

  return { values, width, height };
}

// ─── Named plane extractors ──────────────────────────────────────────────────

export function getAxialSlice(volume: NiftiVolume, index: number) {
  return extractSlice(volume, 2, index);
}

export function getCoronalSlice(volume: NiftiVolume, index: number) {
  return extractSlice(volume, 1, index);
}

export function getSagittalSlice(volume: NiftiVolume, index: number) {
  return extractSlice(volume, 0, index);
}

/**
 * Get total slice count for a named plane.
 */
export function getSliceCount(volume: NiftiVolume, axis: 0 | 1 | 2): number {
  return volume.shape[axis];
}

// ─── Plane axis mapping ──────────────────────────────────────────────────────

import type { PlaneId } from "../types/viewer";

export const PLANE_AXIS: Record<PlaneId, 0 | 1 | 2> = {
  sagittal: 0,
  coronal: 1,
  axial: 2,
};

// ─── Normalization ───────────────────────────────────────────────────────────

/**
 * Normalize a raw float slice to 0-255 grayscale using robust percentile windowing.
 *
 * This is a display-only normalization. The underlying NIfTI data is never modified.
 *
 * @param values - Raw voxel intensities
 * @param pLow   - Lower percentile (default 1%)
 * @param pHigh  - Upper percentile (default 99%)
 * @returns Uint8Array of grayscale pixel values
 */
export function normalizeToGrayscale(
  values: Float32Array,
  pLow = 1,
  pHigh = 99,
): Uint8Array {
  // Find finite values for percentile computation
  const finiteValues: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (Number.isFinite(v)) {
      finiteValues.push(v);
    }
  }

  if (finiteValues.length === 0) {
    return new Uint8Array(values.length);
  }

  finiteValues.sort((a, b) => a - b);

  const lowIdx = Math.floor((pLow / 100) * (finiteValues.length - 1));
  const highIdx = Math.ceil((pHigh / 100) * (finiteValues.length - 1));
  const vMin = finiteValues[lowIdx]!;
  const vMax = finiteValues[highIdx]!;
  const range = vMax - vMin || 1;

  const result = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) {
      result[i] = 0;
      continue;
    }
    const normalized = (v - vMin) / range;
    result[i] = Math.max(0, Math.min(255, Math.round(normalized * 255)));
  }

  return result;
}

// ─── Segmentation overlay ────────────────────────────────────────────────────

/**
 * Render a grayscale MRI slice with segmentation overlay as RGBA ImageData.
 *
 * @param grayPixels  - Grayscale MRI slice (0-255)
 * @param segValues   - Raw segmentation slice (label values)
 * @param width       - Image width
 * @param height      - Image height
 * @param regions     - Region visibility toggles
 * @param opacity     - Overlay opacity (0-1)
 * @param showOverlay - Whether overlay is enabled
 * @returns RGBA pixel data ready for canvas putImageData
 */
export function compositeSlice(
  grayPixels: Uint8Array,
  segValues: Float32Array | null,
  width: number,
  height: number,
  regions: RegionVisibility,
  opacity: number,
  showOverlay: boolean,
): ImageData {
  const imageData = new ImageData(width, height);
  const rgba = imageData.data;
  const len = width * height;

  for (let i = 0; i < len; i++) {
    const gray = grayPixels[i] ?? 0;
    const off = i * 4;

    // Base MRI grayscale
    rgba[off] = gray;
    rgba[off + 1] = gray;
    rgba[off + 2] = gray;
    rgba[off + 3] = 255;

    // Overlay segmentation
    if (showOverlay && segValues !== null) {
      const label = Math.round(segValues[i] ?? 0);
      if (label > 0 && regions[label]) {
        const cfg: RegionConfig | undefined = REGION_CONFIG[label];
        if (cfg) {
          const a = opacity;
          rgba[off] = Math.round(gray * (1 - a) + cfg.rgba[0] * a);
          rgba[off + 1] = Math.round(gray * (1 - a) + cfg.rgba[1] * a);
          rgba[off + 2] = Math.round(gray * (1 - a) + cfg.rgba[2] * a);
        }
      }
    }
  }

  return imageData;
}
