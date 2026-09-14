/**
 * NIfTI volume loader for browser-side parsing.
 *
 * Uses nifti-reader-js for header/data parsing and pako for gzip decompression.
 * Preserves the full affine, shape, and spacing from the NIfTI header.
 * Does NOT resample or modify the stored voxel data.
 */

import * as nifti from "nifti-reader-js";
import * as pako from "pako";

import { apiBaseUrl } from "./api";
import type { NiftiVolume } from "../types/viewer";

/**
 * Fetch and parse a NIfTI artifact from the backend.
 *
 * @param caseId - UUID of the case
 * @param artifact - Allowlisted artifact name (t1, t1ce, t2, flair, segmentation)
 * @returns Parsed NIfTI volume with shape, affine, spacing, and data
 */
export async function loadNiftiFromUrl(
  caseId: string,
  artifact: string,
): Promise<NiftiVolume> {
  const url = `${apiBaseUrl()}/api/v1/cases/${encodeURIComponent(caseId)}/artifacts/${encodeURIComponent(artifact)}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to load artifact '${artifact}': ${response.status} ${response.statusText}`,
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return parseNiftiBuffer(arrayBuffer);
}

/**
 * Parse a raw NIfTI buffer (`.nii` or `.nii.gz`) into a NiftiVolume.
 *
 * Handles gzip decompression transparently.
 * Extracts: shape (3D), affine (4×4), voxel spacing, and voxel data as Float32Array.
 */
export function parseNiftiBuffer(buffer: ArrayBuffer): NiftiVolume {
  let data = buffer;

  // Decompress if gzipped
  if (nifti.isCompressed(data)) {
    data = pako.inflate(new Uint8Array(data)).buffer;
  }

  if (!nifti.isNIFTI(data)) {
    throw new Error("Buffer is not a valid NIfTI file.");
  }

  const header = nifti.readHeader(data);
  if (header === null) {
    throw new Error("Failed to parse NIfTI header.");
  }

  const imageData = nifti.readImage(header, data);
  if (imageData === null) {
    throw new Error("Failed to read NIfTI image data.");
  }

  // Extract 3D shape (squeeze trailing dimensions)
  const dims = header.dims;
  const ndim = dims[0] ?? 3;
  const dim1 = dims[1] ?? 1;
  const dim2 = dims[2] ?? 1;
  const dim3 = dims[3] ?? 1;

  if (ndim < 3) {
    throw new Error(`Expected a 3D volume but got ${ndim}D.`);
  }

  const shape: [number, number, number] = [dim1, dim2, dim3];

  // Extract voxel spacing
  const pixDims = header.pixDims;
  const spacing: [number, number, number] = [
    Math.abs(pixDims[1] ?? 1),
    Math.abs(pixDims[2] ?? 1),
    Math.abs(pixDims[3] ?? 1),
  ];

  // Extract 4×4 affine matrix
  const affine = extractAffine(header);

  // Convert voxel data to Float32Array regardless of original datatype
  const voxelData = convertToFloat32(header, imageData);

  return { data: voxelData, shape, affine, spacing };
}

/**
 * Extract the 4×4 affine matrix from the NIfTI header.
 *
 * Uses the sform (qform_code > 0 → affine stored directly) when available.
 * Falls back to constructing from pixDims if neither sform nor qform is set.
 */
function extractAffine(header: nifti.NIFTI1 | nifti.NIFTI2): Float64Array {
  const m = new Float64Array(16);

  // nifti-reader-js stores the affine in header.affine as a 4x4 array
  if (header.affine && Array.isArray(header.affine)) {
    for (let r = 0; r < 4; r++) {
      const row = header.affine[r];
      if (row) {
        for (let c = 0; c < 4; c++) {
          m[r * 4 + c] = row[c] ?? 0;
        }
      }
    }
    return m;
  }

  // Fallback: identity scaled by pixDims
  m[0] = Math.abs(header.pixDims[1] ?? 1);
  m[5] = Math.abs(header.pixDims[2] ?? 1);
  m[10] = Math.abs(header.pixDims[3] ?? 1);
  m[15] = 1;
  return m;
}

/**
 * Convert NIfTI image data to Float32Array based on the datatype code.
 */
function convertToFloat32(
  header: nifti.NIFTI1 | nifti.NIFTI2,
  imageData: ArrayBuffer,
): Float32Array {
  const dt = header.datatypeCode;
  const numVoxels = (header.dims[1] ?? 1) * (header.dims[2] ?? 1) * (header.dims[3] ?? 1);

  // Map NIfTI datatype codes to TypedArrays
  // DT_UINT8=2, DT_INT16=4, DT_INT32=8, DT_FLOAT32=16, DT_FLOAT64=64, DT_UINT16=512
  let typedArray: ArrayLike<number>;

  switch (dt) {
    case 2: // UINT8
      typedArray = new Uint8Array(imageData, 0, numVoxels);
      break;
    case 4: // INT16
      typedArray = new Int16Array(imageData, 0, numVoxels);
      break;
    case 8: // INT32
      typedArray = new Int32Array(imageData, 0, numVoxels);
      break;
    case 16: // FLOAT32
      return new Float32Array(imageData, 0, numVoxels);
    case 64: // FLOAT64
      typedArray = new Float64Array(imageData, 0, numVoxels);
      break;
    case 512: // UINT16
      typedArray = new Uint16Array(imageData, 0, numVoxels);
      break;
    default:
      // Attempt to interpret as float32
      typedArray = new Float32Array(imageData, 0, numVoxels);
      break;
  }

  const result = new Float32Array(numVoxels);
  for (let i = 0; i < numVoxels; i++) {
    result[i] = typedArray[i] ?? 0;
  }
  return result;
}
