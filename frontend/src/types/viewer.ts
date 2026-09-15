/**
 * Phase 4 — Medical Image Viewer type definitions.
 *
 * Coordinate convention (RAS+):
 *   Axis 0 → Left-Right     (Sagittal plane slices along this axis)
 *   Axis 1 → Posterior-Anterior (Coronal plane slices along this axis)
 *   Axis 2 → Inferior-Superior  (Axial plane slices along this axis)
 *
 * The NIfTI standard stores data in RAS+ by default.
 * We read the affine and spacing from the header but do NOT reorient or resample.
 */

// ─── NIfTI Volume ────────────────────────────────────────────────────────────

/** Parsed NIfTI volume with geometry preserved. */
export interface NiftiVolume {
  /** Raw voxel data as a typed array. */
  data: Float32Array;
  /** 3D shape [dim0, dim1, dim2]. */
  shape: [number, number, number];
  /** 4×4 affine matrix (row-major, 16 floats). */
  affine: Float64Array;
  /** Voxel spacing in mm [sx, sy, sz]. */
  spacing: [number, number, number];
}

/** Subset of volume geometry needed for alignment checks. */
export interface VolumeGeometry {
  shape: [number, number, number];
  spacing: [number, number, number];
}

// ─── Slice ───────────────────────────────────────────────────────────────────

/** Anatomical plane identifier. */
export type PlaneId = "axial" | "coronal" | "sagittal";

/** Extracted 2D slice data ready for canvas rendering. */
export interface SliceData {
  /** Grayscale pixel values (0-255), row-major. */
  pixels: Uint8Array;
  /** Width of the slice image in pixels. */
  width: number;
  /** Height of the slice image in pixels. */
  height: number;
}

/** Per-plane navigation state. */
export interface SliceState {
  index: number;
  total: number;
}

// ─── Segmentation Regions ────────────────────────────────────────────────────

/** BraTS segmentation label values. */
export const BRATS_LABELS = {
  BACKGROUND: 0,
  NCR: 1,
  ED: 2,
  ET: 4,
} as const;

/** Configuration for a single segmentation region. */
export interface RegionConfig {
  /** Short name (NCR, ED, ET). */
  name: string;
  /** Full human-readable name. */
  fullName: string;
  /** Display color (hex). */
  color: string;
  /** RGBA components [0-255]. */
  rgba: [number, number, number, number];
  /** BraTS label value. */
  label: number;
}

/**
 * Centralized region visualization configuration.
 * Keyed by BraTS label value. Colors can be changed here.
 */
export const REGION_CONFIG: Record<number, RegionConfig> = {
  [BRATS_LABELS.NCR]: {
    name: "NCR",
    fullName: "Necrotic / Non-Enhancing Tumor Core",
    color: "#e74c3c",
    rgba: [231, 76, 60, 255],
    label: BRATS_LABELS.NCR,
  },
  [BRATS_LABELS.ED]: {
    name: "ED",
    fullName: "Peritumoral Edema",
    color: "#2ecc71",
    rgba: [46, 204, 113, 255],
    label: BRATS_LABELS.ED,
  },
  [BRATS_LABELS.ET]: {
    name: "ET",
    fullName: "Enhancing Tumor",
    color: "#f1c40f",
    rgba: [241, 196, 15, 255],
    label: BRATS_LABELS.ET,
  },
};

/** Foreground label values (everything except background). */
export const FOREGROUND_LABELS = [BRATS_LABELS.NCR, BRATS_LABELS.ED, BRATS_LABELS.ET] as const;

/** Per-region visibility state. */
export type RegionVisibility = Record<number, boolean>;

// ─── 3D Mesh ─────────────────────────────────────────────────────────────────

/** Triangle mesh data for a single region. */
export interface MeshData {
  /** Flat array of vertex positions [x0,y0,z0, x1,y1,z1, ...]. */
  vertices: Float32Array;
  /** Flat array of face indices [i0,i1,i2, ...]. */
  indices: Uint32Array;
  /** Number of triangles. */
  triangleCount: number;
  /** BraTS label this mesh represents. */
  label: number;
}

/** Maximum triangle count per region mesh. */
export const MAX_TRIANGLES_PER_REGION = 200_000;

// ─── Viewer State ────────────────────────────────────────────────────────────

/** Available MRI modalities. */
export const MODALITIES = ["flair", "t1", "t1ce", "t2"] as const;
export type Modality = (typeof MODALITIES)[number];

/**
 * Display labels for every modality a case can hold, including the optional
 * ground-truth segmentation. Ordered as the PRD presents them.
 */
export const ALL_MODALITY_LABELS: readonly { key: string; label: string }[] = [
  { key: "t1", label: "T1" },
  { key: "t1ce", label: "T1ce" },
  { key: "t2", label: "T2" },
  { key: "flair", label: "FLAIR" },
] as const;

/** Overall viewer loading status. */
export type ViewerStatus = "loading" | "loaded" | "error";

/** Complete viewer state managed by ViewerPage. */
export interface ViewerState {
  status: ViewerStatus;
  errorMessage: string | null;

  /** Currently selected MRI modality. */
  modality: Modality;

  /** Loaded MRI volume for the current modality. */
  mriVolume: NiftiVolume | null;

  /** Loaded segmentation volume. */
  segVolume: NiftiVolume | null;

  /** Per-plane slice indices. */
  slices: Record<PlaneId, SliceState>;

  /** Whether the segmentation overlay is ON. */
  overlayVisible: boolean;

  /** Overlay opacity 0–1. */
  overlayOpacity: number;

  /** Per-region visibility toggles. */
  regionVisibility: RegionVisibility;

  /** Whether the 3D viewer modal is open. */
  show3D: boolean;
}

/** Default region visibility: all regions visible. */
export function defaultRegionVisibility(): RegionVisibility {
  return {
    [BRATS_LABELS.NCR]: true,
    [BRATS_LABELS.ED]: true,
    [BRATS_LABELS.ET]: true,
  };
}
