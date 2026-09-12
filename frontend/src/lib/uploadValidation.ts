export const REQUIRED_MODALITIES = ["t1", "t1ce", "t2", "flair"] as const;
export const OPTIONAL_MODALITIES = ["seg"] as const;

export type RequiredModality = (typeof REQUIRED_MODALITIES)[number];
export type Modality = RequiredModality | "seg";

export const MODALITY_LABELS: Record<Modality, string> = {
  t1: "T1",
  t1ce: "T1ce",
  t2: "T2",
  flair: "FLAIR",
  seg: "seg (optional)",
};

export function hasAllowedNiftiExtension(filename: string): boolean {
  const name = filename.replaceAll("\\", "/").split("/").pop()?.toLowerCase() ?? "";
  return name.endsWith(".nii.gz") || name.endsWith(".nii");
}

export function missingRequiredModalities(files: Partial<Record<Modality, File | null>>): RequiredModality[] {
  return REQUIRED_MODALITIES.filter((modality) => !files[modality]);
}
