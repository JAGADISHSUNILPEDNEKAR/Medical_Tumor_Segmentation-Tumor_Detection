/**
 * Keyboard slice step-through for the segmentation viewer.
 *
 * `KeyJ` steps back and `KeyL` steps forward through the active plane,
 * mirroring the convention used by standard radiology tools (PRD §3,
 * Accessibility & Responsive Design). `KeyI` / `KeyK` cycle which plane is
 * active so all three are reachable without a pointer.
 *
 * Keys are read from `event.code`, not `event.key`, so the shortcuts keep
 * working on non-QWERTY layouts.
 */

import { useEffect } from "react";

import type { PlaneId, SliceState } from "../types/viewer";

const PLANE_ORDER: readonly PlaneId[] = ["axial", "coronal", "sagittal"] as const;

/** True when the event originated in a control that consumes its own keys. */
function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

export interface SliceKeyboardOptions {
  /** Plane currently receiving keyboard steps. */
  activePlane: PlaneId;
  /** Per-plane slice state, used to clamp at the volume bounds. */
  slices: Record<PlaneId, SliceState>;
  /** Called with the clamped target index for the active plane. */
  onSliceChange: (plane: PlaneId, index: number) => void;
  /** Called when the user cycles to a different plane. */
  onActivePlaneChange: (plane: PlaneId) => void;
  /** Disables the bindings while no volume is loaded. */
  enabled: boolean;
}

export function useSliceKeyboard({
  activePlane,
  slices,
  onSliceChange,
  onActivePlaneChange,
  enabled,
}: SliceKeyboardOptions): void {
  useEffect(() => {
    if (!enabled) return;

    function handleKeyDown(event: KeyboardEvent) {
      // Never steal keys from form controls (including the slice sliders
      // themselves, which already handle arrow keys natively).
      if (isTypingTarget(event.target)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      const step = event.code === "KeyJ" ? -1 : event.code === "KeyL" ? 1 : 0;

      if (step !== 0) {
        const state = slices[activePlane];
        const next = Math.max(0, Math.min(state.index + step, state.total - 1));
        if (next !== state.index) onSliceChange(activePlane, next);
        event.preventDefault();
        return;
      }

      const planeStep = event.code === "KeyI" ? -1 : event.code === "KeyK" ? 1 : 0;
      if (planeStep !== 0) {
        const current = PLANE_ORDER.indexOf(activePlane);
        const nextPlane =
          PLANE_ORDER[(current + planeStep + PLANE_ORDER.length) % PLANE_ORDER.length]!;
        onActivePlaneChange(nextPlane);
        event.preventDefault();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activePlane, slices, onSliceChange, onActivePlaneChange, enabled]);
}
