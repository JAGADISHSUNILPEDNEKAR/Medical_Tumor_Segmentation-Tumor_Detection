import { Suspense, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Loader2, AlertTriangle, Box } from "lucide-react";
import * as THREE from "three";

import type {
  NiftiVolume,
  MeshData,
  RegionVisibility as RegionVisibilityType,
} from "../../types/viewer";
import { REGION_CONFIG, FOREGROUND_LABELS } from "../../types/viewer";
import { generateAllMeshes } from "../../lib/meshGeneration";

interface Tumor3DViewerProps {
  segVolume: NiftiVolume | null;
  regionVisibility: RegionVisibilityType;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}

/** Single region mesh rendered as a Three.js mesh. */
function RegionMesh({ mesh }: { mesh: MeshData }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(mesh.vertices, 3));
    geo.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    geo.computeVertexNormals();
    return geo;
  }, [mesh]);

  const color = REGION_CONFIG[mesh.label]?.color ?? "#888888";

  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.85}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

/** The 3D scene content rendered inside the Canvas. */
function SceneContent({
  meshes,
  regionVisibility,
}: {
  meshes: MeshData[];
  regionVisibility: RegionVisibilityType;
}) {
  const controlsRef = useRef(null);

  // Compute bounding box center for camera target
  const center = useMemo(() => {
    if (meshes.length === 0) return new THREE.Vector3(0, 0, 0);
    const box = new THREE.Box3();
    for (const m of meshes) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(m.vertices, 3));
      geo.computeBoundingBox();
      if (geo.boundingBox) box.union(geo.boundingBox);
      geo.dispose();
    }
    const c = new THREE.Vector3();
    box.getCenter(c);
    return c;
  }, [meshes]);



  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[1, 1, 1]} intensity={0.8} />
      <directionalLight position={[-1, -0.5, -1]} intensity={0.3} />

      {meshes.map((mesh) => {
        const visible = regionVisibility[mesh.label] ?? false;
        if (!visible) return null;
        return <RegionMesh key={mesh.label} mesh={mesh} />;
      })}

      <OrbitControls
        ref={controlsRef}
        target={center}
        enableDamping
        dampingFactor={0.15}
      />
    </>
  );
}

export function Tumor3DViewer({
  segVolume,
  regionVisibility,
  loading,
  error,
  onClose,
}: Tumor3DViewerProps) {
  const meshes = useMemo<MeshData[]>(() => {
    if (!segVolume) return [];
    try {
      return generateAllMeshes(segVolume);
    } catch {
      return [];
    }
  }, [segVolume]);

  const hasMeshes = meshes.length > 0;
  const meshError =
    error ?? (!loading && segVolume && !hasMeshes
      ? "Unable to generate 3D mesh. The segmentation may be empty."
      : null);

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 bg-slate-50 px-4 py-3">
        <div className="flex items-center gap-2">
          <Box className="h-4 w-4 text-accent-700" />
          <h2 className="text-sm font-semibold text-ink-950">
            3D Segmentation Mesh
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onClose}
            className="rounded px-2 py-1 text-xs font-medium text-ink-500 transition-colors hover:bg-ink-950/5"
            aria-label="Close 3D viewer"
          >
            Close
          </button>
        </div>
      </div>

      {/* Region legend */}
      <div className="flex items-center gap-4 border-b border-slate-100 bg-white px-4 py-2">
        {FOREGROUND_LABELS.map((label) => {
          const cfg = REGION_CONFIG[label];
          if (!cfg) return null;
          const visible = regionVisibility[label] ?? false;
          return (
            <div
              key={label}
              className={`flex items-center gap-1.5 text-xs ${visible ? "opacity-100" : "opacity-40"}`}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: cfg.color }}
              />
              <span className="font-medium text-ink-700">{cfg.name}</span>
            </div>
          );
        })}
        <span className="ml-auto text-xs text-ink-500">
          Drag to rotate · Scroll to zoom · Right-click to pan
        </span>
      </div>

      {/* 3D Canvas */}
      <div className="relative" style={{ height: 400 }}>
        {loading && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-100">
            <Loader2 className="h-6 w-6 animate-spin text-accent-600" />
            <p className="mt-2 text-xs text-ink-500">Preparing 3D mesh…</p>
          </div>
        )}

        {meshError && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-slate-100 p-4 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-500" />
            <p className="mt-2 text-sm font-medium text-ink-700">
              {meshError}
            </p>
            <p className="mt-1 text-xs text-ink-500">
              The 2D viewer remains available.
            </p>
          </div>
        )}

        {hasMeshes && !loading && (
          <Canvas
            camera={{ position: [0, 0, 300], fov: 50, near: 0.1, far: 10000 }}
            style={{ background: "#f8fafc" }}
          >
            <Suspense fallback={null}>
              <SceneContent
                meshes={meshes}
                regionVisibility={regionVisibility}
              />
            </Suspense>
          </Canvas>
        )}
      </div>
    </div>
  );
}
