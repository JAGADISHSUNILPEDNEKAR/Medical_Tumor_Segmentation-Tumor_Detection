/**
 * Mesh generation tests.
 *
 * Tests the marching cubes implementation with synthetic volumes:
 * - Empty mask → no mesh
 * - Single voxel → small mesh
 * - Filled cube → mesh with correct geometry
 * - Multiple labels → separate meshes
 * - Triangle budget enforcement
 * - Voxel spacing applied to vertices
 */

import { describe, expect, it } from "vitest";

import type { NiftiVolume } from "../types/viewer";
import { generateMeshForLabel, generateAllMeshes, decimateMesh } from "./meshGeneration";

function makeSyntheticSeg(
  nx: number,
  ny: number,
  nz: number,
  fill: (x: number, y: number, z: number) => number,
  spacing: [number, number, number] = [1, 1, 1],
): NiftiVolume {
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        data[x + y * nx + z * nx * ny] = fill(x, y, z);
      }
    }
  }
  return {
    data,
    shape: [nx, ny, nz],
    affine: new Float64Array([
      spacing[0], 0, 0, 0,
      0, spacing[1], 0, 0,
      0, 0, spacing[2], 0,
      0, 0, 0, 1,
    ]),
    spacing,
  };
}

describe("generateMeshForLabel", () => {
  it("returns null for empty mask", () => {
    const vol = makeSyntheticSeg(8, 8, 8, () => 0);
    const mesh = generateMeshForLabel(vol, 1);
    expect(mesh).toBeNull();
  });

  it("returns null for label not present in volume", () => {
    const vol = makeSyntheticSeg(8, 8, 8, (x, y, z) =>
      x >= 2 && x <= 5 && y >= 2 && y <= 5 && z >= 2 && z <= 5 ? 2 : 0,
    );
    const mesh = generateMeshForLabel(vol, 1); // Label 1 not present
    expect(mesh).toBeNull();
  });

  it("generates mesh for a filled cube", () => {
    // Create a 3x3x3 cube of label 1 inside a 10x10x10 volume
    const vol = makeSyntheticSeg(10, 10, 10, (x, y, z) =>
      x >= 3 && x <= 5 && y >= 3 && y <= 5 && z >= 3 && z <= 5 ? 1 : 0,
    );
    const mesh = generateMeshForLabel(vol, 1);
    expect(mesh).not.toBeNull();
    expect(mesh!.triangleCount).toBeGreaterThan(0);
    expect(mesh!.vertices.length).toBeGreaterThan(0);
    expect(mesh!.indices.length).toBeGreaterThan(0);
    expect(mesh!.label).toBe(1);
  });

  it("generates mesh for a single voxel", () => {
    const vol = makeSyntheticSeg(5, 5, 5, (x, y, z) =>
      x === 2 && y === 2 && z === 2 ? 4 : 0,
    );
    const mesh = generateMeshForLabel(vol, 4);
    expect(mesh).not.toBeNull();
    expect(mesh!.triangleCount).toBeGreaterThan(0);
    // A single voxel should produce at most 12 triangles (cube = 6 faces × 2 triangles)
    expect(mesh!.triangleCount).toBeLessThanOrEqual(12);
  });

  it("applies voxel spacing to vertex positions", () => {
    const spacing: [number, number, number] = [2, 3, 4];
    const vol = makeSyntheticSeg(
      6, 6, 6,
      (x, y, z) => (x >= 2 && x <= 3 && y >= 2 && y <= 3 && z >= 2 && z <= 3 ? 1 : 0),
      spacing,
    );
    const mesh = generateMeshForLabel(vol, 1);
    expect(mesh).not.toBeNull();

    // Verify that vertex coordinates are scaled by spacing
    // All x coords should be multiples of spacing[0]=2
    // Find max x coordinate: should be around 4*2 = 8
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < mesh!.vertices.length; i += 3) {
      maxX = Math.max(maxX, mesh!.vertices[i]!);
      maxY = Math.max(maxY, mesh!.vertices[i + 1]!);
      maxZ = Math.max(maxZ, mesh!.vertices[i + 2]!);
    }
    // Max voxel coord is 4 (the edge of the cube at x=3+1), scaled by spacing
    expect(maxX).toBeGreaterThan(2); // At least 1 spacing unit
    expect(maxY).toBeGreaterThan(3);
    expect(maxZ).toBeGreaterThan(4);
  });
});

describe("generateAllMeshes", () => {
  it("generates separate meshes for each present label", () => {
    // Volume with labels 1, 2, and 4 in different regions
    const vol = makeSyntheticSeg(12, 12, 12, (x, y, z) => {
      if (x >= 1 && x <= 3 && y >= 1 && y <= 3 && z >= 1 && z <= 3) return 1;
      if (x >= 5 && x <= 7 && y >= 5 && y <= 7 && z >= 5 && z <= 7) return 2;
      if (x >= 8 && x <= 10 && y >= 8 && y <= 10 && z >= 8 && z <= 10) return 4;
      return 0;
    });

    const meshes = generateAllMeshes(vol);
    expect(meshes.length).toBe(3);

    const labels = meshes.map((m) => m.label).sort();
    expect(labels).toEqual([1, 2, 4]);
  });

  it("skips labels with no voxels", () => {
    // Only label 2 present
    const vol = makeSyntheticSeg(8, 8, 8, (x, y, z) =>
      x >= 2 && x <= 5 && y >= 2 && y <= 5 && z >= 2 && z <= 5 ? 2 : 0,
    );
    const meshes = generateAllMeshes(vol);
    expect(meshes.length).toBe(1);
    expect(meshes[0]!.label).toBe(2);
  });

  it("returns empty array for all-background volume", () => {
    const vol = makeSyntheticSeg(8, 8, 8, () => 0);
    const meshes = generateAllMeshes(vol);
    expect(meshes.length).toBe(0);
  });
});

describe("decimateMesh", () => {
  it("passes through meshes under the triangle budget", () => {
    const vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0];
    const indices = [0, 1, 2];
    const result = decimateMesh(vertices, indices, 100);
    expect(result.indices.length).toBe(3);
    expect(result.vertices.length).toBe(9);
  });

  it("reduces triangle count when over budget", () => {
    // Create a mesh with 100 triangles
    const vertices: number[] = [];
    const indices: number[] = [];
    for (let i = 0; i < 100; i++) {
      const base = vertices.length / 3;
      vertices.push(i, 0, 0, i + 1, 0, 0, i, 1, 0);
      indices.push(base, base + 1, base + 2);
    }

    const result = decimateMesh(vertices, indices, 20);
    const resultTriangles = result.indices.length / 3;
    expect(resultTriangles).toBeLessThanOrEqual(20);
    expect(resultTriangles).toBeGreaterThan(0);
  });
});
