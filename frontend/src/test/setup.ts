import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

import "@testing-library/jest-dom/vitest";

// Polyfill ImageData for jsdom (canvas APIs not available)
if (typeof globalThis.ImageData === "undefined") {
  (globalThis as Record<string, unknown>).ImageData = class ImageData {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
    constructor(width: number, height: number);
    constructor(data: Uint8ClampedArray, width: number, height?: number);
    constructor(widthOrData: number | Uint8ClampedArray, widthOrHeight: number, height?: number) {
      if (widthOrData instanceof Uint8ClampedArray) {
        this.data = widthOrData;
        this.width = widthOrHeight;
        this.height = height ?? (widthOrData.length / (widthOrHeight * 4));
      } else {
        this.width = widthOrData;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}

/**
 * jsdom has no 2D canvas backend, so PlaneView's `getContext("2d")` throws a
 * "Not implemented" error on every render and buries real failures in noise.
 * Stub the handful of methods the viewer actually calls. Pixel output is
 * verified directly against sliceExtraction's compositor, not through canvas.
 */
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() =>
    ({
      putImageData: () => {},
      clearRect: () => {},
      drawImage: () => {},
    })) as unknown as HTMLCanvasElement["getContext"];
}

afterEach(() => {
  cleanup();
});

