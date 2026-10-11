import { useState, useCallback, useRef } from "react";

interface Point {
  x: number;
  y: number;
}

export function useCanvasZoomPan(initialScale = 1) {
  const [scale, setScale] = useState(initialScale);
  const [offset, setOffset] = useState<Point>({ x: 0, y: 0 });
  const isDragging = useRef(false);
  const lastMousePos = useRef<Point>({ x: 0, y: 0 });

  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    // Only intercept wheel events if Shift is held or if we're already zoomed
    // This allows normal page scrolling if the user just scrolls over the canvas
    if (!e.shiftKey && scale === 1 && Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      return;
    }
    
    e.preventDefault();
    e.stopPropagation();
    
    // Determine zoom direction and factor
    const zoomSensitivity = 0.005;
    const zoomDelta = -e.deltaY * zoomSensitivity;
    
    setScale((prevScale) => {
      const newScale = Math.min(Math.max(0.5, prevScale * Math.exp(zoomDelta)), 10);
      
      // We want to zoom in on the mouse pointer.
      // 1. Find mouse position relative to the container
      const rect = e.currentTarget.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      
      // 2. Adjust offset so the point under the mouse stays in the same place
      setOffset((prevOffset) => {
        const scaleRatio = newScale / prevScale;
        return {
          x: mouseX - (mouseX - prevOffset.x) * scaleRatio,
          y: mouseY - (mouseY - prevOffset.y) * scaleRatio,
        };
      });
      
      return newScale;
    });
  }, [scale]);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    // Drag on middle click (button 1) or Shift+LeftClick (button 0 + shift)
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      e.preventDefault();
      isDragging.current = true;
      lastMousePos.current = { x: e.clientX, y: e.clientY };
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  }, []);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging.current) return;
    
    const dx = e.clientX - lastMousePos.current.x;
    const dy = e.clientY - lastMousePos.current.y;
    lastMousePos.current = { x: e.clientX, y: e.clientY };
    
    setOffset((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
  }, []);

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (isDragging.current) {
      isDragging.current = false;
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);
  
  const reset = useCallback(() => {
    setScale(initialScale);
    setOffset({ x: 0, y: 0 });
  }, [initialScale]);

  // Convert a client click back to the unscaled, unpanned fraction coordinates
  const getFractionsFromClick = useCallback((e: React.MouseEvent<HTMLDivElement>): { xFrac: number, yFrac: number } => {
     const rect = e.currentTarget.getBoundingClientRect(); // Get the wrapper rect
     const clickX = e.clientX - rect.left;
     const clickY = e.clientY - rect.top;
     
     // Inverse transform: remove offset, then divide by scale
     const unscaledX = (clickX - offset.x) / scale;
     const unscaledY = (clickY - offset.y) / scale;
     
     const xFrac = unscaledX / rect.width;
     const yFrac = unscaledY / rect.height;
     
     return { xFrac, yFrac };
  }, [offset, scale]);

  const transformStyle = {
    transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
    transformOrigin: "0 0",
    willChange: "transform",
  };

  return {
    scale,
    offset,
    handleWheel,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    reset,
    transformStyle,
    getFractionsFromClick
  };
}
