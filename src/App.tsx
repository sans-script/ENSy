import { useState, useCallback, useEffect } from 'react';
import { SidebarLeft } from './components/SidebarLeft';
import { SidebarRight } from './components/SidebarRight';
import { PerfSample, SpaceVisualizer } from './components/SpaceVisualizer';
import { BottomHUD } from './components/BottomHUD';
import { MetricsPanel } from './components/MetricsPanel';
import { Algorithm, ColorLayer, ModelKey, SemanticNode, SemanticSpace } from './types';

export default function App() {
  const [activeNode, setActiveNode] = useState<SemanticNode | null>(null);
  const [cameraCoords, setCameraCoords] = useState({ x: 22, y: 16, z: 32 });
  const [initialCameraState, setInitialCameraState] = useState<{ x: number; y: number; z: number; rx: number; ry: number; rz: number } | undefined>(undefined);
  const [fps, setFps] = useState(0);
  const [projectionType, setProjectionType] = useState<Algorithm>('umap');
  const [modelKey, setModelKey] = useState<ModelKey>('qwen');
  const [colorLayer, setColorLayer] = useState<ColorLayer>('category');
  const [metricK, setMetricK] = useState(10);
  const [probeEnabled, setProbeEnabled] = useState(true);
  const [perf, setPerf] = useState<PerfSample | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [currentFilter, setCurrentFilter] = useState<'all' | 'positive' | 'negative'>('all');
  const [cameraMode, setCameraMode] = useState<'perspective' | 'parallel' | 'twopoint'>('perspective');
  const [fov, setFov] = useState(30);
  const [cameraPreset, setCameraPreset] = useState<'front' | 'top' | 'side' | 'iso' | null>(null);
  const [showXAxis, setShowXAxis] = useState(true);
  const [showYAxis, setShowYAxis] = useState(true);
  const [showZAxis, setShowZAxis] = useState(true);
  const [showGridPlanes, setShowGridPlanes] = useState(true);
  const [showPlaneBackground, setShowPlaneBackground] = useState(true);
  const [mouseSensitivity, setMouseSensitivity] = useState(0.0015); // Default to 1.5
  const [showBottomHud, setShowBottomHud] = useState(true);
  const [isReady, setIsReady] = useState(false);
  const [semanticSpace, setSemanticSpace] = useState<SemanticSpace | null>(null);
  const [status, setStatus] = useState<'STABLE' | 'COMPUTING' | 'LOADING' | 'OFFLINE'>('STABLE');

  useEffect(() => {
    let active = true;
    const fetchProjection = async () => {
      setStatus('LOADING');
      try {
        const res = await fetch(`/api/projection/${modelKey}/${projectionType}`);
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail ?? res.statusText);
        }
        const json = await res.json();
        if (active) {
          setSemanticSpace(json);
          setLoadError(null);
          setStatus('STABLE');
        }
      } catch (err) {
        // No placeholder data: an unavailable backend must not look like a result.
        console.error('Failed to load projection:', err);
        if (active) {
          setStatus('OFFLINE');
          setSemanticSpace(null);
          setLoadError(err instanceof Error ? err.message : String(err));
        }
      }
    };

    fetchProjection();
    return () => {
      active = false;
    };
  }, [projectionType, modelKey]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem('visualizerSettings');
      if (stored) {
        const p = JSON.parse(stored);
        if (p.currentFilter) setCurrentFilter(p.currentFilter);
        if (p.projectionType) setProjectionType(p.projectionType);
        if (p.modelKey) setModelKey(p.modelKey);
        if (p.colorLayer) setColorLayer(p.colorLayer);
        if (p.metricK) setMetricK(p.metricK);
        if (p.probeEnabled !== undefined) setProbeEnabled(p.probeEnabled);
        if (p.cameraMode) setCameraMode(p.cameraMode);
        if (p.fov) setFov(p.fov);
        if (p.showXAxis !== undefined) setShowXAxis(p.showXAxis);
        if (p.showYAxis !== undefined) setShowYAxis(p.showYAxis);
        if (p.showZAxis !== undefined) setShowZAxis(p.showZAxis);
        if (p.showGridPlanes !== undefined) setShowGridPlanes(p.showGridPlanes);
        if (p.showPlaneBackground !== undefined) setShowPlaneBackground(p.showPlaneBackground);
        if (p.mouseSensitivity !== undefined) setMouseSensitivity(p.mouseSensitivity);
        if (p.showBottomHud !== undefined) setShowBottomHud(p.showBottomHud);
        if (p.cameraCoords) {
          // p.cameraCoords has x, y, z, rx, ry, rz
          setInitialCameraState(p.cameraCoords);
        }
      }
    } catch (e) {
      console.error('Failed to load settings from local storage', e);
    }
    setIsReady(true);
  }, []);

  // Set up periodic save
  useEffect(() => {
    if (!isReady) return;
    
    // Save settings
    const saveState = {
      projectionType,
      modelKey,
      colorLayer,
      metricK,
      probeEnabled,
      currentFilter,
      cameraMode,
      fov,
      showXAxis,
      showYAxis,
      showZAxis,
      showGridPlanes,
      showPlaneBackground,
      mouseSensitivity,
      showBottomHud,
      // Will be saved whenever it changes because we don't save periodically, but we'll include it.
      // Wait, let's use a ref or just save when cameraCoords updates.
    };
    
    localStorage.setItem('visualizerSettings', JSON.stringify(saveState));
  }, [
    isReady, currentFilter, cameraMode, fov, showXAxis, showYAxis, showZAxis,
    showGridPlanes, showPlaneBackground, mouseSensitivity, showBottomHud,
    projectionType, modelKey, colorLayer, metricK, probeEnabled
  ]);

  // Automatic interval-based typesetting triggers to ensure newly rendered LaTeX elements are processed
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Toggle HUD with 'H' key if not typing in an input
      if (e.key.toLowerCase() === 'h' && document.activeElement?.tagName !== 'INPUT' && document.activeElement?.tagName !== 'TEXTAREA') {
        setShowBottomHud(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    const runTypeset = () => {
      const mj = (window as any).MathJax;
      if (mj && mj.typesetPromise) {
        mj.typesetPromise().catch((err: any) => {
          // Silent catch for internal race conditions
        });
      }
    };

    // Fast initial scan, then poll slightly slower to keep processing cost near-zero
    const initialTimer = setTimeout(runTypeset, 350);
    const intervalId = setInterval(runTypeset, 1200);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      clearTimeout(initialTimer);
      clearInterval(intervalId);
    };
  }, [activeNode, currentFilter]);

  const handleNodeHovered = useCallback((node: SemanticNode | null) => {
    // Keep reference and re-trigger state only on actual node selection changes
    setActiveNode((prev) => {
      const prevKey = prev?.id ?? prev?.token_id;
      const nextKey = node?.id ?? node?.token_id;
      if (prevKey === nextKey) return prev;
      return node;
    });
  }, []);

  const handleCameraCoordsChange = useCallback((coords: { x: number; y: number; z: number; rx: number; ry: number; rz: number }) => {
    setCameraCoords(coords);
  }, []);

  // Save settings on an interval to avoid performance hits during 60fps renders
  useEffect(() => {
    if (!isReady) return;
    const interval = setInterval(() => {
      const saveState = {
        projectionType,
        modelKey,
        colorLayer,
        metricK,
        probeEnabled,
        currentFilter,
        cameraMode,
        fov,
        showXAxis,
        showYAxis,
        showZAxis,
        showGridPlanes,
        showPlaneBackground,
        mouseSensitivity,
        showBottomHud,
        cameraCoords, // Saving current coords
      };
      localStorage.setItem('visualizerSettings', JSON.stringify(saveState));
    }, 1000); // Save once a second
    
    return () => clearInterval(interval);
  }, [
    isReady, currentFilter, cameraMode, fov, showXAxis, showYAxis, showZAxis,
    showGridPlanes, showPlaneBackground, mouseSensitivity, showBottomHud, cameraCoords,
    projectionType, modelKey, colorLayer, metricK, probeEnabled
  ]);

  const handleFPSUpdate = useCallback((newFps: number) => {
    setFps(newFps);
  }, []);

  const handleCameraPresetApplied = useCallback(() => {
    setCameraPreset(null);
  }, []);

  return (
    <div className="flex w-screen h-screen overflow-hidden bg-[#fcfcfc] font-serif antialiased select-none">
      
      {/* Left Sidebar Layout (Optics and general dashboard properties) */}
      <SidebarLeft
        modelName={semanticSpace?.meta?.model ?? '—'}
        modelKey={modelKey}
        onModelKeyChange={setModelKey}
        projectionType={projectionType}
        onProjectionTypeChange={setProjectionType}
        colorLayer={colorLayer}
        onColorLayerChange={setColorLayer}
        metricK={metricK}
        onMetricKChange={setMetricK}
        probeEnabled={probeEnabled}
        onProbeEnabledChange={setProbeEnabled}
        perf={perf}
        embeddingDim={semanticSpace?.meta?.embedding_dim ?? 0}
        totalTokens={semanticSpace?.meta?.total_tokens ?? 0}
        fps={fps}
        coords={cameraCoords}
        currentFilter={currentFilter}
        onFilterChange={setCurrentFilter}
        cameraMode={cameraMode}
        onCameraModeChange={setCameraMode}
        fov={fov}
        onFovChange={setFov}
        onApplyPreset={setCameraPreset}
        showXAxis={showXAxis}
        onShowXAxisChange={setShowXAxis}
        showYAxis={showYAxis}
        onShowYAxisChange={setShowYAxis}
        showZAxis={showZAxis}
        onShowZAxisChange={setShowZAxis}
        showGridPlanes={showGridPlanes}
        onShowGridPlanesChange={setShowGridPlanes}
        showPlaneBackground={showPlaneBackground}
        onShowPlaneBackgroundChange={setShowPlaneBackground}
        mouseSensitivity={mouseSensitivity}
        onMouseSensitivityChange={setMouseSensitivity}
        showBottomHud={showBottomHud}
        onShowBottomHudChange={setShowBottomHud}
        status={status}
      />

      {/* Main Viewport Column */}
      <div className="flex-1 h-full relative bg-[#fcfcfc] overflow-hidden border-none p-0 flex flex-col">
        {/* Three.js interactive canvas visualizer */}
        {isReady && (
          <SpaceVisualizer
            colorLayer={colorLayer}
            heatK={metricK}
            probeEnabled={probeEnabled}
            probeK={metricK}
            onPerfUpdate={setPerf}
            projectionType={projectionType}
            currentFilter={currentFilter}
            cameraMode={cameraMode}
            fov={fov}
            onNodeHovered={handleNodeHovered}
            onCameraCoordsChange={handleCameraCoordsChange}
            onFPSUpdate={handleFPSUpdate}
            cameraPreset={cameraPreset}
            onCameraPresetApplied={handleCameraPresetApplied}
            showXAxis={showXAxis}
            showYAxis={showYAxis}
            showZAxis={showZAxis}
            showGridPlanes={showGridPlanes}
            showPlaneBackground={showPlaneBackground}
            mouseSensitivity={mouseSensitivity}
            initialCameraState={initialCameraState}
            data={semanticSpace}
          />
        )}

        <MetricsPanel
          meta={semanticSpace?.meta ?? null}
          metricK={metricK}
          colorLayer={colorLayer}
          probeEnabled={probeEnabled}
          error={loadError}
          loading={status === 'LOADING'}
        />

        {/* Floating keys HUD indicators */}
        {showBottomHud && <BottomHUD />}
      </div>

      {/* Right Sidebar Layout (Semantic nearest neighbors calculations inspector) */}
      <SidebarRight
        activeNode={activeNode}
        nodes={semanticSpace?.nodes ?? null}
        metricK={metricK}
        probeEnabled={probeEnabled}
      />
      
    </div>
  );
}
