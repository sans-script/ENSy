import { useState, useCallback, useEffect, useRef } from 'react';
import { SidebarLeft } from './components/SidebarLeft';
import { SidebarRight } from './components/SidebarRight';
import { PerfSample, SpaceVisualizer } from './components/SpaceVisualizer';
import { BottomHUD } from './components/BottomHUD';
import { MetricsPanel } from './components/MetricsPanel';
import { PresentationRemote } from './components/PresentationRemote';
import { Algorithm, ColorLayer, ExternalDataset, ModelKey, SemanticNode, SemanticSpace } from './types';
import { ImportDialog } from './components/ImportDialog';

// Automation/deep-link URL parameters (?model=&algorithm=&k=&focus=&animate=),
// read once at module scope so every lazy initializer below sees the same
// parsed values without re-reading location.search per field.
const URL_PARAMS = (() => {
  try {
    return new URLSearchParams(window.location.search);
  } catch {
    return new URLSearchParams();
  }
})();
const urlModel = URL_PARAMS.get('model');
const urlAlgorithm = URL_PARAMS.get('algorithm');
const urlK = URL_PARAMS.get('k');
const urlFocusRaw = URL_PARAMS.get('focus');
const urlAnimate = URL_PARAMS.get('animate');
const urlZoom = URL_PARAMS.get('zoom');

const initialModelKey = (): ModelKey =>
  urlModel && /^(qwen|gemma2|gpt2|bert|ext-.+)$/.test(urlModel) ? (urlModel as ModelKey) : 'qwen';
const initialProjectionType = (): Algorithm =>
  urlAlgorithm && /^(umap|tsne|tsne_p30|pca|isomap|pca_nc|bm\d+)$/.test(urlAlgorithm)
    ? (urlAlgorithm as Algorithm)
    : 'umap';
const initialMetricK = (): number => {
  const kNum = urlK ? parseInt(urlK, 10) : NaN;
  return Number.isFinite(kNum) && kNum > 0 ? kNum : 10;
};
/** Parsed once: the token to auto-frame once its dataset has loaded, if any. */
const urlFocusToken: string | number | null = urlFocusRaw
  ? (/^-?\d+$/.test(urlFocusRaw) ? parseInt(urlFocusRaw, 10) : urlFocusRaw)
  : null;
const urlFocusAnimate = urlAnimate !== 'false';
/** Multiplier on the auto-framed camera distance: <1 moves closer (bigger,
 *  more legible labels), >1 moves farther out. Undefined/invalid falls back
 *  to the default framing inside __focusCamera. */
const urlFocusZoom: number | undefined = (() => {
  const z = urlZoom ? parseFloat(urlZoom) : NaN;
  return Number.isFinite(z) && z > 0 ? z : undefined;
})();

// Exact camera pose (?camX=&camY=&camZ=&camRX=&camRY=&camRZ=), as an alternative to
// ?focus=: a shared link can either point at a token (re-framed via __focusCamera,
// approximate) or carry the precise position+rotation the sender actually had on
// screen (reproduced exactly, including free-fly views centred on no token at all).
// All six must be present and finite or none of them apply -- a partial pose isn't
// a meaningful camera state.
const urlCameraState: { x: number; y: number; z: number; rx: number; ry: number; rz: number } | undefined = (() => {
  const keys = ['camX', 'camY', 'camZ', 'camRX', 'camRY', 'camRZ'];
  const vals = keys.map((k) => {
    const raw = URL_PARAMS.get(k);
    return raw !== null ? parseFloat(raw) : NaN;
  });
  if (vals.some((v) => !Number.isFinite(v))) return undefined;
  const [x, y, z, rx, ry, rz] = vals;
  return { x, y, z, rx, ry, rz };
})();

export default function App() {
  const [activeNode, setActiveNode] = useState<SemanticNode | null>(null);
  const [cameraCoords, setCameraCoords] = useState({ x: 22, y: 16, z: 32 });
  const [initialCameraState, setInitialCameraState] = useState<{ x: number; y: number; z: number; rx: number; ry: number; rz: number } | undefined>(urlCameraState);
  const [fps, setFps] = useState(0);
  const [projectionType, setProjectionType] = useState<Algorithm>(initialProjectionType);
  const [modelKey, setModelKey] = useState<ModelKey>(initialModelKey);
  const [colorLayer, setColorLayer] = useState<ColorLayer>('category');
  const [metricK, setMetricK] = useState(initialMetricK);
  // Gamepad presentation walkthrough (PresentationRemote): while true, the
  // left/right analog sticks stop driving free flight in SpaceVisualizer so
  // the shoulder-button step navigation has the camera to itself.
  const [presentationMode, setPresentationMode] = useState(false);
  const [probeEnabled, setProbeEnabled] = useState(false);
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
  // datasets imported from the TF Embedding Projector (kept by the server until it restarts)
  const [externals, setExternals] = useState<ExternalDataset[]>([]);
  const [importOpen, setImportOpen] = useState(false);

  useEffect(() => {
    fetch('/api/external')
      .then((r) => (r.ok ? r.json() : []))
      .then((list: ExternalDataset[]) => {
        setExternals(list);
        // a restored session may point at an import the server no longer has
        setModelKey((k) => (k.startsWith('ext-') && !list.some((e) => e.key === k) ? 'qwen' : k));
        setProjectionType((a) => (a.startsWith('bm') ? 'umap' : a));
      })
      .catch(() => setExternals([]));
  }, []);

  const changeModel = useCallback((key: ModelKey) => {
    setModelKey(key);
    // bookmarks belong to one imported dataset
    setProjectionType((a) => (a.startsWith('bm') ? 'umap' : a));
  }, []);

  const handleImported = useCallback((ds: ExternalDataset) => {
    setExternals((list) => [...list.filter((e) => e.key !== ds.key), ds]);
    setModelKey(ds.key);
    setProjectionType(ds.bookmarks.length > 0 ? ds.bookmarks[0].id : 'umap');
    setImportOpen(false);
  }, []);

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

  // Automation/deep-link camera focus (?focus=&k=&animate=&zoom=): once a dataset
  // containing the requested token has loaded, aim the camera at its local
  // K-neighbourhood via SpaceVisualizer's __focusCamera (local-PCA auto-framing).
  // This reaches views that pointer-lock 6-DOF flight cannot script, so a test
  // run can drive the camera purely from the URL.
  useEffect(() => {
    // An exact ?cam*= pose in the URL takes priority over ?focus=: it's already
    // going to be applied once at scene setup via initialCameraState, and
    // re-framing on top of it here would immediately overwrite that exact pose
    // with an approximate one.
    if (urlFocusToken === null || !semanticSpace || urlCameraState) return;
    const hasToken = semanticSpace.nodes.some(
      (n) => n.label === urlFocusToken || n.token_id === urlFocusToken || n.id === urlFocusToken,
    );
    if (!hasToken) return;
    const focusCamera = (window as any).__focusCamera;
    if (typeof focusCamera !== 'function') return;
    focusCamera(urlFocusToken, metricK, urlFocusAnimate, urlFocusZoom);
  }, [semanticSpace, metricK]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem('visualizerSettings');
      if (stored) {
        const p = JSON.parse(stored);
        if (p.currentFilter) setCurrentFilter(p.currentFilter);
        // A URL automation param for model/algorithm/K wins over the stored
        // session, so a deep link always lands on what it asked for.
        if (p.projectionType && !urlAlgorithm) setProjectionType(p.projectionType);
        if (p.modelKey && !urlModel) setModelKey(p.modelKey);
        if (p.colorLayer) setColorLayer(p.colorLayer);
        if (p.metricK && !urlK) setMetricK(p.metricK);
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
        if (p.cameraCoords && !urlCameraState) {
          // p.cameraCoords has x, y, z, rx, ry, rz. A URL pose wins over the
          // stored session, same as the model/algorithm/k params above.
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

  // Refs mirroring the latest values of things the URL-sync settle-timer (below)
  // needs but can't close over fresh, since it's armed from inside a callback
  // that itself isn't re-created every render (handleCameraCoordsChange has an
  // empty dep array, unchanged from before, to keep it referentially stable for
  // SpaceVisualizer's per-frame call).
  const modelKeyRef = useRef(modelKey);
  const projectionTypeRef = useRef(projectionType);
  const metricKRef = useRef(metricK);
  const activeNodeRef = useRef(activeNode);
  useEffect(() => { modelKeyRef.current = modelKey; }, [modelKey]);
  useEffect(() => { projectionTypeRef.current = projectionType; }, [projectionType]);
  useEffect(() => { metricKRef.current = metricK; }, [metricK]);
  useEffect(() => { activeNodeRef.current = activeNode; }, [activeNode]);

  const lastCameraCoordsRef = useRef<{ x: number; y: number; z: number; rx: number; ry: number; rz: number } | null>(null);
  const cameraSettleTimerRef = useRef<number | undefined>(undefined);

  // Keep the address bar in sync with where navigation (WASD, gamepad, presentation
  // steps, clicking a word) has actually landed, so the current view is always a
  // copy-pasteable link. Carries two things: the ?model=&algorithm=&k=&focus= shape
  // __focusCamera already reads on load (an approximate re-frame of the nearest
  // token), *and* the exact ?camX=&camY=&camZ=&camRX=&camRY=&camRZ= pose, so sharing
  // a link reproduces exactly what was on screen -- including a free-fly view
  // centred on no particular token, which ?focus= alone can't express.
  //
  // Fed every frame (via onCameraCoordsChange -> handleCameraCoordsChange), not
  // keyed off activeNode: the reticle's nearest token only changes when the
  // camera crosses into another token's neighbourhood, so an activeNode-keyed
  // sync would miss most free-fly motion and leave camX/camY/camZ stale. Instead
  // each incoming frame's coords are epsilon-compared against the last URL-written
  // pose, and a 500ms settle-timer is reset on every meaningful change -- so the
  // URL only commits once motion has actually paused, rather than thrashing at
  // 60fps. Uses replaceState (not pushState), so this never grows browser history
  // or wires up the Back button; it only ever updates the current entry in place.
  const POS_EPSILON = 0.01;
  const ROT_EPSILON = 0.001;
  const handleCameraCoordsChange = useCallback((coords: { x: number; y: number; z: number; rx: number; ry: number; rz: number }) => {
    setCameraCoords(coords);

    const last = lastCameraCoordsRef.current;
    const changed =
      !last ||
      Math.abs(coords.x - last.x) > POS_EPSILON ||
      Math.abs(coords.y - last.y) > POS_EPSILON ||
      Math.abs(coords.z - last.z) > POS_EPSILON ||
      Math.abs(coords.rx - last.rx) > ROT_EPSILON ||
      Math.abs(coords.ry - last.ry) > ROT_EPSILON ||
      Math.abs(coords.rz - last.rz) > ROT_EPSILON;
    if (!changed) return;
    lastCameraCoordsRef.current = coords;

    if (cameraSettleTimerRef.current !== undefined) {
      window.clearTimeout(cameraSettleTimerRef.current);
    }
    cameraSettleTimerRef.current = window.setTimeout(() => {
      const settled = lastCameraCoordsRef.current;
      if (!settled) return;
      const params = new URLSearchParams();
      params.set('model', modelKeyRef.current);
      params.set('algorithm', projectionTypeRef.current);
      params.set('k', String(metricKRef.current));
      const node = activeNodeRef.current;
      const focusVal = node ? (node.label ?? node.token_id ?? node.id) : undefined;
      if (focusVal !== undefined) params.set('focus', String(focusVal));
      params.set('camX', settled.x.toFixed(4));
      params.set('camY', settled.y.toFixed(4));
      params.set('camZ', settled.z.toFixed(4));
      params.set('camRX', settled.rx.toFixed(4));
      params.set('camRY', settled.ry.toFixed(4));
      params.set('camRZ', settled.rz.toFixed(4));
      const newSearch = `?${params.toString()}`;
      if (newSearch !== window.location.search) {
        window.history.replaceState(null, '', `${window.location.pathname}${newSearch}`);
      }
    }, 500);
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
        onModelKeyChange={changeModel}
        externals={externals}
        onImportClick={() => setImportOpen(true)}
        projectionType={projectionType}
        onProjectionTypeChange={setProjectionType}
        colorLayer={colorLayer}
        onColorLayerChange={setColorLayer}
        metricK={metricK}
        kMax={semanticSpace?.meta?.K_slider_max ?? 20}
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
            gamepadFreeFlyEnabled={!presentationMode}
            initialCameraState={initialCameraState}
            data={semanticSpace}
          />
        )}

        <PresentationRemote
          activeNode={activeNode}
          metricK={metricK}
          semanticSpace={semanticSpace}
          presentationMode={presentationMode}
          onPresentationModeChange={setPresentationMode}
        />

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
        modelKey={modelKey}
      />

      {importOpen && <ImportDialog onClose={() => setImportOpen(false)} onImported={handleImported} />}
      
    </div>
  );
}
