import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Algorithm, ColorLayer, SemanticNode, SemanticSpace, probeSets } from "../types";

const EMPTY_NODES: SemanticNode[] = [];

/** Lightweight per-token handle; the rendered geometry is one shared Points cloud. */
interface TokenHandle {
  position: THREE.Vector3;
  visible: boolean;
  userData: any;
}

export interface PerfSample {
  fps: number;
  frameTimeMs: number;
  selectionLoopMs: number;
  drawCalls: number;
  geometries: number;
  textures: number;
  visibleLabels: number;
}

interface LabelEntry {
  dom: HTMLDivElement;
  position: THREE.Vector3;
  type: "token" | "latex";
  userData?: SemanticNode;
  axisId?: "x" | "y" | "z";
}

const LABEL_GAP = 5; // px between a token's dot and its word
// Token words are drawn on one 2D canvas over the WebGL view, as in the TF Projector:
// a white stroke under each word keeps it legible over dots of its own colour.
const LABEL_FONT = '"Times New Roman", Times, Georgia, serif';
const LABEL_SIZE = 13; // px at scale 1
const LABEL_SIZE_AIMED = 18;
const LABEL_HALO = 3; // px, white stroke width
const MAX_TOKEN_LABELS = 450; // upper bound; in practice set by screen-space collisions (plan 8.2)

// Layer 2 (plan 9.7): behaviour map. A fixed scale (same value, same colour in every
// projection) in 7 steps of 0.1 centred on 0. Most B_NX,i lie within ±0.3, so a
// continuous ramp over [-1, 1] left nearly every token pale. Colour-blind-safe
// diverging palette (ColorBrewer RdBu without its near-white centre); the neutral
// step is mid-grey so neutral tokens stay visible on white. Never green (probing).
export const HEAT_STEPS: { max: number; color: string; label: string }[] = [
  { max: -0.25, color: "#2166ac", label: "≤ −0.3" },
  { max: -0.15, color: "#4393c3", label: "−0.2" },
  { max: -0.05, color: "#92c5de", label: "−0.1" },
  { max: 0.05, color: "#bababa", label: "0" },
  { max: 0.15, color: "#f4a582", label: "+0.1" },
  { max: 0.25, color: "#d6604d", label: "+0.2" },
  { max: Infinity, color: "#b2182b", label: "≥ +0.3" },
];
const HEAT_STEP_COLORS = HEAT_STEPS.map((s) => new THREE.Color(s.color));
// Layer 3 (plan 9.7): probing.
const PROBE_TRUE = new THREE.Color("#16a34a");
const PROBE_FALSE = new THREE.Color("#dc2626");
const PROBE_MISSING = new THREE.Color("#3b82f6");
const PROBE_SELECTED = new THREE.Color("#111111");
// Tokens outside the probed neighbourhood stay opaque but neutral, so the cloud stays legible
const PROBE_CONTEXT = new THREE.Color("#a1a1aa");
const PROBE_CONTEXT_LABEL = "#52525b";

/** Behaviour-map step of a local value: the nearest multiple of 0.1, capped at ±0.3
 *  (boundaries at ±0.05, ±0.15, ±0.25; a value on a boundary goes to the smaller |step|). */
export const heatStep = (b: number) =>
  3 + Math.sign(b) * Math.min(3, Math.round(Math.abs(b) / 0.1 - 1e-9));
export const heatColor = (b: number) => HEAT_STEP_COLORS[heatStep(b)].clone();
/** Dot size in the behaviour map: 1.0 at 0, 1.6 at |b| ≥ 0.3, so extremes stand out by size too. */
export const heatScale = (b: number) => 1 + 0.6 * Math.min(1, Math.abs(b) / 0.3);

interface SpaceVisualizerProps {
  colorLayer: ColorLayer;
  heatK: number;
  probeEnabled: boolean;
  probeK: number;
  onPerfUpdate?: (sample: PerfSample) => void;
  projectionType: Algorithm;
  currentFilter: "all" | "positive" | "negative";
  cameraMode: "perspective" | "parallel" | "twopoint";
  fov: number;
  onNodeHovered: (node: SemanticNode | null) => void;
  onCameraCoordsChange: (coords: {
    x: number;
    y: number;
    z: number;
    rx: number;
    ry: number;
    rz: number;
  }) => void;
  onFPSUpdate: (fps: number) => void;
  cameraPreset: "front" | "top" | "side" | "iso" | null;
  onCameraPresetApplied: () => void;
  showXAxis: boolean;
  showYAxis: boolean;
  showZAxis: boolean;
  showGridPlanes: boolean;
  showPlaneBackground: boolean;
  mouseSensitivity: number;
  initialCameraState?: {
    x: number;
    y: number;
    z: number;
    rx: number;
    ry: number;
    rz: number;
  };
  data?: SemanticSpace | null;
}

export const SpaceVisualizer: React.FC<SpaceVisualizerProps> = ({
  colorLayer,
  heatK,
  probeEnabled,
  probeK,
  onPerfUpdate,
  projectionType,
  currentFilter,
  cameraMode,
  fov,
  onNodeHovered,
  onCameraCoordsChange,
  onFPSUpdate,
  cameraPreset,
  onCameraPresetApplied,
  showXAxis,
  showYAxis,
  showZAxis,
  showGridPlanes,
  showPlaneBackground,
  mouseSensitivity,
  initialCameraState,
  data,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const labelsContainerRef = useRef<HTMLDivElement>(null);
  const tokenCanvasRef = useRef<HTMLCanvasElement>(null);
  // screen boxes (CSS px) of the words drawn in the last frame, for click-to-pin
  const placedBoxesRef = useRef<{ id: number; x0: number; y0: number; x1: number; y1: number }[]>([]);
  const labelWidthRef = useRef<Map<string, number>>(new Map());
  const gridNumbersContainerRef = useRef<HTMLDivElement>(null);
  const minimapCanvasRef = useRef<HTMLCanvasElement>(null);

  // States
  const [isLocked, setIsLocked] = useState(false);

  // Refs for ThreeJS objects
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const perspCameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const orthoCameraRef = useRef<THREE.OrthographicCamera | null>(null);
  const activeCameraRef = useRef<THREE.Camera | null>(null);
  const tokenPointsRef = useRef<TokenHandle[]>([]);
  const cloudRef = useRef<THREE.Points | null>(null);
  const tokenLabelByIdRef = useRef<Map<number, LabelEntry>>(new Map());
  const tokenSetKeyRef = useRef<string>("");
  const colorLayerRef = useRef(colorLayer);
  const heatKRef = useRef(heatK);
  const probeEnabledRef = useRef(probeEnabled);
  const probeKRef = useRef(probeK);
  const hoveredIdRef = useRef<number | null>(null);
  const pinnedIdRef = useRef<number | null>(null);
  const colorsDirtyRef = useRef(true);
  const perfRef = useRef({ frames: 0, frameTime: 0, selection: 0, visibleLabels: 0 });
  const onPerfUpdateRef = useRef(onPerfUpdate);
  const gridGroupRef = useRef<THREE.Group | null>(null);
  const lastTimeRef = useRef(performance.now());

  // Input states
  const flyRef = useRef({
    F: false,
    B: false,
    L: false,
    R: false,
    U: false,
    D: false,
  });
  const targetLerpStateRef = useRef<{
    active: boolean;
    position: THREE.Vector3;
    rotation: THREE.Euler;
  } | null>(null);

  // Floating coordinates labels & grid scales
  const labelElementsRef = useRef<LabelEntry[]>([]);
  const gridNumberElementsRef = useRef<
    { dom: HTMLDivElement; position: THREE.Vector3 }[]
  >([]);

  const currentStepResolutionRef = useRef<number>(6);
  const currentFilterRef = useRef(currentFilter);
  const currentModeRef = useRef(cameraMode);
  const currentFovRef = useRef(fov);
  const fovFadeProgressRef = useRef(1);
  const projectionTypeRef = useRef(projectionType);
  const latestDataRef = useRef<SemanticSpace | null>(data ?? null);
  const unlockBlockUntilRef = useRef(0);

  const showXAxisRef = useRef(showXAxis);
  const showYAxisRef = useRef(showYAxis);
  const showZAxisRef = useRef(showZAxis);
  const showGridPlanesRef = useRef(showGridPlanes);
  const showPlaneBackgroundRef = useRef(showPlaneBackground);
  const mouseSensitivityRef = useRef(mouseSensitivity);

  const SIZE = 60; // CONSTANT SIZE OF THE COORDINATE BOX

  // Runtime guards for out-of-range coordinates introduced during debugging.
  const MAX_COORD = 10000; // generous cap to catch runaway values
  const isValidVector = (v: THREE.Vector3) =>
    Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z) &&
    Math.abs(v.x) < MAX_COORD && Math.abs(v.y) < MAX_COORD && Math.abs(v.z) < MAX_COORD;

  const safeApplyPosition = (
    pts: TokenHandle,
    target: THREE.Vector3,
    context = "",
  ) => {
    const node = pts.userData as SemanticNode | undefined;
    if (!isValidVector(target)) {
      // Helpful console output for tracing provenance in the browser console.
      // Keep this log lightweight so it can be pasted into the chat if needed.
      console.warn("SpaceVisualizer: rejected out-of-range position", {
        node: node ? (node.label ?? node.token_id ?? node.id) : "<unknown>",
        target: [target.x, target.y, target.z],
        context,
        stack: new Error().stack,
      });
      return false;
    }
    pts.position.copy(target);
    return true;
  };

  useEffect(() => {
    mouseSensitivityRef.current = mouseSensitivity;
  }, [mouseSensitivity]);

  useEffect(() => {
    showXAxisRef.current = showXAxis;
    showYAxisRef.current = showYAxis;
    showZAxisRef.current = showZAxis;
    showGridPlanesRef.current = showGridPlanes;
    showPlaneBackgroundRef.current = showPlaneBackground;
    rebuildGridPrivate();
  }, [showXAxis, showYAxis, showZAxis, showGridPlanes, showPlaneBackground]);

  useEffect(() => {
    currentFilterRef.current = currentFilter;
    applyFilterVisibilityPrivate();
  }, [currentFilter]);

  useEffect(() => {
    latestDataRef.current = data ?? null;
  }, [data]);

  useEffect(() => {
    projectionTypeRef.current = projectionType;
  }, [projectionType]);

  useEffect(() => {
    if (!sceneRef.current) return;
    syncProjectionDataPrivate();
  }, [data]);

  useEffect(() => {
    colorLayerRef.current = colorLayer;
    heatKRef.current = heatK;
    probeEnabledRef.current = probeEnabled;
    probeKRef.current = probeK;
    colorsDirtyRef.current = true;
  }, [colorLayer, heatK, probeEnabled, probeK]);

  useEffect(() => {
    onPerfUpdateRef.current = onPerfUpdate;
  }, [onPerfUpdate]);

  useEffect(() => {
    currentModeRef.current = cameraMode;
    switchCameraModePrivate(cameraMode);
  }, [cameraMode]);

  useEffect(() => {
    currentFovRef.current = fov;
    fovFadeProgressRef.current = 0;
    updateFovPrivate(fov);
  }, [fov]);

  useEffect(() => {
    if (cameraPreset) {
      applyCameraPresetPrivate(cameraPreset);
      onCameraPresetApplied();
    }
  }, [cameraPreset]);

  // Handle pointer lock state
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resetFlightState = () => {
      flyRef.current.F = false;
      flyRef.current.B = false;
      flyRef.current.L = false;
      flyRef.current.R = false;
      flyRef.current.U = false;
      flyRef.current.D = false;
    };

    const onLockChange = () => {
      const locked = document.pointerLockElement === container;
      setIsLocked(locked);
      if (!locked) {
        unlockBlockUntilRef.current = performance.now() + 200;
        resetFlightState();
      }
    };

    const onWindowBlur = () => {
      resetFlightState();
    };

    const onVisibilityChange = () => {
      if (document.visibilityState !== "visible") {
        resetFlightState();
      }
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (document.pointerLockElement !== container && e.code !== 'KeyP') return;

      targetLerpStateRef.current = null; // Interrupt lerp on keyboard movement

      if (e.code === "KeyW") flyRef.current.F = true;
      if (e.code === "KeyS") flyRef.current.B = true;
      if (e.code === "KeyA") flyRef.current.L = true;
      if (e.code === "KeyD") flyRef.current.R = true;
      if (e.code === "KeyR") flyRef.current.U = true;
      if (e.code === "KeyF") flyRef.current.D = true;

      // Update keyboard indicators in the UI
      const kIndicator = document.getElementById(`key-${e.code}`);
      if (kIndicator) kIndicator.classList.add("active");

      // Debug: press 'P' (while not pointer-locked) to print cloud stats
      if (e.code === 'KeyP') {
        logTokenCloudDebug();
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "KeyW") flyRef.current.F = false;
      if (e.code === "KeyS") flyRef.current.B = false;
      if (e.code === "KeyA") flyRef.current.L = false;
      if (e.code === "KeyD") flyRef.current.R = false;
      if (e.code === "KeyR") flyRef.current.U = false;
      if (e.code === "KeyF") flyRef.current.D = false;

      const kIndicator = document.getElementById(`key-${e.code}`);
      if (kIndicator) kIndicator.classList.remove("active");
    };

    const logTokenCloudDebug = () => {
      const pts = tokenPointsRef.current;
      if (!pts || pts.length === 0) {
        // eslint-disable-next-line no-console
        console.log('Token cloud: no points present');
        return;
      }

      let mn = [Infinity, Infinity, Infinity];
      let mx = [-Infinity, -Infinity, -Infinity];
      const distances: { idx: number; d: number }[] = [];

      pts.forEach((p, i) => {
        const x = p.position.x;
        const y = p.position.y;
        const z = p.position.z;
        mn[0] = Math.min(mn[0], x);
        mn[1] = Math.min(mn[1], y);
        mn[2] = Math.min(mn[2], z);
        mx[0] = Math.max(mx[0], x);
        mx[1] = Math.max(mx[1], y);
        mx[2] = Math.max(mx[2], z);
        distances.push({ idx: i, d: Math.hypot(x, y, z) });
      });

      distances.sort((a, b) => b.d - a.d);

      // Build duplicate position map (rounded to 2 decimals)
      const dupMap = new Map<string, number>();
      pts.forEach((p) => {
        const key = `${p.position.x.toFixed(2)},${p.position.y.toFixed(2)},${p.position.z.toFixed(2)}`;
        dupMap.set(key, (dupMap.get(key) || 0) + 1);
      });

      const duplicates = Array.from(dupMap.entries()).filter(([, c]) => c > 1).map(([pos, c]) => ({ pos, count: c }));

      const bboxSize = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
      const diag = Math.hypot(bboxSize[0], bboxSize[1], bboxSize[2]);

      // Mark outliers as points with distance > diag * 1.5
      const outlierThreshold = Math.max(1, diag * 1.5);
      const outliers = distances.filter((d) => d.d > outlierThreshold).slice(0, 24).map((s) => {
        const p = pts[s.idx];
        const node = p.userData as SemanticNode;
        return { idx: s.idx, d: Number(s.d.toFixed(2)), label: node?.label, pos: [Number(p.position.x.toFixed(2)), Number(p.position.y.toFixed(2)), Number(p.position.z.toFixed(2))] };
      });

      const top = distances.slice(0, 12).map((s) => {
        const p = pts[s.idx];
        const node = p.userData as SemanticNode;
        return { idx: s.idx, d: Number(s.d.toFixed(2)), label: node?.label, pos: [Number(p.position.x.toFixed(2)), Number(p.position.y.toFixed(2)), Number(p.position.z.toFixed(2))] };
      });

      // eslint-disable-next-line no-console
      console.log(JSON.stringify({ bounds: { mn, mx, count: pts.length, bboxSize, diag }, duplicates, outliers, farthest: top }, null, 2));
    };

    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== container) return;
      if (performance.now() < unlockBlockUntilRef.current) return;
      const camera = activeCameraRef.current;
      if (!camera) return;

      targetLerpStateRef.current = null; // Interrupt lerp on mouse move

      const LOOK_SPEED = mouseSensitivityRef.current;
      const clampedMovementX = THREE.MathUtils.clamp(e.movementX, -60, 60);
      const clampedMovementY = THREE.MathUtils.clamp(e.movementY, -60, 60);
      camera.rotation.order = "YXZ";
      camera.rotation.y -= clampedMovementX * LOOK_SPEED;
      camera.rotation.x -= clampedMovementY * LOOK_SPEED;
      camera.rotation.x = Math.max(
        -Math.PI / 2 + 0.05,
        Math.min(Math.PI / 2 - 0.05, camera.rotation.x),
      );
    };

    const onWheel = (e: WheelEvent) => {
      const camera = activeCameraRef.current;
      if (!camera) return;

      targetLerpStateRef.current = null; // Interrupt lerp on zoom

      e.preventDefault();

      if (camera instanceof THREE.OrthographicCamera) {
        const ZOOM_SPEED = 0.1;
        const zoomFactor = e.deltaY > 0 ? 1 - ZOOM_SPEED : 1 + ZOOM_SPEED;
        camera.zoom = Math.max(0.1, Math.min(20, camera.zoom * zoomFactor));
        camera.updateProjectionMatrix();
      } else {
        const MOVE_SPEED = 0.03;
        camera.translateZ(e.deltaY * MOVE_SPEED);
      }
    };

    document.addEventListener("pointerlockchange", onLockChange);
    window.addEventListener("blur", onWindowBlur);
    document.addEventListener("visibilitychange", onVisibilityChange);
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    document.addEventListener("mousemove", onMouseMove);
    container.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      document.removeEventListener("pointerlockchange", onLockChange);
      window.removeEventListener("blur", onWindowBlur);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
      document.removeEventListener("mousemove", onMouseMove);
      container.removeEventListener("wheel", onWheel);
    };
  }, []);

  // Set up Three.js visualizer
  useEffect(() => {
    if (!containerRef.current) return;

    if (labelsContainerRef.current) labelsContainerRef.current.innerHTML = "";
    if (gridNumbersContainerRef.current) gridNumbersContainerRef.current.innerHTML = "";

    // Reset refs to prevent duplication in React StrictMode
    labelElementsRef.current = [];
    tokenPointsRef.current = [];

    // Create scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0xfcfcfc, 0.003);
    sceneRef.current = scene;

    const width = containerRef.current.clientWidth || 600;
    const height = containerRef.current.clientHeight || 600;
    const aspect = width / height;

    // Initialize Perspective Camera
    const perspCamera = new THREE.PerspectiveCamera(
      currentFovRef.current,
      aspect,
      0.1,
      600,
    );
    perspCamera.rotation.order = "YXZ";
    if (initialCameraState) {
      perspCamera.position.set(
        initialCameraState.x,
        initialCameraState.y,
        initialCameraState.z,
      );
      perspCamera.rotation.set(
        initialCameraState.rx,
        initialCameraState.ry,
        initialCameraState.rz,
      );
    } else {
      perspCamera.position.set(22, 16, 32); // Beautiful default angle pointing down at semantic space
      perspCamera.rotation.y = 0.5;
      perspCamera.rotation.x = -0.3;
    }
    perspCameraRef.current = perspCamera;

    // Initialize Orthographic Camera
    const distance = perspCamera.position.length();
    const halfHeight =
      distance * Math.tan(THREE.MathUtils.degToRad(currentFovRef.current / 2));
    const halfWidth = halfHeight * aspect;
    const orthoCamera = new THREE.OrthographicCamera(
      -halfWidth,
      halfWidth,
      halfHeight,
      -halfHeight,
      0.1,
      600,
    );
    orthoCamera.position.copy(perspCamera.position);
    orthoCamera.rotation.copy(perspCamera.rotation);
    orthoCameraRef.current = orthoCamera;

    // Set active camera
    activeCameraRef.current = perspCamera;

    // Initialize WebGLRenderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xfcfcfc);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Added Ambient Light
    const ambientLight = new THREE.AmbientLight(0xffffff, 1);
    scene.add(ambientLight);

    // Grid Group
    const gridGroup = new THREE.Group();
    scene.add(gridGroup);
    gridGroupRef.current = gridGroup;

    // Add Axes labels & base meshes
    setupOriginLabel();
    setupBaseStructure();
    setupParticlesBackground();

    // Load nodes data and create points and labels
    createTokenSystems();

    // Trigger initial filter and grid ticks
    filterNodesPrivate();
    rebuildGridPrivate();

    // Start FPS calculations
    let frameCount = 0;
    let lastTime = performance.now();
    let animationFrameId: number;

    const updateOrthoBoundsPrivate = () => {
      const activeCamera = activeCameraRef.current;
      const ortho = orthoCameraRef.current;
      const persp = perspCameraRef.current;
      if (!ortho || !persp || !containerRef.current) return;

      const w = containerRef.current.clientWidth || 300;
      const h = containerRef.current.clientHeight || 300;
      const asp = w / h;
      const dist = persp.position.length();
      const halfH =
        dist * Math.tan(THREE.MathUtils.degToRad(currentFovRef.current / 2));
      const halfW = halfH * asp;

      ortho.left = -halfW;
      ortho.right = halfW;
      ortho.top = halfH;
      ortho.bottom = -halfH;
      ortho.updateProjectionMatrix();
    };

    // Main render loop
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      const now = performance.now();
      const dt = (now - lastTimeRef.current) / 1000;
      lastTimeRef.current = now;

      applyFlightPrivate(dt);

      const activeCamera = activeCameraRef.current;
      const persp = perspCameraRef.current;
      const ortho = orthoCameraRef.current;

      const targetState = targetLerpStateRef.current;

      // Processamento da cinemática orbital de interpolação amortecida
      if (targetState?.active && persp && activeCamera) {
        // Amortecimento no vetor de posição (Fator de LERP: 0.06 por quadro)
        persp.position.lerp(targetState.position, 0.06);

        // Amortecimento na rotação utilizando a Esfera Quatérnio (SLERP) para evitar travamento de cardan (Gimbal Lock)
        const currQ = new THREE.Quaternion().setFromEuler(persp.rotation);
        const targetQ = new THREE.Quaternion().setFromEuler(
          targetState.rotation,
        );
        currQ.slerp(targetQ, 0.06);
        persp.rotation.setFromQuaternion(currQ, "YXZ");

        // Seletor de proximidade: desliga o loop quando a distância atinge limiares microscópicos
        if (
          persp.position.distanceTo(targetState.position) < 0.01 &&
          currQ.angleTo(targetQ) < 0.01
        ) {
          persp.position.copy(targetState.position);
          persp.rotation.copy(targetState.rotation);
          targetLerpStateRef.current = null;
        }
      }

      if (activeCamera && persp && ortho) {
        // Synchronize positions/rotations between standard perspective and projection parallel
        if (currentModeRef.current === "parallel") {
          updateOrthoBoundsPrivate();
          if (targetState?.active) {
            // a preset transition drives the perspective camera; the orthographic one follows it
            ortho.position.copy(persp.position);
            ortho.rotation.copy(persp.rotation);
          } else {
            persp.position.copy(ortho.position);
            persp.rotation.copy(ortho.rotation);
          }
        } else {
          ortho.position.copy(persp.position);
          ortho.rotation.copy(persp.rotation);
        }

        // Shear matrix computation if two-point perspective is requested
        if (currentModeRef.current === "twopoint") {
          persp.updateProjectionMatrix();
          persp.projectionMatrix.elements[9] = Math.tan(persp.rotation.x);
        } else {
          persp.projectionMatrix.elements[9] = 0;
        }

        // Monitor camera focal distance for interval tick calibration (GeoGebra Zoom)
        checkZoomResolutionPrivate(activeCamera.position.length());

        // Smoothly interpolate token positions (animation only; endpoints are the exact projection)
        const cloud = cloudRef.current;
        const posAttr = cloud?.geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
        tokenPointsRef.current.forEach((pts, idx) => {
          if (pts.userData && pts.userData.targetPosition) {
            pts.position.lerp(pts.userData.targetPosition, 0.04);
            const label = tokenLabelByIdRef.current.get(pts.userData.id);
            if (label) label.position.copy(pts.position);
            if (posAttr) posAttr.setXYZ(idx, pts.position.x, pts.position.y, pts.position.z);
          }
        });
        if (posAttr) {
          posAttr.needsUpdate = true;
          cloud!.geometry.computeBoundingSphere();
        }

        // Target nearest nodes inside screen coordinate projections (Crosshair UX)
        const selT0 = performance.now();
        runNodeSelectionInspector();
        perfRef.current.selection += performance.now() - selT0;

        if (colorsDirtyRef.current) {
          applyColorLayersPrivate();
          colorsDirtyRef.current = false;
        }

        // Project positions to HTML floating layer
        updateOverlayLabels();

        // Call callback with active camera coordinates
        onCameraCoordsChange({
          x: activeCamera.position.x,
          y: activeCamera.position.y,
          z: activeCamera.position.z,
          rx: activeCamera.rotation.x,
          ry: activeCamera.rotation.y,
          rz: activeCamera.rotation.z,
        });

        drawMinimapPrivate(activeCamera, dt);

        updateFogPrivate(scene, activeCamera);
        renderer.render(scene, activeCamera);

        // Profiling (plan 8.4): engineering telemetry, not a scientific result
        frameCount++;
        const currTime = performance.now();
        perfRef.current.frameTime += currTime - now;
        if (currTime - lastTime >= 1000) {
          const fpsNow = (frameCount * 1000) / (currTime - lastTime);
          onFPSUpdate(Math.round(fpsNow));
          onPerfUpdateRef.current?.({
            fps: fpsNow,
            frameTimeMs: perfRef.current.frameTime / frameCount,
            selectionLoopMs: perfRef.current.selection / frameCount,
            drawCalls: renderer.info.render.calls,
            geometries: renderer.info.memory.geometries,
            textures: renderer.info.memory.textures,
            visibleLabels: perfRef.current.visibleLabels,
          });
          perfRef.current.frameTime = 0;
          perfRef.current.selection = 0;
          frameCount = 0;
          lastTime = currTime;
        }
      }
    };

    animate();

    // Use ResizeObserver for precise container resizing and fluid performance
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w === 0 || h === 0) return;

        renderer.setSize(w, h);
        if (perspCameraRef.current) {
          perspCameraRef.current.aspect = w / h;
          perspCameraRef.current.updateProjectionMatrix();
        }
        if (orthoCameraRef.current) {
          updateOrthoBoundsPrivate();
        }
      }
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    // Cleanup resources
    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      if (rendererRef.current && rendererRef.current.domElement) {
        rendererRef.current.domElement.remove();
      }

      // Cleanup DOM label overlays
      if (labelsContainerRef.current) {
        labelsContainerRef.current.innerHTML = "";
      }
      if (gridNumbersContainerRef.current) {
        gridNumbersContainerRef.current.innerHTML = "";
      }

      // Dispose webGL objects
      scene.clear();
      renderer.dispose();
    };
  }, []);

  /**
   * Depth cue as in the TF Embedding Projector: fog in the background colour from the
   * nearest dot to farthest × (2 − min(N, 5000)/5000), so small clouds get little fog.
   * Only the token dots are fogged (grid, axes and labels stay crisp); none in the
   * parallel camera, where depth has no perspective meaning.
   */
  const fogTmp = useRef(new THREE.Vector3());
  const updateFogPrivate = (scene: THREE.Scene, camera: THREE.Camera) => {
    const cloud = cloudRef.current;
    if (!cloud || !(camera instanceof THREE.PerspectiveCamera)) {
      scene.fog = null;
      return;
    }
    let near = Infinity, far = 0;
    const v = fogTmp.current;
    for (const h of tokenPointsRef.current) {
      if (!h.visible) continue;
      const d = -v.copy(h.position).applyMatrix4(camera.matrixWorldInverse).z;
      if (d <= 0) continue;
      if (d < near) near = d;
      if (d > far) far = d;
    }
    if (!isFinite(near) || far <= near) {
      scene.fog = null;
      return;
    }
    const n = tokenPointsRef.current.length;
    if (!(scene.fog instanceof THREE.Fog)) scene.fog = new THREE.Fog(0xfcfcfc, near, far);
    scene.fog.near = near;
    scene.fog.far = far * (2 - Math.min(n, 5000) / 5000);
    // fog applies to every material with fog enabled: switch it off for all but the dots
    scene.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!m || o === cloud) return;
      for (const mat of Array.isArray(m) ? m : [m]) {
        if ((mat as THREE.Material & { fog?: boolean }).fog) {
          (mat as THREE.Material & { fog?: boolean }).fog = false;
          mat.needsUpdate = true;
        }
      }
    });
  };

  /** The token whose drawn word is under a viewport point (CSS px), if any. */
  const wordAt = (clientX: number, clientY: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const x = clientX - rect.left, y = clientY - rect.top;
    // last drawn on top: search from the end
    for (let i = placedBoxesRef.current.length - 1; i >= 0; i--) {
      const b = placedBoxesRef.current[i];
      if (x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1) return b.id;
    }
    return null;
  };

  const handleContainerMouseMove = (e: React.MouseEvent) => {
    if (document.pointerLockElement === containerRef.current || !containerRef.current) return;
    containerRef.current.style.cursor = wordAt(e.clientX, e.clientY) !== null ? "pointer" : "";
  };

  const handleContainerClick = (e: React.MouseEvent) => {
    // clicking a word (cursor free) pins that token instead of starting flight
    if (document.pointerLockElement !== containerRef.current) {
      const id = wordAt(e.clientX, e.clientY);
      if (id !== null) {
        pinnedIdRef.current = id;
        return;
      }
    }
    if (document.pointerLockElement !== containerRef.current) {
      try {
        const promise = containerRef.current?.requestPointerLock();
        if (promise) {
          promise.catch((e) => {
            // Silently handle pointer lock errors - browsers restrict how often this can be called
            console.debug("Pointer lock deferred:", e);
          });
        }
      } catch (e) {
        console.debug("Pointer lock unavailable:", e);
      }
    }
  };

  // Flying vector kinematics translation
  const applyFlightPrivate = (dt: number) => {
    const activeCamera = activeCameraRef.current;
    if (!activeCamera) return;

    const fly = flyRef.current;
    const MOVE_SPEED = 14;
    const cappedDt = Math.min(dt, 0.033);
    const vel = MOVE_SPEED * cappedDt;

    const dir = new THREE.Vector3();
    const right = new THREE.Vector3();

    // W/S fly along the full aim direction (pitch included), so the camera goes
    // where the reticle points; A/D strafe horizontally and R/F stay world-vertical.
    activeCamera.getWorldDirection(dir).normalize();

    right.set(1, 0, 0).applyQuaternion(activeCamera.quaternion);
    right.y = 0;
    if (right.lengthSq() < 0.0001) {
      right.set(1, 0, 0);
    } else {
      right.normalize();
    }

    if (fly.F) activeCamera.position.addScaledVector(dir, vel);
    if (fly.B) activeCamera.position.addScaledVector(dir, -vel);
    if (fly.R) activeCamera.position.addScaledVector(right, vel);
    if (fly.L) activeCamera.position.addScaledVector(right, -vel);
    if (fly.U) activeCamera.position.y += vel;
    if (fly.D) activeCamera.position.y -= vel;
  };

  // Helper: Arrow mesh constructor
  const buildVectorArrow = (
    direction: THREE.Vector3,
    colorHex: number,
    length: number,
    parent: THREE.Group,
  ) => {
    const group = new THREE.Group();
    const thickness = 0.035;
    const coneHeight = 0.5;

    const cylinderGeo = new THREE.CylinderGeometry(
      thickness,
      thickness,
      length - coneHeight,
      8,
    );
    const material = new THREE.MeshBasicMaterial({ color: colorHex });
    const cylinder = new THREE.Mesh(cylinderGeo, material);
    cylinder.position.y = (length - coneHeight) / 2;
    group.add(cylinder);

    const coneGeo = new THREE.ConeGeometry(thickness * 3.5, coneHeight, 8);
    const cone = new THREE.Mesh(coneGeo, material);
    cone.position.y = length - coneHeight / 2;
    group.add(cone);

    if (direction.x > 0) group.rotation.z = -Math.PI / 2;
    else if (direction.x < 0) group.rotation.z = Math.PI / 2;
    else if (direction.z > 0) group.rotation.x = Math.PI / 2;
    else if (direction.z < 0) group.rotation.x = -Math.PI / 2;
    else if (direction.y < 0) group.rotation.z = Math.PI;

    parent.add(group);
  };

  const setupOriginLabel = () => {
    if (!labelsContainerRef.current) return;

    const dom = document.createElement("div");
    dom.className =
      "spatial-label absolute pointer-events-none select-none z-10";
    dom.style.color = "#000000";
    dom.style.fontSize = "14px";
    dom.style.fontStyle = "italic";
    dom.style.fontWeight = "normal";
    dom.textContent = "O";
    labelsContainerRef.current.appendChild(dom);

    labelElementsRef.current.push({
      dom,
      position: new THREE.Vector3(0.35, 0.35, 0),
      type: "token",
    });
  };

  const setupBaseStructure = () => {
    const AXIS_TOP = SIZE / 2;
    injectLatexLabelPrivate(
      "x",
      new THREE.Vector3(AXIS_TOP + 1, 0, 0),
      "latex-axis-label text-[#111111] font-bold text-sm tracking-tight",
      "x",
    );
    injectLatexLabelPrivate(
      "y",
      new THREE.Vector3(0, AXIS_TOP + 1, 0),
      "latex-axis-label text-[#111111] font-bold text-sm tracking-tight",
      "y",
    );
    injectLatexLabelPrivate(
      "z",
      new THREE.Vector3(0, 0, AXIS_TOP + 1),
      "latex-axis-label text-[#111111] font-bold text-sm tracking-tight",
      "z",
    );
  };

  const injectLatexLabelPrivate = (
    text: string,
    position: THREE.Vector3,
    className: string,
    axisId?: "x" | "y" | "z",
  ) => {
    if (!labelsContainerRef.current) return;

    const dom = document.createElement("div");
    dom.className = `absolute pointer-events-none select-none z-10 ${className}`;
    dom.innerHTML = `$${text}$`;
    labelsContainerRef.current.appendChild(dom);

    labelElementsRef.current.push({ dom, position, type: "latex", axisId });

    // Typeset newly injected LaTeX formula
    if ((window as any).MathJax && (window as any).MathJax.typesetPromise) {
      (window as any).MathJax.typesetPromise([dom]);
    }
  };

  const setupParticlesBackground = () => {
    return;
  };

  const getNodeDisplayColor = (color?: string) => color ?? "#52525b";

  const currentNodes = () => latestDataRef.current?.nodes ?? EMPTY_NODES;

  const tokenSetKey = (nodes: SemanticNode[]) =>
    `${latestDataRef.current?.meta?.dataset_hash ?? ""}:${nodes.length}:${nodes
      .map((n) => n.id)
      .join(",")}`;

  /** Remove the token cloud and token labels (axis labels are kept). */
  const disposeTokenSystemsPrivate = () => {
    if (cloudRef.current) {
      sceneRef.current?.remove(cloudRef.current);
      cloudRef.current.geometry.dispose();
      (cloudRef.current.material as THREE.PointsMaterial).map?.dispose();
      (cloudRef.current.material as THREE.Material).dispose();
      cloudRef.current = null;
    }
    labelElementsRef.current = labelElementsRef.current.filter((l) => {
      if (l.type !== "token") return true;
      l.dom.remove();
      return false;
    });
    tokenLabelByIdRef.current.clear();
    tokenPointsRef.current = [];
  };

  const createTokenSystems = () => {
    const container = labelsContainerRef.current;
    if (!container || !sceneRef.current) return;
    disposeTokenSystemsPrivate();

    const nodes = currentNodes();
    tokenSetKeyRef.current = tokenSetKey(nodes);
    if (nodes.length === 0) return;

    // One BufferGeometry for the whole cloud: one draw call regardless of N (plan 8.1)
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(nodes.length * 3), 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(nodes.length * 4).fill(1), 4),
    );
    // per-dot size factor (behaviour map: larger dots for larger |B_NX,i|)
    geometry.setAttribute(
      "aScale",
      new THREE.BufferAttribute(new Float32Array(nodes.length).fill(1), 1),
    );
    // Round sprite (PointsMaterial draws squares by default). Every token is drawn as
    // a dot; its word is shown beside the dot when there is room (updateOverlayLabels).
    const disc = document.createElement("canvas");
    disc.width = disc.height = 64;
    const dctx = disc.getContext("2d")!;
    dctx.fillStyle = "#ffffff";
    dctx.beginPath();
    dctx.arc(32, 32, 30, 0, Math.PI * 2);
    dctx.fill();
    const material = new THREE.PointsMaterial({
      size: 0.4,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      // The dots write depth and are drawn before the semi-opaque grid planes
      // (renderOrder below), so a plane dims the dots behind it but never paints over
      // dots in front of it. alphaTest discards the disc's corners, so depth stays round.
      depthWrite: true,
      map: new THREE.CanvasTexture(disc),
      alphaTest: 0.05,
    });
    material.onBeforeCompile = (shader) => {
      shader.vertexShader = "attribute float aScale;\n" +
        shader.vertexShader.replace("gl_PointSize = size;", "gl_PointSize = size * aScale;");
    };
    const cloud = new THREE.Points(geometry, material);
    cloud.frustumCulled = false;
    cloud.renderOrder = -1;
    sceneRef.current.add(cloud);
    cloudRef.current = cloud;

    nodes.forEach((node) => {
      const handle: TokenHandle = {
        position: new THREE.Vector3(),
        visible: true,
        userData: node,
      };
      tokenPointsRef.current.push(handle);

      // A detached element: never in the DOM, it only carries the word's colour and
      // state (dim, filtered, aimed) for the canvas pass in updateOverlayLabels.
      const dom = document.createElement("div");
      dom.style.color = getNodeDisplayColor(node.color);
      dom.textContent = node.label;

      const entry: LabelEntry = {
        dom,
        position: handle.position.clone(),
        userData: node,
        type: "token",
      };
      labelElementsRef.current.push(entry);
      tokenLabelByIdRef.current.set(node.id!, entry);
    });

    filterNodesPrivate();
    // First placement is immediate; later projection switches animate.
    tokenPointsRef.current.forEach((h) => {
      if (h.userData.targetPosition) safeApplyPosition(h, h.userData.targetPosition, "create");
    });
    colorsDirtyRef.current = true;

    // Expose selectToken helper on window (used by screenshot tests)
    (window as any).__selectToken = (target: string | number) => {
      const nodesList = currentNodes();
      const targetNode =
        nodesList.find(
          (n) => n.label === target || n.token_id === target || n.id === target,
        ) || nodesList[0];
      if (targetNode) {
        pinnedIdRef.current = targetNode.id ?? null;
        return true;
      }
      return false;
    };
  };

  /** Rewrite the cloud's RGBA attribute and label colours. Precedence: probe > heat > category. */
  const applyColorLayersPrivate = () => {
    const cloud = cloudRef.current;
    if (!cloud) return;
    const colorAttr = cloud.geometry.getAttribute("color") as THREE.BufferAttribute;
    const scaleAttr = cloud.geometry.getAttribute("aScale") as THREE.BufferAttribute;
    const layer = colorLayerRef.current;
    const kKey = String(heatKRef.current);

    let probe = null as ReturnType<typeof probeSets>;
    if (probeEnabledRef.current && hoveredIdRef.current !== null) {
      const sel = tokenPointsRef.current.find((h) => h.userData.id === hoveredIdRef.current);
      if (sel) probe = probeSets(sel.userData as SemanticNode, probeKRef.current);
    }

    const tmp = new THREE.Color();
    tokenPointsRef.current.forEach((h, idx) => {
      const node = h.userData as SemanticNode;
      let alpha = h.visible ? 1 : 0;
      // On the behaviour-map scale most values are near 0 (pale), so there the dot carries
      // the colour and the word stays dark for legibility, as in the TF Projector.
      let labelColor: string | null = null;
      let dotScale = 1;
      if (layer === "heat" && node.b_nx_local && kKey in node.b_nx_local) {
        tmp.copy(heatColor(node.b_nx_local[kKey]));
        dotScale = heatScale(node.b_nx_local[kKey]);
        labelColor = "#3f3f46";
      } else {
        tmp.set(getNodeDisplayColor(node.color));
      }
      if (probe) {
        const id = node.id!;
        const probed = id === probe.selected || probe.true.has(id) || probe.false.has(id) || probe.missing.has(id);
        if (id === probe.selected) tmp.copy(PROBE_SELECTED);
        else if (probe.true.has(id)) tmp.copy(PROBE_TRUE);
        else if (probe.false.has(id)) tmp.copy(PROBE_FALSE);
        else if (probe.missing.has(id)) {
          tmp.copy(PROBE_MISSING);
          alpha *= 0.6;
        } else tmp.copy(PROBE_CONTEXT);
        labelColor = probed ? null : PROBE_CONTEXT_LABEL; // probing colours apply to the word too
      }
      colorAttr.setXYZW(idx, tmp.r, tmp.g, tmp.b, alpha);
      scaleAttr?.setX(idx, probe ? 1 : dotScale);
      const label = tokenLabelByIdRef.current.get(node.id!);
      if (label) {
        label.dom.style.color = labelColor ?? `#${tmp.getHexString()}`;
        // the aimed word is drawn dark; this keeps the dot's colour on its underline
        label.dom.style.setProperty("--dot", `#${tmp.getHexString()}`);
        label.dom.dataset.dim = alpha < 0.5 && h.visible ? "1" : "0";
      }
    });
    colorAttr.needsUpdate = true;
    if (scaleAttr) scaleAttr.needsUpdate = true;
  };

  const syncProjectionDataPrivate = () => {
    const nodes = currentNodes();
    // A different dataset (or model with a different item set) needs a fresh cloud
    if (tokenSetKey(nodes) !== tokenSetKeyRef.current || tokenPointsRef.current.length === 0) {
      createTokenSystems();
      return;
    }
    const byNodeKey = new Map<number, SemanticNode>();

    const COORD_WARN = 200; // warn when incoming coords exceed this magnitude

    nodes.forEach((node) => {
      const nodeKey = node.id ?? node.token_id;
      byNodeKey.set(nodeKey, node);
    });

    tokenPointsRef.current.forEach((pts) => {
      const currentNode = pts.userData as SemanticNode;
      const currentKey = currentNode.id ?? currentNode.token_id;
      const nextNode = byNodeKey.get(currentKey);
      if (!nextNode) return;

      // If the incoming projection data contains runaway coordinates, log them
      try {
        const c = (nextNode as any).coords ?? (nextNode as any);
        const x = c?.x3d ?? c?.x ?? 0;
        const y = c?.y3d ?? c?.y ?? 0;
        const z = c?.z3d ?? c?.z ?? 0;
        if (
          !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z) ||
          Math.abs(x) > COORD_WARN || Math.abs(y) > COORD_WARN || Math.abs(z) > COORD_WARN
        ) {
          console.warn("SpaceVisualizer: syncProjectionDataPrivate received large coords", {
            key: currentKey,
            nextCoords: [x, y, z],
            sample: JSON.stringify(nextNode).slice(0, 200),
            stack: new Error().stack,
          });
        }
      } catch (err) {
        console.warn("SpaceVisualizer: error inspecting nextNode coords", err);
      }

      Object.assign(currentNode, nextNode);
      const syncedNode = currentNode as SemanticNode & {
        targetPosition?: THREE.Vector3;
      };
      pts.userData = syncedNode;

      const matchingLabel = tokenLabelByIdRef.current.get(nextNode.id!);
      if (matchingLabel) {
        matchingLabel.userData = syncedNode;
        matchingLabel.dom.textContent = nextNode.label;
      }
    });

    filterNodesPrivate();
    applyFilterVisibilityPrivate();
    colorsDirtyRef.current = true;
  };

  // Scene placement: one isotropic similarity map (common centre, one scale for
  // all three axes), which preserves every ratio of Euclidean distances, so the
  // rendered geometry is the projection itself -- no per-algorithm warps.
  // Centre = coordinate-wise median and scale = 99th-percentile radius, so a few
  // ejected points (e.g. t-SNE at low perplexity) do not squeeze the bulk; such
  // points are drawn outside the box at their true relative position.
  const filterNodesPrivate = () => {
    const filter = currentFilterRef.current;
    colorsDirtyRef.current = true;

    const nodes = currentNodes();
    const activeNodeKeys = new Set(nodes.map((n) => n.id ?? n.token_id));

    const median = (v: number[]) => {
      const s = [...v].sort((a, b) => a - b);
      const m = s.length >> 1;
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    const centre = [
      median(nodes.map((n) => n.coords.x3d)),
      median(nodes.map((n) => n.coords.y3d)),
      median(nodes.map((n) => n.coords.z3d)),
    ];
    const radii = nodes
      .map((n) => Math.hypot(n.coords.x3d - centre[0], n.coords.y3d - centre[1], n.coords.z3d - centre[2]))
      .sort((a, b) => a - b);
    const r99 = Math.max(radii[Math.min(radii.length - 1, Math.floor(0.99 * radii.length))] ?? 0, 1e-6);
    const mn = centre.map((c) => c - r99);
    const mx = centre.map((c) => c + r99);
    const range = 2 * r99;
    const S = 26;

    tokenPointsRef.current.forEach((pts) => {
      const node = pts.userData as SemanticNode;
      const matchingLabel = tokenLabelByIdRef.current.get(node.id!);

      const x = node.coords.x3d;
      const y = node.coords.y3d;
      const z = node.coords.z3d;

      const vx = ((x - (mn[0] + mx[0]) / 2) / range) * S;
      const vy = ((y - (mn[1] + mx[1]) / 2) / range) * S;
      const vz = ((z - (mn[2] + mx[2]) / 2) / range) * S;

      let shouldShow = activeNodeKeys.has(node.id ?? node.token_id);

      if (shouldShow) {
        if (filter === "positive") {
          if (vx < 0 || vy < 0 || vz < 0) shouldShow = false;
        } else if (filter === "negative") {
          if (vx > 0 || vy > 0 || vz > 0) shouldShow = false;
        }
      }

      pts.visible = shouldShow;
      if (matchingLabel) {
        matchingLabel.dom.dataset.filterVisible = shouldShow ? "1" : "0";
      }

      const targetPos = new THREE.Vector3(vx, vy, vz);

      if (!pts.userData.targetPosition) {
        // initial placement: apply to mesh only if validated
        if (safeApplyPosition(pts, targetPos, "filter-initial")) {
          pts.userData.targetPosition = targetPos.clone();
        } else {
          // still keep the target for inspection but don't apply unsafe values
          pts.userData.targetPosition = targetPos.clone();
        }
      } else {
        // always update stored target so debugger can inspect it; application to
        // pts.position is gated by safeApplyPosition where relevant
        pts.userData.targetPosition.copy(targetPos);
      }
    });
  };

  const applyFilterVisibilityPrivate = () => {
    const filter = currentFilterRef.current;
    const nodes = currentNodes();
    const activeNodeKeys = new Set(nodes.map((n) => n.id ?? n.token_id));
    colorsDirtyRef.current = true;

    tokenPointsRef.current.forEach((pts) => {
      const node = pts.userData as SemanticNode;
      const matchingLabel = tokenLabelByIdRef.current.get(node.id!);

      const targetPos = pts.userData.targetPosition || pts.position;
      let shouldShow = activeNodeKeys.has(node.id ?? node.token_id);

      if (shouldShow) {
        if (filter === "positive") {
          if (targetPos.x < 0 || targetPos.y < 0 || targetPos.z < 0) shouldShow = false;
        } else if (filter === "negative") {
          if (targetPos.x > 0 || targetPos.y > 0 || targetPos.z > 0) shouldShow = false;
        }
      }

      pts.visible = shouldShow;
      if (matchingLabel) {
        matchingLabel.dom.dataset.filterVisible = shouldShow ? "1" : "0";
      }
    });
  };

  // Re-build standard grid lines on domain filter adjustments
  const rebuildGridPrivate = () => {
    const gridGroup = gridGroupRef.current;
    if (!gridGroup) return;

    // Dispose previous grid line segments
    while (gridGroup.children.length > 0) {
      const child = gridGroup.children[0] as any;
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material))
          child.material.forEach((m: THREE.Material) => m.dispose());
        else child.material.dispose();
      }
      gridGroup.remove(child);
    }

    const DIV = 20;
    const L = 1;
    const filter = currentFilterRef.current;

    const buildGridPlaneMesh = (
      size: number,
      pos: THREE.Vector3,
      rotX: number,
      rotY: number,
      rotZ: number,
    ) => {
      if (!showPlaneBackgroundRef.current) return;
      const geom = new THREE.PlaneGeometry(size, size);
      const mat = new THREE.MeshBasicMaterial({
        color: 0xfdfdfd,
        transparent: true,
        opacity: 0.85,
        side: THREE.DoubleSide,
        depthWrite: false, // Helps grid lines and other elements render on top correctly
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.name = "bgPlane";
      mesh.position.copy(pos);
      mesh.rotation.set(rotX, rotY, rotZ);
      gridGroup.add(mesh);
    };

    if (filter === "positive") {
      const halfSize = SIZE / 2;
      const center = halfSize / 2;

      if (showGridPlanesRef.current) {
        if (showXAxisRef.current && showZAxisRef.current) {
          const gXZ = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gXZ.position.set(center, 0, center);
          gridGroup.add(gXZ);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(center, 0, center),
            Math.PI / 2,
            0,
            0,
          );
        }

        if (showXAxisRef.current && showYAxisRef.current) {
          const gXY = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gXY.rotation.x = Math.PI / 2;
          gXY.position.set(center, center, 0);
          gridGroup.add(gXY);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(center, center, 0),
            0,
            0,
            0,
          );
        }

        if (showYAxisRef.current && showZAxisRef.current) {
          const gYZ = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gYZ.rotation.z = Math.PI / 2;
          gYZ.position.set(0, center, center);
          gridGroup.add(gYZ);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(0, center, center),
            0,
            Math.PI / 2,
            0,
          );
        }
      }

      const buildAxisLine = (start: THREE.Vector3, end: THREE.Vector3) => {
        const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
        const mat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          linewidth: 2,
        });
        gridGroup.add(new THREE.Line(geo, mat));
      };

      if (showXAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(halfSize, 0, 0),
        );
      if (showYAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(0, halfSize, 0),
        );
      if (showZAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(0, 0, halfSize),
        );

      // Positive chromatic basis vectors
      if (showXAxisRef.current)
        buildVectorArrow(new THREE.Vector3(1, 0, 0), 0xff0000, L, gridGroup);
      if (showYAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, 1, 0), 0x00ff00, L, gridGroup);
      if (showZAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, 0, 1), 0x0000ff, L, gridGroup);
    } else if (filter === "negative") {
      const halfSize = SIZE / 2;
      const center = -halfSize / 2;

      if (showGridPlanesRef.current) {
        if (showXAxisRef.current && showZAxisRef.current) {
          const gXZ = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gXZ.position.set(center, 0, center);
          gridGroup.add(gXZ);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(center, 0, center),
            Math.PI / 2,
            0,
            0,
          );
        }

        if (showXAxisRef.current && showYAxisRef.current) {
          const gXY = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gXY.rotation.x = Math.PI / 2;
          gXY.position.set(center, center, 0);
          gridGroup.add(gXY);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(center, center, 0),
            0,
            0,
            0,
          );
        }

        if (showYAxisRef.current && showZAxisRef.current) {
          const gYZ = new THREE.GridHelper(
            halfSize,
            DIV / 2,
            0xe2e8f0,
            0xe2e8f0,
          );
          gYZ.rotation.z = Math.PI / 2;
          gYZ.position.set(0, center, center);
          gridGroup.add(gYZ);
          buildGridPlaneMesh(
            halfSize,
            new THREE.Vector3(0, center, center),
            0,
            Math.PI / 2,
            0,
          );
        }
      }

      const buildAxisLine = (start: THREE.Vector3, end: THREE.Vector3) => {
        const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
        const mat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          linewidth: 2,
        });
        gridGroup.add(new THREE.Line(geo, mat));
      };

      if (showXAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(-halfSize, 0, 0),
        );
      if (showYAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(0, -halfSize, 0),
        );
      if (showZAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, 0),
          new THREE.Vector3(0, 0, -halfSize),
        );

      // Negative coordinates pointing axis
      if (showXAxisRef.current)
        buildVectorArrow(new THREE.Vector3(-1, 0, 0), 0xff0000, L, gridGroup);
      if (showYAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, -1, 0), 0x00ff00, L, gridGroup);
      if (showZAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, 0, -1), 0x0000ff, L, gridGroup);
    } else {
      if (showGridPlanesRef.current) {
        if (showXAxisRef.current && showZAxisRef.current) {
          const gXZ = new THREE.GridHelper(SIZE, DIV, 0xe2e8f0, 0xe2e8f0);
          gridGroup.add(gXZ);
          buildGridPlaneMesh(
            SIZE,
            new THREE.Vector3(0, 0, 0),
            Math.PI / 2,
            0,
            0,
          );
        }

        if (showXAxisRef.current && showYAxisRef.current) {
          const gXY = new THREE.GridHelper(SIZE, DIV, 0xe2e8f0, 0xe2e8f0);
          gXY.rotation.x = Math.PI / 2;
          gridGroup.add(gXY);
          buildGridPlaneMesh(SIZE, new THREE.Vector3(0, 0, 0), 0, 0, 0);
        }

        if (showYAxisRef.current && showZAxisRef.current) {
          const gYZ = new THREE.GridHelper(SIZE, DIV, 0xe2e8f0, 0xe2e8f0);
          gYZ.rotation.z = Math.PI / 2;
          gridGroup.add(gYZ);
          buildGridPlaneMesh(
            SIZE,
            new THREE.Vector3(0, 0, 0),
            0,
            Math.PI / 2,
            0,
          );
        }
      }

      const buildAxisLine = (start: THREE.Vector3, end: THREE.Vector3) => {
        const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
        const mat = new THREE.LineBasicMaterial({
          color: 0x94a3b8,
          linewidth: 2,
        });
        gridGroup.add(new THREE.Line(geo, mat));
      };

      const halfSize = SIZE / 2;
      if (showXAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(-halfSize, 0, 0),
          new THREE.Vector3(halfSize, 0, 0),
        );
      if (showYAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, -halfSize, 0),
          new THREE.Vector3(0, halfSize, 0),
        );
      if (showZAxisRef.current)
        buildAxisLine(
          new THREE.Vector3(0, 0, -halfSize),
          new THREE.Vector3(0, 0, halfSize),
        );

      // Standard canonical axes
      if (showXAxisRef.current)
        buildVectorArrow(new THREE.Vector3(1, 0, 0), 0xff0000, L, gridGroup);
      if (showYAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, 1, 0), 0x00ff00, L, gridGroup);
      if (showZAxisRef.current)
        buildVectorArrow(new THREE.Vector3(0, 0, 1), 0x0000ff, L, gridGroup);
    }

    // Keep axis LaTeX indicators locked at the edges
    const AXIS_TOP = SIZE / 2;
    const labelX = labelElementsRef.current.find((l) => l.axisId === "x");
    const labelY = labelElementsRef.current.find((l) => l.axisId === "y");
    const labelZ = labelElementsRef.current.find((l) => l.axisId === "z");
    const labelO = labelElementsRef.current.find(
      (l) => l.type === "token" && l.dom.textContent === "O",
    );

    if (labelX)
      labelX.dom.style.display = showXAxisRef.current ? "block" : "none";
    if (labelY)
      labelY.dom.style.display = showYAxisRef.current ? "block" : "none";
    if (labelZ)
      labelZ.dom.style.display = showZAxisRef.current ? "block" : "none";

    if (labelO) {
      labelO.dom.style.display =
        showXAxisRef.current || showYAxisRef.current || showZAxisRef.current
          ? "block"
          : "none";
    }

    if (labelX && labelY && labelZ) {
      if (filter === "positive") {
        labelX.position.set(AXIS_TOP + 1, 0, 0);
        labelY.position.set(0, AXIS_TOP + 1, 0);
        labelZ.position.set(0, 0, AXIS_TOP + 1);
      } else if (filter === "negative") {
        labelX.position.set(-AXIS_TOP - 1, 0, 0);
        labelY.position.set(0, -AXIS_TOP - 1, 0);
        labelZ.position.set(0, 0, -AXIS_TOP - 1);
      } else {
        labelX.position.set(AXIS_TOP + 1, 0, 0);
        labelY.position.set(0, AXIS_TOP + 1, 0);
        labelZ.position.set(0, 0, AXIS_TOP + 1);
      }
    }

    updateDynamicGridNumbersPrivate(currentStepResolutionRef.current);
  };

  const updateDynamicGridNumbersPrivate = (step: number) => {
    const container = gridNumbersContainerRef.current;
    if (!container) return;

    container.innerHTML = "";
    gridNumberElementsRef.current = [];

    const filter = currentFilterRef.current;
    const gridGroup = gridGroupRef.current;

    for (let current = -SIZE / 2; current <= SIZE / 2; current += step) {
      if (Math.abs(current) < 0.01) continue;

      if (filter === "positive" && current < 0) continue;
      if (filter === "negative" && current > 0) continue;

      const formattedNum = Number.isInteger(current)
        ? current.toString()
        : current.toFixed(1);

      // Create coordinate scale text cards (GeoGebra Style) on X, Y and Z axes
      if (showXAxisRef.current) {
        injectAdaptiveGridNumberPrivate(
          formattedNum,
          new THREE.Vector3(current, -0.6, 0.2),
        );
      }
      if (showYAxisRef.current) {
        injectAdaptiveGridNumberPrivate(
          formattedNum,
          new THREE.Vector3(-0.6, current, 0.2),
        );
      }
      if (showZAxisRef.current) {
        injectAdaptiveGridNumberPrivate(
          formattedNum,
          new THREE.Vector3(0.2, -0.6, current),
        );
      }

      // Add visual 3D tick marks precisely at these plane line coordinates
      if (gridGroup) {
        // Thick visual tick lines (scholarly/academic plot markings)
        const tickMaterial = new THREE.LineBasicMaterial({
          color: 0x111111,
          transparent: true,
          opacity: 0.6,
        });

        // X Axis Tick
        if (showXAxisRef.current) {
          const xTickGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(current, -0.2, 0),
            new THREE.Vector3(current, 0.2, 0),
          ]);
          const xTick = new THREE.Line(xTickGeo, tickMaterial);
          gridGroup.add(xTick);
        }

        // Y Axis Tick
        if (showYAxisRef.current) {
          const yTickGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-0.2, current, 0),
            new THREE.Vector3(0.2, current, 0),
          ]);
          const yTick = new THREE.Line(yTickGeo, tickMaterial);
          gridGroup.add(yTick);
        }

        // Z Axis Tick
        if (showZAxisRef.current) {
          const zTickGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, -0.2, current),
            new THREE.Vector3(0, 0.2, current),
          ]);
          const zTick = new THREE.Line(zTickGeo, tickMaterial);
          gridGroup.add(zTick);
        }
      }
    }
  };

  const injectAdaptiveGridNumberPrivate = (
    text: string,
    position: THREE.Vector3,
  ) => {
    const container = gridNumbersContainerRef.current;
    if (!container) return;

    const dom = document.createElement("div");
    dom.className =
      "absolute pointer-events-none select-none z-10 text-[10.5px] text-[#222222] font-times-numbers font-semibold border-none bg-transparent";
    dom.textContent = text;
    container.appendChild(dom);

    gridNumberElementsRef.current.push({ dom, position });
  };

  const checkZoomResolutionPrivate = (cameraDistance: number) => {
    let targetStep = 6;

    if (cameraDistance < 15) {
      targetStep = 0.5; // Very near zoom (decimals: 0.5, 1.0, 1.5...)
    } else if (cameraDistance < 30) {
      targetStep = 1; // Near zoom (1 by 1 intervals)
    } else if (cameraDistance < 50) {
      targetStep = 2; // Standard research resolve (includes -6, -4, -2, 2, 4, 6...)
    } else if (cameraDistance < 80) {
      targetStep = 6; // Normal zoom (intervals of 6)
    } else {
      targetStep = 12; // Wide panorama zoom (intervals of 12)
    }

    if (targetStep !== currentStepResolutionRef.current) {
      currentStepResolutionRef.current = targetStep;
      rebuildGridPrivate();
    }
  };

  // Switch between perspective/parallel orthographic modes
  const switchCameraModePrivate = (
    mode: "perspective" | "parallel" | "twopoint",
  ) => {
    const persp = perspCameraRef.current;
    const ortho = orthoCameraRef.current;
    if (!persp || !ortho) return;

    if (mode === "parallel") {
      const w = containerRef.current?.clientWidth || 300;
      const h = containerRef.current?.clientHeight || 300;
      const aspect = w / h;
      const distance = persp.position.length();
      const halfHeight =
        distance *
        Math.tan(THREE.MathUtils.degToRad(currentFovRef.current / 2));
      const halfWidth = halfHeight * aspect;

      ortho.left = -halfWidth;
      ortho.right = halfWidth;
      ortho.top = halfHeight;
      ortho.bottom = -halfHeight;
      ortho.position.copy(persp.position);
      ortho.rotation.copy(persp.rotation);
      ortho.updateProjectionMatrix();

      activeCameraRef.current = ortho;
    } else {
      persp.projectionMatrix.elements[9] = 0;
      persp.updateProjectionMatrix();
      activeCameraRef.current = persp;
    }
  };

  const updateFovPrivate = (val: number) => {
    const persp = perspCameraRef.current;
    const ortho = orthoCameraRef.current;

    if (persp) {
      persp.fov = val;
      persp.updateProjectionMatrix();
    }

    if (currentModeRef.current === "parallel" && ortho && persp) {
      const w = containerRef.current?.clientWidth || 300;
      const h = containerRef.current?.clientHeight || 300;
      const aspect = w / h;
      const distance = persp.position.length();
      const halfHeight = distance * Math.tan(THREE.MathUtils.degToRad(val / 2));
      const halfWidth = halfHeight * aspect;

      ortho.left = -halfWidth;
      ortho.right = halfWidth;
      ortho.top = halfHeight;
      ortho.bottom = -halfHeight;
      ortho.updateProjectionMatrix();
    }
  };

  // Set default view angles from presets
  const applyCameraPresetPrivate = (
    preset: "front" | "top" | "side" | "iso",
  ) => {
    const persp = perspCameraRef.current;
    if (!persp) return;

    if (document.pointerLockElement) {
      document.exitPointerLock();
    }

    const targetPos = new THREE.Vector3();
    const targetRot = new THREE.Euler(0, 0, 0, "YXZ");

    if (preset === "iso") {
      targetPos.set(24, 18, 24);
      targetRot.y = Math.PI / 4;
      targetRot.x = -Math.atan2(18, Math.sqrt(24 * 24 + 24 * 24));
    } else if (preset === "front") {
      targetPos.set(0, 0, 45); // Travado no plano de face XY
    } else if (preset === "top") {
      targetPos.set(0, 45, 0); // Travado na visão aérea XZ
      targetRot.set(-Math.PI / 2, 0, 0, "YXZ");
    } else if (preset === "side") {
      targetPos.set(45, 0, 0); // Travado na visão lateral YZ
      targetRot.set(0, Math.PI / 2, 0, "YXZ");
    }

    targetLerpStateRef.current = {
      active: true,
      position: targetPos,
      rotation: targetRot,
    };

    if (currentModeRef.current === "parallel") {
      // In parallel mode, directly update the orthographic camera or switch appropriately
      switchCameraModePrivate("parallel");
    }
  };

  // Core selector: screen-space 2D proximity calculations
  const runNodeSelectionInspector = () => {
    const activeCamera = activeCameraRef.current;
    if (!activeCamera || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const cx = width / 2;
    const cy = height / 2;
    const maxDistancePX = 25; // 25px target buffer for strict aiming

    const dir = new THREE.Vector3();
    activeCamera.getWorldDirection(dir);

    let bestNode: SemanticNode | null = null;
    let minDistance = Infinity;

    tokenPointsRef.current.forEach((pts) => {
      if (!pts.visible) return;

      const node = pts.userData as SemanticNode;
      const pos = pts.position.clone();
      const toPos = pos.clone().sub(activeCamera.position);

      // Verify node is actually positioned in front of camera lens
      if (toPos.dot(dir) < 0) return;

      pos.project(activeCamera);
      const x = (pos.x * 0.5 + 0.5) * width;
      const y = (pos.y * -0.5 + 0.5) * height;

      // 2D screen coordinate distance from crosshair
      const dist2D = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

      if (dist2D < maxDistancePX && dist2D < minDistance) {
        minDistance = dist2D;
        bestNode = node;
      }
    });

    // A token chosen by label click / __selectToken stays selected until the reticle acquires another.
    if (bestNode) pinnedIdRef.current = null;
    else if (pinnedIdRef.current !== null) {
      bestNode = (tokenPointsRef.current.find((h) => h.userData.id === pinnedIdRef.current)
        ?.userData as SemanticNode) ?? null;
    }
    const bestId = (bestNode as SemanticNode | null)?.id ?? null;
    if (bestId !== hoveredIdRef.current) {
      const prev = hoveredIdRef.current !== null
        ? tokenLabelByIdRef.current.get(hoveredIdRef.current)
        : undefined;
      prev?.dom.classList.remove("highlighted-acad");
      if (bestId !== null) tokenLabelByIdRef.current.get(bestId)?.dom.classList.add("highlighted-acad");
      hoveredIdRef.current = bestId;
      if (probeEnabledRef.current) colorsDirtyRef.current = true;
    }
    onNodeHovered(bestNode);
  };

  const updateOverlayLabels = () => {
    const activeCamera = activeCameraRef.current;
    if (!activeCamera || !containerRef.current) return;

    activeCamera.updateMatrixWorld();
    activeCamera.matrixWorldInverse.copy(activeCamera.matrixWorld).invert();

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const dir = new THREE.Vector3();
    activeCamera.getWorldDirection(dir);

    const tempV = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();
    const bgPlanes = gridGroupRef.current
      ? gridGroupRef.current.children.filter((obj) => obj.name === "bgPlane")
      : [];

    // Every label is first projected to the screen; token labels are then placed
    // greedily, nearest first, wherever their box does not overlap an already
    // placed label (the strategy of the TensorFlow Embedding Projector). Labels
    // stay anchored to their points; a token whose label does not fit keeps its
    // dot. Zooming in spreads the points, so more words fit.
    const projectedLabels: Array<{
      dom: HTMLDivElement;
      x: number;
      y: number;
      dist: number;
      fade: number;
      scale: number;
      type: string;
      id?: number;
      world: THREE.Vector3;
      isOrigin?: boolean;
    }> = [];

    labelElementsRef.current.forEach(({ dom, position, type, axisId, userData }) => {
      if (
        (axisId === "x" && !showXAxisRef.current) ||
        (axisId === "y" && !showYAxisRef.current) ||
        (axisId === "z" && !showZAxisRef.current)
      ) {
        dom.style.opacity = "0";
        return;
      }

      if (dom.textContent === "O") {
        const factor = Math.max(0.4, Math.min(4.0, activeCamera.position.length() / 30.0));
        tempV.copy(position).multiplyScalar(factor);
      } else {
        tempV.copy(position);
      }
      const toPos = tempV.clone().sub(activeCamera.position);
      if (toPos.dot(dir) < 0) {
        if (dom.style.opacity !== "0") dom.style.opacity = "0";
        return;
      }
      const dist = toPos.length();
      const world = tempV.clone();

      tempV.project(activeCamera);
      const x = (tempV.x * 0.5 + 0.5) * width;
      const y = (tempV.y * -0.5 + 0.5) * height;
      if (type === "token" && (x < -40 || x > width + 40 || y < -20 || y > height + 20)) {
        if (dom.style.opacity !== "0") dom.style.opacity = "0";
        return;
      }

      const isOrigin = dom.textContent === "O";
      let fade: number;
      let scale = Math.max(0.3, 1 - dist / 120);
      if (type === "token") {
        // distance still fades words, but a placed word never drops below legibility
        fade = Math.max(0.8, Math.min(1, 1 - (dist - 35) / 90));
        if ((dom as HTMLElement).dataset.filterVisible === "0") fade = 0;
        if ((dom as HTMLElement).dataset.dim === "1") fade *= 0.3; // outside the probed neighbourhood
      } else {
        fade = Math.max(0, Math.min(1, 1 - (dist - 55) / 90));
      }
      if (isOrigin) {
        // Origin label "O" scale grows with distance to remain a visible anchor, capped at 1.3
        scale = Math.min(1.3, 0.7 + dist * 0.012);
        fade = Math.max(0.6, 1 - dist / 180);
      }

      projectedLabels.push({ dom, x, y, dist, fade, scale, type, id: userData?.id, world, isOrigin });
    });

    // Greedy collision-free placement of token labels
    const CELL = 6;
    const occupied = new Set<number>();
    const cols = Math.ceil(width / CELL) + 2;
    const tokens = projectedLabels
      .filter((p) => p.type === "token" && p.fade > 0)
      .sort((a, b) =>
        (a.id === hoveredIdRef.current ? -1 : 0) - (b.id === hoveredIdRef.current ? -1 : 0) || a.dist - b.dist);
    const placed = new Set<HTMLDivElement>();
    const canvas = tokenCanvasRef.current;
    const ctx = canvas?.getContext("2d") ?? null;
    const fontOf = (aimed: boolean, size: number) =>
      `italic ${aimed ? "bold " : ""}${size.toFixed(2)}px ${LABEL_FONT}`;
    /** text width at 1 px font size (scales linearly with the size) */
    const unitWidth = (text: string, aimed: boolean) => {
      const key = (aimed ? "b:" : "r:") + text;
      let w = labelWidthRef.current.get(key);
      if (w === undefined) {
        if (ctx) {
          ctx.font = fontOf(aimed, 100);
          w = ctx.measureText(text).width / 100;
        } else w = text.length * 0.5;
        labelWidthRef.current.set(key, w);
      }
      return w;
    };
    const sizeOf = (p: { scale: number; dom: HTMLDivElement }) =>
      p.dom.classList.contains("highlighted-acad")
        ? Math.max(14, LABEL_SIZE_AIMED * p.scale)
        : LABEL_SIZE * p.scale;
    const boxes: { id: number; x0: number; y0: number; x1: number; y1: number }[] = [];
    for (const p of tokens) {
      if (placed.size >= MAX_TOKEN_LABELS) break;
      const aimed = p.dom.classList.contains("highlighted-acad");
      const size = sizeOf(p);
      const w = unitWidth(p.dom.textContent!, aimed) * size;
      const halfH = size * 0.6 + 1;
      // box of a label written to the right of its dot (LABEL_GAP px away)
      const x0 = Math.floor((p.x + LABEL_GAP - 1) / CELL);
      const x1 = Math.floor((p.x + LABEL_GAP + w + 2) / CELL);
      const y0 = Math.floor((p.y - halfH) / CELL), y1 = Math.floor((p.y + halfH) / CELL);
      let free = true;
      for (let cy = y0; cy <= y1 && free; cy++)
        for (let cx = x0; cx <= x1; cx++)
          if (occupied.has(cy * cols + cx)) { free = false; break; }
      if (!free && p.id !== hoveredIdRef.current) continue;
      for (let cy = y0; cy <= y1; cy++) for (let cx = x0; cx <= x1; cx++) occupied.add(cy * cols + cx);
      placed.add(p.dom);
      if (p.id !== undefined)
        boxes.push({ id: p.id, x0: p.x - 4, y0: p.y - halfH, x1: p.x + LABEL_GAP + w, y1: p.y + halfH });
    }
    placedBoxesRef.current = boxes;
    perfRef.current.visibleLabels = placed.size;

    // Occlusion by the semi-opaque plane backgrounds (raycast for placed labels only)
    if (bgPlanes.length > 0) {
      projectedLabels.forEach((p) => {
        if (p.type === "token" && !placed.has(p.dom)) return;
        const toPos = p.world.clone().sub(activeCamera.position);
        raycaster.set(activeCamera.position, toPos.clone().normalize());
        const hits = raycaster.intersectObjects(bgPlanes, false);
        if (hits.length > 0 && hits[0].distance < toPos.length() - 0.5) p.fade *= 0.15;
      });
    }

    // Token words: one canvas pass, far words first so nearer ones (and the aimed
    // word, drawn last) sit on top. Each word gets a white stroke, then its fill.
    if (canvas && ctx) {
      const dpr = window.devicePixelRatio || 1;
      const cw = Math.round(width * dpr), ch = Math.round(height * dpr);
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.textBaseline = "middle";
      ctx.lineJoin = "round";
      const drawn = projectedLabels
        .filter((p) => p.type === "token" && placed.has(p.dom))
        .sort((a, b) =>
          (a.dom.classList.contains("highlighted-acad") ? 1 : 0) - (b.dom.classList.contains("highlighted-acad") ? 1 : 0)
          || b.dist - a.dist);
      for (const p of drawn) {
        const aimed = p.dom.classList.contains("highlighted-acad");
        const size = sizeOf(p);
        const text = p.dom.textContent!;
        const x = p.x + LABEL_GAP;
        ctx.font = fontOf(aimed, size);
        ctx.globalAlpha = aimed ? 1 : p.fade;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = LABEL_HALO;
        ctx.strokeText(text, x, p.y);
        // the aimed word is dark so it never matches its dot; its underline keeps the dot's colour
        ctx.fillStyle = aimed ? "#111111" : p.dom.style.color || "#111111";
        ctx.fillText(text, x, p.y);
        if (aimed) {
          ctx.fillStyle = p.dom.style.getPropertyValue("--dot") || "#111111";
          ctx.fillRect(x, p.y + size * 0.55, unitWidth(text, true) * size, 2);
        }
      }
      ctx.globalAlpha = 1;
    }

    // Axis labels stay DOM elements
    projectedLabels.forEach((p) => {
      if (p.type === "token") return;
      p.dom.style.left = `${p.x}px`;
      p.dom.style.top = `${p.y}px`;
      p.dom.style.transform = `translate(-50%, -50%) scale(${p.scale})`;
      p.dom.style.opacity = p.fade.toFixed(2);
      p.dom.style.zIndex = Math.max(1, Math.floor(1000 - p.dist)).toString();
    });

    // Dynamic grid numbers
    gridNumberElementsRef.current.forEach(({ dom, position }) => {
      tempV.copy(position);
      const toPos = tempV.clone().sub(activeCamera.position);

      if (toPos.dot(dir) < 0) {
        dom.style.opacity = "0";
        return;
      }

      const dist = toPos.length();

      // Check occlusion
      let isOccluded = false;
      if (bgPlanes.length > 0) {
        raycaster.set(activeCamera.position, toPos.clone().normalize());
        const intersects = raycaster.intersectObjects(bgPlanes, false);
        if (intersects.length > 0 && intersects[0].distance < dist - 0.5) {
          isOccluded = true;
        }
      }

      tempV.project(activeCamera);
      const x = (tempV.x * 0.5 + 0.5) * width;
      const y = (tempV.y * -0.5 + 0.5) * height;

      let fade = Math.max(0, Math.min(1, 1 - (dist - 55) / 60));
      let scale = Math.max(0.3, 1 - dist / 120);

      if (isOccluded) {
        fade *= 0.15; // significantly reduce opacity if behind a background plane
      }

      dom.style.left = `${x}px`;
      dom.style.top = `${y}px`;
      dom.style.transform = `translate(-50%, -50%) scale(${scale})`;
      dom.style.opacity = fade.toFixed(2);
      dom.style.zIndex = Math.max(1, Math.floor(1000 - dist)).toString();
    });
  };

  const drawMinimapPrivate = (activeCamera: THREE.Camera, dt: number) => {
    const canvas = minimapCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) / 2 - 3;
    const extent = SIZE / 2;
    const mapRadius = radius - 2;
    const unitsToPixels = mapRadius / extent;

    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.clip();

    // Clean background.
    ctx.fillStyle = "#fcfcfc";
    ctx.fillRect(0, 0, w, h);

    const cam = activeCamera.position;
    const dir = new THREE.Vector3();
    activeCamera.getWorldDirection(dir);
    dir.y = 0;
    if (dir.lengthSq() < 0.0001) {
      dir.set(0, 0, -1);
    } else {
      dir.normalize();
    }

    const rightX = -dir.z;
    const rightZ = dir.x;

    const worldToRadar = (worldX: number, worldZ: number) => {
      const dx = worldX - cam.x;
      const dz = worldZ - cam.z;
      const localX = dx * rightX + dz * rightZ;
      const localForward = dx * dir.x + dz * dir.z;

      return {
        x: cx + localX * unitsToPixels,
        y: cy - localForward * unitsToPixels,
      };
    };

    // Tactical grid in world-space (rotates with camera).
    const gridSpacingWorld = Math.max(8, Math.round(extent / 6));
    const gridMinX = cam.x - extent;
    const gridMaxX = cam.x + extent;
    const gridMinZ = cam.z - extent;
    const gridMaxZ = cam.z + extent;

    const startX = Math.floor(gridMinX / gridSpacingWorld) * gridSpacingWorld;
    const endX = Math.ceil(gridMaxX / gridSpacingWorld) * gridSpacingWorld;
    const startZ = Math.floor(gridMinZ / gridSpacingWorld) * gridSpacingWorld;
    const endZ = Math.ceil(gridMaxZ / gridSpacingWorld) * gridSpacingWorld;

    ctx.lineWidth = 1;
    for (let x = startX; x <= endX; x += gridSpacingWorld) {
      const a = worldToRadar(x, gridMinZ);
      const b = worldToRadar(x, gridMaxZ);
      ctx.strokeStyle = "rgba(17, 17, 17, 0.14)";
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    for (let z = startZ; z <= endZ; z += gridSpacingWorld) {
      const a = worldToRadar(gridMinX, z);
      const b = worldToRadar(gridMaxX, z);
      ctx.strokeStyle = "rgba(17, 17, 17, 0.14)";
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }

    const pointRadiusPx = 2.5;
    const pointRadiusWorld = pointRadiusPx / unitsToPixels;
    const pointsOnMap: Array<{
      node: SemanticNode;
      x: number;
      y: number;
      localX: number;
      localForward: number;
      distanceWorld: number;
      visible: boolean;
    }> = [];

    tokenPointsRef.current.forEach((pts) => {
      if (!pts.visible) return;
      const node = pts.userData as SemanticNode;
      const worldX = pts.position.x;
      const worldZ = pts.position.z;
      const p = worldToRadar(worldX, worldZ);
      const localX = worldX - cam.x;
      const localZ = worldZ - cam.z;
      const localSide = localX * rightX + localZ * rightZ;
      const localForward = localX * dir.x + localZ * dir.z;

      const dx = p.x - cx;
      const dy = p.y - cy;
      const insideMap = dx * dx + dy * dy <= mapRadius * mapRadius;

      pointsOnMap.push({
        node,
        x: p.x,
        y: p.y,
        localX: localSide,
        localForward,
        distanceWorld: Math.hypot(localSide, localForward),
        visible: insideMap,
      });
    });

    const tipLen = 6;
    const fovOriginX = cx;
    const fovOriginY = cy - tipLen;

    const fovDeg = currentFovRef.current;
    const fovRad = THREE.MathUtils.degToRad(fovDeg);
    const halfFov = fovRad / 2;
    const maxFovRangePx = mapRadius * 0.8;
    const maxFovRangeWorld = maxFovRangePx / unitsToPixels;
    const rayCount = 80;
    const rayRanges = Array.from({ length: rayCount }, () => maxFovRangeWorld);

    const toRayIndex = (theta: number) => {
      const t = (theta + halfFov) / (2 * halfFov);
      return Math.max(0, Math.min(rayCount - 1, Math.round(t * (rayCount - 1))));
    };

    pointsOnMap.forEach((point) => {
      if (!point.visible || point.localForward <= 0.01) return;

      const theta = Math.atan2(point.localX, point.localForward);
      if (Math.abs(theta) > halfFov + 0.001) return;

      const centerDist = point.distanceWorld;
      if (centerDist <= 0.001) return;

      const angularRadius = Math.asin(
        Math.min(0.98, pointRadiusWorld / centerDist),
      );
      const minTheta = Math.max(-halfFov, theta - angularRadius);
      const maxTheta = Math.min(halfFov, theta + angularRadius);
      const minIdx = toRayIndex(minTheta);
      const maxIdx = toRayIndex(maxTheta);
      const stopDist = Math.max(0.001, centerDist - pointRadiusWorld);

      for (let i = minIdx; i <= maxIdx; i += 1) {
        rayRanges[i] = Math.min(rayRanges[i], stopDist);
      }
    });

    fovFadeProgressRef.current = Math.min(
      1,
      fovFadeProgressRef.current + dt * 2.2,
    );
    const fadeT = fovFadeProgressRef.current;
    const fadeEase = 1 - Math.pow(1 - fadeT, 3);
    const fovAlpha = 0.22 * fadeEase;
    const maxVisibleRangePx = Math.max(...rayRanges) * unitsToPixels;

    ctx.beginPath();
    ctx.moveTo(fovOriginX, fovOriginY);
    for (let i = 0; i < rayCount; i += 1) {
      const theta = -halfFov + (i / (rayCount - 1)) * (2 * halfFov);
      const rPx = rayRanges[i] * unitsToPixels;
      const px = fovOriginX + Math.sin(theta) * rPx;
      const py = fovOriginY - Math.cos(theta) * rPx;
      ctx.lineTo(px, py);
    }
    ctx.closePath();

    ctx.save();
    ctx.clip();
    const gradient = ctx.createRadialGradient(
      fovOriginX,
      fovOriginY,
      0,
      fovOriginX,
      fovOriginY,
      maxVisibleRangePx,
    );
    gradient.addColorStop(0, `rgba(17, 17, 17, ${fovAlpha.toFixed(3)})`);
    gradient.addColorStop(0.32, `rgba(17, 17, 17, ${(fovAlpha * 0.72).toFixed(3)})`);
    gradient.addColorStop(0.7, `rgba(17, 17, 17, ${(fovAlpha * 0.28).toFixed(3)})`);
    gradient.addColorStop(1, "rgba(17, 17, 17, 0.00)");
    ctx.fillStyle = gradient;
    ctx.fillRect(
      fovOriginX - maxVisibleRangePx,
      fovOriginY - maxVisibleRangePx,
      maxVisibleRangePx * 2,
      maxVisibleRangePx * 2,
    );
    ctx.restore();

    const focusedPoints = new Set<number>();
    pointsOnMap.forEach((point, index) => {
      if (!point.visible || point.localForward <= 0.01) return;
      const theta = Math.atan2(point.localX, point.localForward);
      if (Math.abs(theta) > halfFov) return;

      const rayIdx = toRayIndex(theta);
      const collisionDistance = rayRanges[rayIdx] + pointRadiusWorld * 1.1;
      if (point.distanceWorld <= collisionDistance) {
        focusedPoints.add(index);
      }
    });

    // Same colour as the dots of the active layer (category / behaviour / probe)
    const colorAttr = cloudRef.current?.geometry.getAttribute("color") as THREE.BufferAttribute | undefined;
    const indexById = new Map<number, number>();
    tokenPointsRef.current.forEach((h, i) => indexById.set(h.userData.id, i));
    const dotColor = new THREE.Color();

    pointsOnMap.forEach((point, index) => {
      if (!point.visible) return;

      ctx.beginPath();
      const i = indexById.get(point.node.id!);
      if (colorAttr && i !== undefined) {
        dotColor.setRGB(colorAttr.getX(i), colorAttr.getY(i), colorAttr.getZ(i));
        ctx.fillStyle = `#${dotColor.getHexString()}`;
      } else {
        ctx.fillStyle = getNodeDisplayColor(point.node.color);
      }
      ctx.arc(point.x, point.y, pointRadiusPx, 0, Math.PI * 2);
      ctx.fill();

      if (focusedPoints.has(index)) {
        ctx.beginPath();
        ctx.strokeStyle = "#111111";
        ctx.lineWidth = 1;
        ctx.arc(point.x, point.y, pointRadiusPx + 1.4, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    const cameraBodyWidth = 10;
    const cameraBodyHeight = 18;
    const bodyX = cx - cameraBodyWidth / 2;
    const bodyY = cy - cameraBodyHeight / 2 + 3;
    const roofHalfWidth = 7;
    const roofHeight = 6;

    ctx.fillStyle = "#111111";
    ctx.fillRect(bodyX, bodyY, cameraBodyWidth, cameraBodyHeight);

    ctx.beginPath();
    ctx.moveTo(cx - roofHalfWidth, bodyY - roofHeight);
    ctx.lineTo(cx + roofHalfWidth, bodyY - roofHeight);
    ctx.lineTo(cx, bodyY);
    ctx.closePath();
    ctx.fill();

    ctx.restore();

    // Single outer border.
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 2;
    ctx.stroke();
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full select-none outline-none overflow-hidden"
      onClick={handleContainerClick}
      onMouseMove={handleContainerMouseMove}
    >
      {/* Target Reticle Crosshair HUD */}
      <div
        className="absolute top-1/2 left-1/2 w-[12px] h-[12px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-black/30 pointer-events-none z-20
          before:content-[''] before:absolute before:bg-black/25 before:w-[1px] before:h-[6px] before:left-1/2 before:-top-[8px] before:-translate-x-1/2
          after:content-[''] after:absolute after:bg-black/25 after:w-[6px] after:h-[1px] after:top-1/2 after:-left-[8px] after:-translate-y-1/2"
      />

      {/* Token words (one canvas) and axis labels (DOM) over the WebGL view */}
      <canvas
        ref={tokenCanvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none select-none z-10"
      />
      <div
        ref={labelsContainerRef}
        className="absolute inset-0 w-full h-full pointer-events-none select-none z-10"
      />
      <div
        ref={gridNumbersContainerRef}
        className="absolute inset-0 w-full h-full pointer-events-none select-none z-0"
      />

      {/* Interactive cursor pointer capture banner */}
      {!isLocked && (
        <div data-hud="lock-hint" className="absolute top-[20px] left-1/2 -translate-x-1/2 bg-white/98 border border-[#111111] px-5 py-[10px] z-50 shadow-[3px_3px_0px_rgba(0,0,0,1)] font-sans pointer-events-none transition-all duration-200 rounded-[4px] flex items-center justify-center">
          <span className="inline-flex items-center whitespace-nowrap font-sans text-[12px] text-[#111111] font-medium tracking-wide leading-none">
            Click viewport to navigate &nbsp;|&nbsp; Press{" "}
            <span className="key mx-1 align-middle">
              ESC
            </span>{" "}
            to adjust HUD options
          </span>
        </div>
      )}

      <div data-hud="minimap" className="absolute top-3 left-3 z-40 pointer-events-none">
        <div className="rounded-full">
          <canvas
            ref={minimapCanvasRef}
            width={168}
            height={168}
            className="block h-[168px] w-[168px] rounded-full"
          />
        </div>
      </div>
    </div>
  );
};
