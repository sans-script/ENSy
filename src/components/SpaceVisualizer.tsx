import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { SemanticNode, SemanticSpace } from "../types";
import { semanticSpaceData } from "../data/semanticSpace";

interface SpaceVisualizerProps {
  projectionType: "umap" | "tsne" | "pca" | "isomap";
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
  const tokenPointsRef = useRef<THREE.Points[]>([]);
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
  const labelElementsRef = useRef<
    {
      dom: HTMLDivElement;
      position: THREE.Vector3;
      type: "token" | "latex";
      userData?: SemanticNode;
      axisId?: "x" | "y" | "z";
    }[]
  >([]);
  const gridNumberElementsRef = useRef<
    { dom: HTMLDivElement; position: THREE.Vector3 }[]
  >([]);

  const currentStepResolutionRef = useRef<number>(6);
  const currentFilterRef = useRef(currentFilter);
  const currentModeRef = useRef(cameraMode);
  const currentFovRef = useRef(fov);
  const fovFadeProgressRef = useRef(1);
  const projectionTypeRef = useRef(projectionType);

  const showXAxisRef = useRef(showXAxis);
  const showYAxisRef = useRef(showYAxis);
  const showZAxisRef = useRef(showZAxis);
  const showGridPlanesRef = useRef(showGridPlanes);
  const showPlaneBackgroundRef = useRef(showPlaneBackground);
  const mouseSensitivityRef = useRef(mouseSensitivity);

  const SIZE = 60; // CONSTANT SIZE OF THE COORDINATE BOX

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
    filterNodesPrivate();
    rebuildGridPrivate();
  }, [currentFilter]);

  useEffect(() => {
    projectionTypeRef.current = projectionType;
    // When projection type changes, we trigger position recalculation
    filterNodesPrivate();
  }, [projectionType]);

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
      if (document.pointerLockElement !== container) return;

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

    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== container) return;
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
          persp.position.copy(ortho.position);
          persp.rotation.copy(ortho.rotation);
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

        // Smoothly interpolate token positions dynamically for dimensionality reduction comparison
        tokenPointsRef.current.forEach((pts) => {
          if (pts.userData && pts.userData.targetPosition) {
            pts.position.lerp(pts.userData.targetPosition, 0.04);
            const matchingLabel = labelElementsRef.current.find(
              (l) => l.userData && l.userData === pts.userData,
            );
            if (matchingLabel) {
              matchingLabel.position.copy(pts.position);
            }
          }
        });

        // Target nearest nodes inside screen coordinate projections (Crosshair UX)
        runNodeSelectionInspector();

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

        // Speed check FPS counter
        frameCount++;
        const currTime = performance.now();
        if (currTime - lastTime >= 1000) {
          onFPSUpdate(Math.round((frameCount * 1000) / (currTime - lastTime)));
          frameCount = 0;
          lastTime = currTime;
        }

        renderer.render(scene, activeCamera);
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
  }, [data]);

  const handleContainerClick = () => {
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

    activeCamera.getWorldDirection(dir);
    dir.y = 0;
    if (dir.lengthSq() < 0.0001) {
      dir.set(0, 0, -1);
    } else {
      dir.normalize();
    }

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

  const createTokenSystems = () => {
    const container = labelsContainerRef.current;
    if (!container || !sceneRef.current) return;

    let mn = [1e9, 1e9, 1e9];
    let mx = [-1e9, -1e9, -1e9];

    const nodes = data?.nodes || semanticSpaceData.nodes;

    nodes.forEach((n) => {
      mn[0] = Math.min(mn[0], n.coords.x3d);
      mx[0] = Math.max(mx[0], n.coords.x3d);
      mn[1] = Math.min(mn[1], n.coords.y3d);
      mx[1] = Math.max(mx[1], n.coords.y3d);
      mn[2] = Math.min(mn[2], n.coords.z3d);
      mx[2] = Math.max(mx[2], n.coords.z3d);
    });

    const range = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 0.001);
    const S = 26; // Node projection scale factor

    tokenPointsRef.current = [];

    nodes.forEach((node) => {
      const pos = new THREE.Vector3(
        ((node.coords.x3d - (mn[0] + mx[0]) / 2) / range) * S,
        ((node.coords.y3d - (mn[1] + mx[1]) / 2) / range) * S,
        ((node.coords.z3d - (mn[2] + mx[2]) / 2) / range) * S,
      );

      // Create a invisible helper vector point for ThreeJS raycasting
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute([0, 0, 0], 3),
      );
      const pts = new THREE.Points(
        geo,
        new THREE.PointsMaterial({ size: 2.5, transparent: true, opacity: 0 }),
      );
      pts.position.copy(pos);
      pts.userData = node;
      sceneRef.current?.add(pts);
      tokenPointsRef.current.push(pts);

      // Generate HTML label elements overlaid on WebGL Canvas (EB Garamond)
      const dom = document.createElement("div");
      dom.className =
        "spatial-label absolute select-none pointer-events-none transition-all duration-75 text-xs font-serif font-normal italic tracking-wide";

      // Academic specific aesthetic: blues for Technology, greens for Biological, deep crimson for Actions
      if (node.color === "#378ADD") {
        dom.style.color = "#1e40af"; // Navy slate
      } else if (node.color === "#1D9E75") {
        dom.style.color = "#065f46"; // Dark emerald
      } else if (node.color === "#D85A30") {
        dom.style.color = "#b91c1c"; // Crimson red
      } else {
        dom.style.color = "#86198f"; // Pink magenta mutant
      }
      dom.textContent = node.label;
      container.appendChild(dom);

      labelElementsRef.current.push({
        dom,
        position: pos,
        userData: node,
        type: "token",
      });
    });
  };

  // Node filtering logic matching target quadrants
  const getProjectedCoords = (n: SemanticNode, type: string) => {
    const { x3d: x, y3d: y, z3d: z } = n.coords;
    if (type === "pca") {
      return { x3d: x * 0.8 + y * 0.6, y3d: -x * 0.6 + y * 0.8, z3d: z * 1.2 };
    }
    if (type === "tsne") {
      return {
        x3d: Math.sign(x) * Math.abs(x) ** 0.8 * 1.5,
        y3d: Math.sign(y) * Math.abs(y) ** 0.8 * 1.5,
        z3d: Math.sign(z) * Math.abs(z) ** 0.8 * 1.5,
      };
    }
    if (type === "isomap") {
      const r = Math.sqrt(x * x + y * y);
      const theta = Math.atan2(y, x);
      return {
        x3d: r * Math.cos(theta + r * 0.1),
        y3d: r * Math.sin(theta + r * 0.1),
        z3d: z,
      };
    }
    return { x3d: x, y3d: y, z3d: z };
  };

  const filterNodesPrivate = () => {
    const filter = currentFilterRef.current;
    const projType = projectionTypeRef.current;

    let mn = [1e9, 1e9, 1e9];
    let mx = [-1e9, -1e9, -1e9];
    const nodes = data?.nodes || semanticSpaceData.nodes;

    // First pass to get bounds based on projected coordinates
    nodes.forEach((n) => {
      const pc = getProjectedCoords(n, projType);
      mn[0] = Math.min(mn[0], pc.x3d);
      mx[0] = Math.max(mx[0], pc.x3d);
      mn[1] = Math.min(mn[1], pc.y3d);
      mx[1] = Math.max(mx[1], pc.y3d);
      mn[2] = Math.min(mn[2], pc.z3d);
      mx[2] = Math.max(mx[2], pc.z3d);
    });
    const range = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 0.001);
    const S = 26;

    tokenPointsRef.current.forEach((pts) => {
      const node = pts.userData as SemanticNode;
      const matchingLabel = labelElementsRef.current.find(
        (l) => l.userData && l.userData === node,
      );

      const pc = getProjectedCoords(node, projType);
      const x = pc.x3d;
      const y = pc.y3d;
      const z = pc.z3d;

      const vx = ((x - (mn[0] + mx[0]) / 2) / range) * S;
      const vy = ((y - (mn[1] + mx[1]) / 2) / range) * S;
      const vz = ((z - (mn[2] + mx[2]) / 2) / range) * S;

      let shouldShow = true;

      if (filter === "positive") {
        if (vx < 0 || vy < 0 || vz < 0) shouldShow = false;
      } else if (filter === "negative") {
        if (vx > 0 || vy > 0 || vz > 0) shouldShow = false;
      }

      pts.visible = shouldShow;
      if (matchingLabel) {
        matchingLabel.dom.style.display = shouldShow ? "block" : "none";
      }

      const targetPos = new THREE.Vector3(vx, vy, vz);

      if (!pts.userData.targetPosition) {
        pts.userData.targetPosition = targetPos.clone();
      } else {
        pts.userData.targetPosition.copy(targetPos);
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

    if (bestNode) {
      const bn = bestNode as SemanticNode;
      const pc = getProjectedCoords(bn, projectionTypeRef.current);
      const projectedNode: SemanticNode = {
        id: bn.id,
        token_id: bn.token_id,
        label: bn.label,
        cluster: bn.cluster,
        color: bn.color,
        coords: { ...bn.coords, ...pc },
        top5_neighbors: bn.top5_neighbors,
      };
      onNodeHovered(projectedNode);
    } else {
      onNodeHovered(null);
    }

    // Apply underlines and visual highlights to active DOM target texts
    labelElementsRef.current.forEach((tbl) => {
      if (tbl.userData) {
        if (bestNode && tbl.userData.label === bestNode.label) {
          tbl.dom.classList.add("highlighted-acad");
        } else {
          tbl.dom.classList.remove("highlighted-acad");
        }
      }
    });
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

    // Projected labels
    labelElementsRef.current.forEach(({ dom, position, type, axisId }) => {
      // Check axis visibility
      if (axisId === "x" && !showXAxisRef.current) {
        dom.style.opacity = "0";
        return;
      }
      if (axisId === "y" && !showYAxisRef.current) {
        dom.style.opacity = "0";
        return;
      }
      if (axisId === "z" && !showZAxisRef.current) {
        dom.style.opacity = "0";
        return;
      }

      if (dom.style.display === "none") return;

      if (dom.textContent === "O") {
        const factor = Math.max(0.4, Math.min(4.0, activeCamera.position.length() / 30.0));
        tempV.copy(position).multiplyScalar(factor);
      } else {
        tempV.copy(position);
      }
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

      const fadeThreshold = type === "token" ? 35 : 55;
      let fade = Math.max(0, Math.min(1, 1 - (dist - fadeThreshold) / 90));
      let scale = Math.max(0.3, 1 - dist / 120);

      if (dom.textContent === "O") {
        // Origin label "O" scale grows with distance to remain a visible anchor, capped at 1.3
        scale = Math.min(1.3, 0.7 + dist * 0.012);
        // Ensure "O" doesn't completely fade away easily, remaining a clean anchor
        fade = Math.max(0.6, 1 - dist / 180);
      }

      if (isOccluded) {
        fade *= 0.15; // significantly reduce opacity if behind a background plane
      }

      dom.style.left = `${x}px`;
      dom.style.top = `${y}px`;
      dom.style.transform = `translate(-50%, -50%) scale(${scale})`;
      dom.style.opacity = fade.toFixed(2);
      dom.style.zIndex = Math.max(1, Math.floor(1000 - dist)).toString();
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

    pointsOnMap.forEach((point, index) => {
      if (!point.visible) return;

      ctx.beginPath();
      ctx.fillStyle = point.node.color || "#888780";
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
    >
      {/* Target Reticle Crosshair HUD */}
      <div
        className="absolute top-1/2 left-1/2 w-[12px] h-[12px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-black/30 pointer-events-none z-20
          before:content-[''] before:absolute before:bg-black/25 before:w-[1px] before:h-[6px] before:left-1/2 before:-top-[8px] before:-translate-x-1/2
          after:content-[''] after:absolute after:bg-black/25 after:w-[6px] after:h-[1px] after:top-1/2 after:-left-[8px] after:-translate-y-1/2"
      />

      {/* Absolutely overlayed CSS layers targeting projected nodes coordinates */}
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
        <div className="absolute top-[20px] left-1/2 -translate-x-1/2 bg-white/98 border border-[#111111] px-5 py-[10px] z-50 shadow-[3px_3px_0px_rgba(0,0,0,1)] font-sans pointer-events-none transition-all duration-200 rounded-[4px] flex items-center justify-center">
          <span className="font-sans text-[12px] text-[#111111] font-medium tracking-wide">
            Click viewport to navigate &nbsp;|&nbsp; Press{" "}
            <span className="inline-block relative bg-white border border-[#111111] rounded-[3px] py-0.5 px-1.5 text-[9px] font-bold font-mono text-center min-w-[25px] shadow-[0_1.5px_0_0_#111111] mx-1 top-[-1px] leading-none text-black">
              ESC
            </span>{" "}
            to adjust HUD options
          </span>
        </div>
      )}

      <div className="absolute top-3 left-3 z-40 pointer-events-none">
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
