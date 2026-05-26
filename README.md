# Semantic Manifold Vector Visualizer

## Overview

The **Semantic Manifold Vector Visualizer** is an interactive, browser-based 3D computational environment designed for the qualitative analysis and geometric interpretation of high-dimensional semantic vector spaces. By projecting embedding tokens (e.g., word embeddings from linguistic models) into a continuous Three.js-powered Cartesian coordinate system, this instrument enables researchers to evaluate cosine similarities and local neighborhood topology in real-time.

This application is built with modern web technologies (React, Three.js) and adheres to rigorous academic presentation standards, featuring dynamic heuristic rendering and metric export capabilities for downstream analysis.

## Core Capabilities

### 1. 3D Spatial Navigation & Projection
- **Real-Time Rendering:** Utilizes `WebGL` via `Three.js` to render coordinate spaces at 60 FPS, providing fluid interactions.
- **Flight Controls:** Unconstrained traversal of the manifold architecture using WASDRF keybindings paired with pitch/yaw mouse lock.
- **Camera Interpolation:** Implements linear interpolation ($LERP$) and spherical linear interpolation ($SLERP$) for seamless cinematic transitioning between fixed perspectives (Front, Top, Isometric), aiding in intuitive spatial orientation.

### 2. Topological Neighborhood Inspection
- **Vector Reticle Selection:** Aligning the central reticle with any token vector in the space resolves its immediate geometrical context.
- **Cosine Similarity Mapping:** Dynamically queries and renders the nearest topological neighbors based on vector cosine similarity distances.
- **Statistical Export:** Facilitates direct data dumps of active local neighborhoods to `.csv` format. This feature is particularly engineered to streamline methodological documentation for researchers and data scientists.

### 3. Academic-Grade Visual Processing
- **Dynamic Occlusion (Depth Sorting):** Employs real-time raycasting and dynamic Z-index recalculations corresponding to the Euclidean distance from the camera, ensuring tokens closer to the viewport naturally occlude distant elements without DOM overlap conflicts.
- **Mathematical Typography:** High-quality typographical rendering of vector formulations (e.g., $$\vec{v}_{\text{token}}$$) using standard academic font pairings, enforcing scientific visualization norms.
- **Customizable Cartesian Apparatus:** Toggleable dimensional axes (X, Y, Z), underlying grid planes, and dynamic coordinate scales tailored per dataset bounds. Configurable mouse sensitivities and visual properties persist across sessions via local storage.

## Technical Architecture

- **Frontend Framework:** React 18+ leveraging Vite
- **3D Engine:** Three.js
- **Styling Engine:** Tailwind CSS
- **Data Handling:** Zero-dependency client-side architecture with scalable internal metric state mappings.

## Usage Guide

### Development mode

For hot reload on frontend changes, run the app in two terminals:

1. Frontend with Vite: `npm run dev`
2. Backend API with Python: `npm run dev:backend`

If you prefer a single command for both services, use `npm run dev:all`.

1. **Free Movement:** Click anywhere on the main 3D canvas to engage the pointer-lock mechanism. Use your mouse to orient the camera and `W`, `A`, `S`, `D` to translate. Use `R` for ascension and `F` for descension.
2. **Preset Views:** Use the left sidebar to smoothly interpolate to rigid orthographic-like axes (Front, Top, Side) or Isometric projections.
3. **Data Inspection:** Point the central crosshair at any active token label. The right-hand "Manifold Vector Analysis" pane will load the specific semantic identity along with proximity metrics of its immediate subspace.
4. **Metadata Extraction:** With a target localized, click the "Export CSV Dump" button in the right pane to download a comma-separated manifest of the local topological distribution.

## Scientific Motivation

This tooling infrastructure was constructed to provide structural transparency into the traditionally opaque latent spaces of embedding representational models. By bridging multidimensional tensors and interpretable 3-dimensional geometric plots, it serves as a critical interpretability lens for evaluating cluster cohesion, semantic orientation, and linguistic vector alignment in artificial intelligence research.
