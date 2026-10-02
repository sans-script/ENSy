# TensorFlow Embedding Projector: technical review for ENSy

This document reverse-engineers how the TensorFlow Embedding Projector
(<https://projector.tensorflow.org>) draws points and labels, how its
neighbour slider works and which file formats it reads and writes. It then
proposes how ENSy can adopt the useful parts and exchange data with it.

**Sources reviewed**

- `tensorflow/embedding-projector-standalone` at `6ff71f7233` (2019-05-28). This
  repository holds only the compiled bundle (`index.html`) and the demo data
  (`oss_data/`); it contains no readable source.
- The TypeScript source, in `tensorflow/tensorboard` at `4a09df254f`
  (2026-08-24), directory `tensorboard/plugins/projector/vz_projector/`. All
  file names below refer to this directory.

Constants and formulas are quoted from the source. Statements about runtime
behaviour that were not observed in a browser are marked *(from source)*.

## 1. Summary

| Topic | Embedding Projector | ENSy today | Recommendation |
|---|---|---|---|
| Points | GL point sprites, circle cut in the fragment shader, size ∝ 1/log N, fog for depth | One `THREE.Points` cloud with a disc texture, RGBA vertex colours | Keep; optionally add fog as a depth cue |
| Labels (default mode) | Only the hovered point, the selection and its neighbours are labelled, on one 2D canvas, black text with a 3 px white stroke | Every token labelled; DOM spans, greedy grid | Keep "label everything" (N = 438) but move to one 2D canvas with a stroke halo |
| Labels (3D mode) | Every point drawn as camera-facing WebGL text from a glyph atlas; no dots, no collision avoidance | — | Not needed |
| Neighbour count | `paper-slider`, 5–999, default 100, applied on release | Dropdown K ∈ {5, 10, 15, 20} | Slider K = 1…50; all metrics precomputed (≈ 0.25 s) |
| Numeric colouring | Two-colour linear ramp `#ffffdd → #1f2d86` over [min, max]; ≤ 50 distinct values are coloured as categories | Linear ramp on a fixed [−1, 1] scale | Stepped diverging palette on a fixed ±0.3 scale |
| Formats | Vectors TSV or float32 `.bytes`, metadata TSV, `projector_config.json`, bookmark "state" JSON | JSON API | Export all four; import vectors, metadata and state |

## 2. Rendering architecture

`ScatterPlot` (`scatterPlot.ts`) owns the Three.js scene, the camera
(perspective in 3D, orthographic in 2D) and a list of *visualizers*. The
adapter `ProjectorScatterPlotAdapter` (`projectorScatterPlotAdapter.ts`)
converts application state (hover, selection, neighbours, colour option) into
packed typed arrays and passes them to the visualizers through a
`RenderContext` each frame.

The adapter installs one of two sets of visualizers (`createVisualizers`):

- **Default:** `ScatterPlotVisualizerSprites` (points) and
  `ScatterPlotVisualizerCanvasLabels` (labels on a 2D canvas).
- **3D labels mode** (the "A" button in the toolbar):
  `ScatterPlotVisualizer3DLabels` only, which draws text instead of points.

`ScatterPlotVisualizerPolylines` is added in both modes for sequence data.

### 2.1 Points (`scatterPlotVisualizerSprites.ts`)

- **Geometry:** one `THREE.Points` with attributes `position`, `color` (RGB)
  and `scaleFactor`; there is no per-point alpha.
- **Shape:** the fragment shader discards fragments outside the unit circle
  (`point_in_unit_circle(gl_PointCoord)`), so points are hard-edged discs with
  no texture.
- **Size:** `pointSize = 200 / ln(N) / ln(8)` (divided by 1.5 in 2D).
  - In 3D, `gl_PointSize = −pointSize / z_camera`, so points grow as the camera
    approaches.
  - In 2D, the size follows a bounded arctan function of the zoom.
  - The size never falls below 5 px (`MIN_POINT_SIZE`).
  - Hovered, selected and neighbour points are scaled by 1.2.
- **Depth cue:** a `THREE.Fog` in the background colour, from the nearest
  point's depth to `farthest × (2 − min(N, 5000)/5000)`. Fewer points means
  less fog.
- **Picking:** hover detection does not ray-cast. A second render pass draws
  every point in a unique colour (`new THREE.Color(i)`) into an off-screen
  target, and the pixel under the cursor gives the point index.

### 2.2 Labels, default mode (`scatterPlotVisualizerCanvasLabels.ts`, `label.ts`)

**Which points get a label.**
`generateVisibleLabelRenderParams` builds the label list in priority order: the
hovered point, then the selected points, then the neighbours of the first
selected point. *Unselected points are never labelled in this mode.* This is
why the Projector looks clean: with no selection it shows dots only.

**Drawing.** Labels go on one `<canvas>` laid over the WebGL canvas, cleared
and redrawn every frame:

- **Font.** `10px` Roboto. The hovered point and single selections are drawn
  at 2× (`LABEL_SCALE_LARGE`).
- **Placement.** Each label starts 4 px to the right of its point (`xShift`),
  with a 2 px margin.
- **Contrast.** Each label is drawn twice: `strokeText` with
  `lineWidth = 3` in the stroke colour (white), then `fillText` in the fill
  colour (black). The white stroke is a halo that separates the text from the
  points behind it.
- **Depth.** Opacity is `d3.scalePow().exponent(e)` of the distance to the
  camera, mapped to [0.1, 1]. The hovered point and single selections are
  always opaque.
- **Limit.** At most 10 000 labels per frame.

**Collision avoidance** (`CollisionGrid`, a fork of `amoeba.CollisionGrid`):

- The canvas is divided into a uniform grid of 25 × 50 cells. Each cell keeps
  the bounding boxes of the labels already placed in it.
- A candidate label is tested in two steps:
  1. A cheap test with a box 1 px wide.
  2. If that passes, the text width is measured (`measureText`, the expensive
     call) and the box is inserted.
- A label that overlaps an earlier one is skipped. Earlier labels have
  priority, so hover and selection always win.

### 2.3 Labels, 3D mode (`scatterPlotVisualizer3DLabels.ts`)

- **Glyph atlas.** One 8192 × 80 canvas holds the first 256 character codes,
  in bold Roboto at 60 px, black on white. Two arrays store each glyph's
  offset and width.
- **Geometry.** Each label becomes two triangles per letter in a single
  `BufferGeometry`, with attributes:
  - `posObj`: the letter corner, relative to the label centre;
  - `position`: the point's world position;
  - `uv`: coordinates into the atlas;
  - `color`: the label colour.
- **Billboarding.** The vertex shader rebuilds the camera's right and up
  vectors from `modelViewMatrix` and offsets each corner along them, so the
  text always faces the camera.
- **Colour.** The point's colour multiplied by the atlas texel. On white
  backgrounds the uncoloured labels are white (`LABELS_3D_COLOR_*`).
- **Limitations.**
  - No collision avoidance: dense regions become unreadable.
  - No points are drawn in this mode.
  - Only 256 character codes are covered.

### 2.4 Colours (`projectorScatterPlotAdapter.ts`, `vz-projector-data-panel.ts`)

**Points by state:**

| State | Colour |
|---|---|
| No selection | `#7575d9` |
| Unselected while a selection exists | `#e3e3e3` |
| Selected | `#fa6666` |
| Hovered | `#760b4f` |

**Neighbours** are coloured by distance with `NN_COLOR_SCALE`: domain
`[1, 0.7, 0.4]` of normalised distance, ramping from `hsl(285,80%,40%)` through
`hsl(0,80%,65%)` to `hsl(40,70%,60%)`.

**"Colour by" a metadata column:**

- A column with at most 50 distinct values (`NUM_COLORS_COLOR_MAP`) is
  categorical: `d3.schemeCategory10`, reordered.
- A numeric column with more distinct values gets a two-stop linear ramp,
  `#ffffdd` (min) to `#1f2d86` (max).

## 3. The neighbour slider (`vz-projector-inspector-panel*.ts`)

```html
<paper-slider class="nn-slider" pin min="5" max="999" editable
              value="{{numNN}}" on-change="updateNumNN">
```

- **Range.** Default 100 (`DEFAULT_NEIGHBORS`), editable as a number, with a
  pin that shows the value while dragging.
- **Update timing.** `on-change` fires on release, not while dragging.
  `updateNumNN` then re-notifies the selection, which recomputes the
  neighbours.
- **Neighbour space.** Neighbours are computed in the *original* space, with a
  COSINE | EUCLIDEAN toggle (`vector.cosDist`, `vector.dist`).
- **Display.** The list shows each neighbour with a bar proportional to its
  normalised distance, coloured with `NN_COLOR_SCALE`.

### 3.1 What a slider means in ENSy

In ENSy, K is more than the number of neighbours shown. It also parametrises
the co-ranking metrics and the per-token decomposition $B_{NX,i}(K)$. Today the
server computes these only for `K_CURVE = (5, 10, 15, 20)`, which is why K is a
dropdown.

All quantities for every K follow from the co-ranking matrix that is already
computed. Measured for N = 438 (random data, same code path,
`ensy/coranking.py`):

| Step | Time |
|---|---|
| Ranks + co-ranking matrix Q | 0.14 s |
| $M_T, M_C, Q_{NX}, B_{NX}$ for K = 1…50 | 0.17 s |
| $B_{NX,i}$ for K = 1…50 | 0.07 s |

**Proposal: a continuous slider, K = 1…50.**

- **Server.** Compute the curves and $B_{NX,i}$ for every K in 1…50, plus the
  50 nearest original-space and projected neighbour lists. This adds about
  0.25 s per projection. The payload grows by about 438 × 50 values per array
  (≈ 150–250 kB of JSON).
- **Client.** All K values are then local, so the slider can update on every
  `input` event: the behaviour map, the probe and the metrics panel all react
  while dragging. The Projector, by contrast, recomputes only on release.
- **Paper.** K = 10 stays the reference value and {5, 10, 15, 20} the curves
  shown. The slider is an exploration aid and changes no reported result.
- **Design.** A thin rule with a square thumb, a `K = 10` readout in LaTeX and
  ticks at 5, 10, 15, 20 that snap within ±1. This matches the existing
  sidebar style better than `paper-slider`'s Material look.

## 4. Why the behaviour map is hard to read

`heatColor` interpolates linearly from a pale neutral to cyan (−1) or orange
(+1) over the fixed domain [−1, 1]. The measured distribution of $B_{NX,i}(10)$
does not use that domain:

| Projection | min…max | 5th–95th percentile | $\lvert B_{NX,i}\rvert < 0.1$ |
|---|---|---|---|
| Qwen, t-SNE p = 5 | −0.5 … +0.5 | −0.2 … +0.3 | 37 % |
| Qwen, t-SNE p = 30 | −0.4 … +0.3 | −0.3 … +0.2 | 42 % |
| Qwen, UMAP | −0.4 … +0.7 | −0.2 … +0.3 | 43 % |
| GPT-2, PCA | −0.4 … +0.4 | −0.2 … +0.2 | 35 % |
| BERT, Isomap | −0.4 … +0.3 | −0.2 … +0.2 | 41 % |

A typical value of ±0.1 sits 10 % of the way from neutral to full colour, so
most tokens look almost identical. The values are also discrete: multiples of
1/K, so only 21 levels exist at K = 10.

**Proposal:**

1. **Domain.** Keep the scale fixed, so the same value has the same colour in
   every projection, as the paper requires, but narrow it to ±0.3, clamped. The
   legend states "≤ −0.3" and "≥ +0.3".
2. **Steps.** Use a stepped scale that matches the discreteness: 7 classes,
   centred on 0 with a ±0.05 neutral band.
3. **Palette.** A colour-blind-safe diverging palette with a *mid-grey*
   neutral (e.g. ColorBrewer `RdBu` with `#bababa` at 0), so that neutral
   tokens remain visible on the white background.
4. **Size.** In this layer, enlarge dots in proportion to |value|
   (scale 1.0–1.6), as the Projector does for selection (×1.2). Extreme tokens
   then stand out by size as well as hue.

## 5. File formats

### 5.1 Vectors and metadata (`data-provider.ts`)

- **Vectors TSV** (`parseTensors`): one point per line, tab-separated floats.
  If the first field is not a number, it is taken as the label.
  - All rows must have the same dimension, and the dimension must be at
    least 2.
- **Vectors binary** (`parseTensorsFromFloat32Array`): raw little-endian
  float32, row-major, with the shape given by `tensorShape` in the config.
- **Metadata TSV** (`parseMetadata`):
  - With one column, there is no header.
  - With several columns, the first row is the header.
  - Its row count must match the number of vectors.
- **"Download"** in the data panel exports only the metadata
  (`metadata-edited.tsv`), including labels edited in the UI. It does not
  export vectors or coordinates.

### 5.2 `projector_config.json` and "Publish"

```json
{
  "embeddings": [{
    "tensorName": "Word2Vec 10K",
    "tensorShape": [10000, 200],
    "tensorPath": "oss_data/word2vec_10000_200d_tensors.bytes",
    "metadataPath": "oss_data/word2vec_10000_200d_labels.tsv",
    "bookmarksPath": "oss_data/word2vec_full_bookmarks.txt",
    "sprite": {"imagePath": "...png", "singleImageDim": [28, 28]}
  }],
  "modelCheckpointPath": "Demo datasets"
}
```

"Publish" amounts to hosting this JSON and its files at public URLs (a GitHub
gist or raw files) and opening `https://projector.tensorflow.org/?config=<url>`.

### 5.3 Bookmarks: the `State` JSON (`data.ts`, `vz-projector-bookmark-panel.ts`)

The bookmark panel downloads and uploads a file named `state`: a JSON array of
`State` objects. The fields that matter are:

| Field | Meaning |
|---|---|
| `projections` | one object per point, e.g. `{"umap-0": x, "umap-1": y, "umap-2": z}`; keys are `<pca\|tsne\|umap>-<component>` |
| `selectedProjection` | `"pca"`, `"tsne"`, `"umap"` or `"custom"` |
| `dataSetDimensions` | `[N, d]` |
| `pcaComponentDimensions`, `tSNEis3d`, `umapIs3d`, `tSNEIteration`, `tSNEPerplexity`, `umapNeighbors`, `umapMinDist` | parameters shown in the projection panel |
| `cameraDef`, `selectedPoints`, `filteredPoints`, `selectedColorOptionName`, `selectedLabelOption`, `label` | view state |

**Restoring a bookmark sets coordinates without recomputing them**
(`vz-projector.ts`, `loadState`):

- `loadState` copies `state.projections[i]` into each point.
- `restoreUIFromBookmark` disables the re-projection that UI changes would
  normally trigger.
- For t-SNE, the coordinates are rendered only if `tSNEIteration > 0`
  (`projectionCanBeRendered`).

This is the hook for showing ENSy's exact layouts in the Projector.

## 6. Interoperability design

### 6.1 Export: ENSy → Projector

A new endpoint, `GET /api/export/projector?model=…`, returns a zip containing:

- **`vectors.tsv`:** the N × d rows ENSy measures (L2-normalised), in dataset
  order.
- **`metadata.tsv`:** with a header row and the columns:
  - `word`, `category`, `token_id`, `model`;
  - `b_nx_10_<algorithm>` for each algorithm (and other K values if needed).
- **`state.txt`:** one bookmark per ENSy algorithm, each holding:
  - ENSy's coordinates (from `/api/projection`, so the Projector shows ENSy's
    layout, not its own re-run);
  - `selectedColorOptionName: "category"` and `selectedLabelOption: "word"`;
  - `label: "ENSy — <model> — <algorithm> (M_T(10)=…, B_NX(10)=…)"`.
- **`projector_config.json`:** one embedding entry per model, with
  `bookmarksPath` pointing at the state file. Uploaded to a public URL, this
  gives a shareable Projector link.

**Mapping ENSy algorithms to bookmark keys:**

| ENSy | Bookmark |
|---|---|
| `pca` | `pca-0..2`, `pcaComponentDimensions: [0,1,2]` |
| `umap` | `umap-0..2`, `umapIs3d: true` |
| `tsne`, `tsne_p30` | `tsne-0..2`, `tSNEis3d: true`, `tSNEIteration: 1000`, `tSNEPerplexity: 5 \| 30` |
| `isomap`, `pca_nc` | No slot. Store under `umap-*` and say so in the bookmark `label`. Isomap's giant-component subset needs a `filteredPoints` list. |

**Caveats:**

- **B_NX columns are coloured as categories.** At K = 10, $B_{NX,i}$ has at
  most 21 distinct values, below the 50-value threshold. The Projector
  therefore colours these columns as categories with `schemeCategory10`, not
  as a gradient. ENSy's behaviour map stays the better view of these values.
- **The Projector rescales coordinates.** It normalises the positions into its
  own cube (`SCATTER_PLOT_CUBE_LENGTH = 2`), so absolute scales differ from
  ENSy's. Relative layouts are identical.

### 6.2 Import: Projector → ENSy

The Projector exports little: metadata TSV and the `state` file. ENSy can
accept the files a user already has:

1. **Vectors TSV and metadata TSV.** These form an *external dataset* with no
   Hugging Face token ids.
   - The reference partition comes from a metadata column the user chooses
     (default `category`, else the first categorical column).
   - External datasets are hashed like ENSy's own, but marked `frozen: false`
     and kept apart from the paper's frozen dataset.
2. **A `state` file.** Each bookmark's `projections` become a projection that
   ENSy measures *as given*, without re-running anything. ENSy then reports
   $M_T, M_C, Q_{NX}, B_{NX}$ and the behaviour map for the Projector's own
   t-SNE (`bh_tsne.ts`), UMAP (`umap-js`) or approximate PCA layouts.
   - The Projector's PCA is approximate: it samples ≤ 50 000 points and ≤ 200
     dimensions (`PCA_SAMPLE_SIZE`, `PCA_SAMPLE_DIM`).
   - Its t-SNE and UMAP are separate JavaScript implementations from
     scikit-learn and umap-learn, so their numbers will differ from ENSy's.

Import makes ENSy a quality auditor for layouts produced in the Projector,
which is the question the paper argues every projection should be asked.

### 6.3 What not to copy

- **3D labels mode:** it has no collision avoidance and drops the points.
- **The 2-colour numeric ramp:** a signed quantity needs a diverging palette.
- **Recompute-on-release:** with everything precomputed, ENSy can update live.

## 7. Implementation plan

| # | Change | Where | Size |
|---|---|---|---|
| 1 | Behaviour-map palette: fixed ±0.3, 7-step diverging, mid-grey neutral, |value|-scaled dots; legend | `SpaceVisualizer.tsx` (`heatColor`), `MetricsPanel.tsx` | S |
| 2 | Labels on one 2D canvas with a 3 px white stroke; priority aimed > probed > nearest; keep the occupancy grid | `SpaceVisualizer.tsx` | M |
| 3 | K = 1…50 precomputed on the server; slider with snapping ticks in the sidebar | `app.py`, `ensy/coranking.py`, `SidebarLeft.tsx`, `App.tsx` | M |
| 4 | Export zip (vectors, metadata, state, config) and an "Export to Projector" button | `app.py`, `SidebarRight.tsx` | M |
| 5 | Import of vectors and metadata TSV as an external dataset, and of `state` bookmarks as measured projections | `app.py`, `ensy/dataset.py`, new UI dialog | L |
| 6 | Optional: fog depth cue (near = nearest point, far ∝ point count) | `SpaceVisualizer.tsx` | S |

Items 1–3 change the viewer only. Items 4–5 add an API surface that should get
tests, a round trip ENSy → Projector state → ENSy that reproduces the same
coordinates and metrics, before it is documented in the paper.
