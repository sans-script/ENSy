#!/usr/bin/env python3
"""Scripted tour of ENSy that captures the figures used in DOCS.md.

The tour drives the real UI like a user would: it opens the custom dropdowns,
presses the camera-preset buttons, engages pointer lock by clicking the
viewport and flies with W/A/S/D/R/F and the mouse. The only programmatic hook
is ``window.__selectToken`` (pins the inspected token so a figure always shows
the same word).

Requires the backend (``python app.py``, with HF_TOKEN for the Gemma models)
serving a fresh ``vite build`` on http://127.0.0.1:3000.

Run:  python docs/capture_screenshots.py [--base-url URL] [--out docs/screenshots]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1600, "height": 950}
BASE_SETTINGS = {
    "projectionType": "umap", "modelKey": "qwen", "colorLayer": "category",
    "probeEnabled": True, "currentFilter": "all", "metricK": 10,
    "showBottomHud": True, "cameraMode": "perspective", "fov": 30,
}
MODEL_NAMES = {"qwen": "Qwen 2.5 0.5B-Instruct", "gemma": "Gemma 2B (v1)",
               "gemma2": "Gemma 2 2B", "bert": "BERT base uncased"}
ALGO_NAMES = {"pca": "PCA (centered)", "tsne": "t-SNE (perplexity 5)",
              "tsne_p30": "t-SNE (perplexity 30)", "umap": "UMAP", "isomap": "Isomap"}


class Tour:
    def __init__(self, page: Page, base_url: str, out: Path):
        self.page, self.base_url, self.out = page, base_url, out

    # ── state -------------------------------------------------------------
    def open(self, **settings):
        """Fresh page with the given persisted settings (what a returning user would see).

        The settings are written from a same-origin page that does not run the app;
        otherwise the running app (which saves its state every second) could
        overwrite them before the reload.
        """
        self.page.goto(self.base_url.rstrip("/") + "/api/health")
        merged = json.dumps({**BASE_SETTINGS, **settings})
        self.page.evaluate(f"localStorage.setItem('visualizerSettings', JSON.stringify({merged}))")
        self.page.goto(self.base_url)
        self.wait_for_data()

    def wait_for_data(self):
        self.page.wait_for_selector("text=Co-ranking", timeout=300_000)
        self.page.wait_for_selector("text=Computing projection", state="detached", timeout=300_000)
        self.page.wait_for_timeout(2500)  # token positions animate into place

    # ── user actions --------------------------------------------------------
    def choose(self, dropdown: str, value: str):
        self.page.click(f"button[aria-label='{dropdown}']")
        self.page.wait_for_timeout(250)
        self.page.click(f"[role='listbox'] [data-value='{value}']")
        self.page.wait_for_timeout(250)

    def preset(self, name: str):
        self.page.get_by_role("button", name=name).click()
        self.page.wait_for_timeout(2600)  # LERP/SLERP transition
        self.page.locator("aside").first.evaluate("el => el.scrollTop = 0")  # the click scrolled it

    def pin(self, word: str):
        self.page.evaluate(f"window.__selectToken({json.dumps(word)})")
        self.page.wait_for_timeout(800)

    def engage_flight(self):
        self.page.mouse.click(VIEWPORT["width"] // 2, VIEWPORT["height"] // 2)
        self.page.wait_for_timeout(300)

    def fly(self, key: str, ms: int):
        self.page.keyboard.down(key)
        self.page.wait_for_timeout(ms)
        self.page.keyboard.up(key)
        self.page.wait_for_timeout(200)

    def look(self, dx: int, dy: int, steps: int = 12):
        x, y = VIEWPORT["width"] // 2, VIEWPORT["height"] // 2
        self.page.mouse.move(x, y)
        self.page.mouse.move(x + dx, y + dy, steps=steps)
        self.page.wait_for_timeout(300)

    def approach(self, ms: int = 650):
        """Fly straight ahead from the current preset (same distance for every tile)."""
        self.engage_flight()
        self.fly("KeyW", ms)
        self.release()

    def release(self):
        # A synthetic Escape does not leave pointer lock in automation; exitPointerLock()
        # is what the browser itself calls when a user presses Esc.
        self.page.keyboard.press("Escape")
        self.page.evaluate("document.exitPointerLock()")
        self.page.wait_for_timeout(400)

    # ── capture -----------------------------------------------------------
    def shot(self, name: str, element: str | None = None, max_height: int | None = None) -> Path:
        path = self.out / f"{name}.png"
        if element:
            self.page.locator(element).first.screenshot(path=str(path))
        else:
            self.page.screenshot(path=str(path))
        if max_height:  # drop the empty lower part of tall sidebars
            img = Image.open(path)
            img.crop((0, 0, img.width, min(img.height, max_height))).save(path)
        print("  ", path.name)
        return path


def grid(paths: list[Path], labels: list[str], out: Path, cols: int = 2, crop=(320, 0, 1260, 950)):
    """Compose viewport crops into a labelled grid figure."""
    tiles = [Image.open(p).crop(crop) for p in paths]
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    try:
        font = ImageFont.truetype("times.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(tile, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x, y, x + w - 1, y + h - 1], outline="#111111", width=2)
        d.rectangle([x + 12, y + h - 58, x + 14 + d.textlength(label, font=font) + 16, y + h - 14],
                    fill="white", outline="#111111", width=2)
        d.text((x + 22, y + h - 52), label, fill="#111111", font=font)
    sheet.save(out)
    print("  ", out.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:3000/")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "screenshots")
    ap.add_argument("--channel", default="msedge", help="browser channel ('' for bundled Chromium)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "_tiles"
    tmp.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel or None)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        t = Tour(page, args.base_url, args.out)

        print("[1] full workspace")
        t.open(modelKey="qwen", projectionType="umap")
        t.preset("Isometric")
        t.pin("market")
        t.shot("fig01_workspace")

        print("[2] the five projections of one model (same camera)")
        tiles = []
        t.open(modelKey="gemma2", projectionType="pca", probeEnabled=False, showBottomHud=False, showPlaneBackground=False)
        for algo in ALGO_NAMES:
            t.choose("Projection", algo)
            t.wait_for_data()
            t.preset("Isometric")
            t.approach()
            tiles.append(t.shot(f"_tiles/proj_{algo}"))
        grid(tiles, [ALGO_NAMES[a] for a in ALGO_NAMES], args.out / "fig02_projections_gemma2.png", cols=3)

        print("[3] the same algorithm across the four models")
        tiles = []
        t.open(modelKey="qwen", projectionType="umap", probeEnabled=False, showBottomHud=False, showPlaneBackground=False)
        for model in MODEL_NAMES:
            t.choose("Model", model)
            t.wait_for_data()
            t.preset("Isometric")
            t.approach()
            tiles.append(t.shot(f"_tiles/model_{model}"))
        grid(tiles, [MODEL_NAMES[m] for m in MODEL_NAMES], args.out / "fig03_models_umap.png")

        print("[4] thermal camera, K = 10 and K = 20")
        t.open(modelKey="qwen", projectionType="tsne", probeEnabled=False, showPlaneBackground=False)
        t.choose("Colour layer", "heat")
        t.preset("Isometric")
        t.approach()
        t.shot("fig04_thermal_k10")
        t.choose("Neighbourhood K", "20")
        t.page.wait_for_timeout(800)
        t.shot("fig05_thermal_k20")

        print("[5] neighbourhood probing after flying towards a token")
        t.open(modelKey="bert", projectionType="umap", showPlaneBackground=False)
        t.preset("Isometric")
        t.approach(800)
        t.pin("wanted")  # a verb in the visible cluster; the pin holds while the reticle is empty
        t.shot("fig06_probing")
        t.shot("fig07_inspector", "aside >> nth=1", max_height=600)

        print("[6] interface details")
        t.open(modelKey="gemma", projectionType="isomap")
        t.preset("Isometric")
        t.shot("fig08_metrics_panel", "div.absolute.bottom-\\[16px\\].right-\\[16px\\]")
        t.shot("fig09_controls", "aside >> nth=0", max_height=700)
        t.page.click("button[aria-label='Projection']")
        t.page.wait_for_timeout(400)
        t.page.screenshot(path=str(args.out / "fig10_dropdown.png"),
                          clip={"x": 0, "y": 440, "width": 320, "height": 260})
        print("   fig10_dropdown.png")
        t.page.keyboard.press("Escape")

        print("[7] camera modes")
        t.open(modelKey="qwen", projectionType="pca")
        for btn, name in [("Front (XY)", "fig11_front"), ("Top (XZ)", "fig12_top")]:
            t.preset(btn)
            t.shot(name)

        browser.close()

    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
