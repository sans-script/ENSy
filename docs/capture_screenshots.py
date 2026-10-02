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

from PIL import Image
from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1600, "height": 950}
VIEWPORT_CLIP = {"x": 320, "y": 0, "width": 940, "height": 950}  # between the 320 px and 340 px sidebars
APPROACH_MS = 1000  # forward flight after the isometric preset (14 units/s along the aim)
DEVICE_SCALE = 2  # screenshots at 2x pixel density
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
    def view(self, name: str) -> Path:
        """The 3D viewport only (between the two sidebars)."""
        path = self.out / f"{name}.png"
        self.page.screenshot(path=str(path), clip=VIEWPORT_CLIP)
        print("  ", path.name)
        return path

    def shot(self, name: str, element: str | None = None, max_height: int | None = None) -> Path:
        path = self.out / f"{name}.png"
        if element:
            self.page.locator(element).first.screenshot(path=str(path))
        else:
            self.page.screenshot(path=str(path))
        if max_height:  # drop the empty lower part of tall sidebars (max_height in CSS px)
            img = Image.open(path)
            h = max_height * DEVICE_SCALE
            img.crop((0, 0, img.width, min(img.height, h))).save(path)
        print("  ", path.name)
        return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:3000/")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "screenshots")
    ap.add_argument("--channel", default="msedge", help="browser channel ('' for bundled Chromium)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel or None)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=DEVICE_SCALE)
        t = Tour(page, args.base_url, args.out)

        print("[1] full workspace")
        t.open(modelKey="qwen", projectionType="umap")
        t.preset("Isometric")
        t.pin("market")
        t.shot("fig01_workspace")

        # Figure views: viewport only (sidebars cropped), grid planes and their
        # backgrounds off, HUD hidden, and a closer camera than the preset.
        clean = dict(probeEnabled=False, showBottomHud=False, showGridPlanes=False, showPlaneBackground=False)

        print("[2] the five projections of one model (same camera)")
        t.open(modelKey="gemma2", projectionType="pca", **clean)
        for algo in ALGO_NAMES:
            t.choose("Projection", algo)
            t.wait_for_data()
            t.preset("Isometric")
            t.approach(APPROACH_MS)
            t.view(f"fig02_{algo}")

        print("[3] the same algorithm across the four models")
        t.open(modelKey="qwen", projectionType="umap", **clean)
        for model in MODEL_NAMES:
            t.choose("Model", model)
            t.wait_for_data()
            t.preset("Isometric")
            t.approach(APPROACH_MS)
            t.view(f"fig03_{model}")

        print("[4] thermal camera, K = 10 and K = 20")
        t.open(modelKey="qwen", projectionType="tsne", **clean)
        t.choose("Colour layer", "heat")
        t.preset("Isometric")
        t.approach(APPROACH_MS)
        t.view("fig04_thermal_k10")
        t.choose("Neighbourhood K", "20")
        t.page.wait_for_timeout(800)
        t.view("fig05_thermal_k20")

        print("[5] neighbourhood probing after flying towards a token")
        t.open(modelKey="bert", projectionType="umap", **{**clean, "probeEnabled": True})
        t.preset("Isometric")
        t.approach(APPROACH_MS)
        t.pin("wanted")  # a verb in the visible cluster; the pin holds while the reticle is empty
        t.view("fig06_probing")
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



if __name__ == "__main__":
    main()
