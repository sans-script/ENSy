"""
tests/screenshots.py — Automated 1920x1080 UI screenshots for the SMVV thesis documentation.

Captures high-quality, full-HD figures in Isometric viewing alignment with an active
token inspected so both the 3D manifold and the right-hand analytical sidebar are populated.

Usage:
    .venv-tests/Scripts/pytest tests/screenshots.py -v
"""

import os
import pytest
from playwright.sync_api import Page


DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
SCREENSHOTS_DIR = os.environ.get("ENSY_SCREENSHOTS_DIR", os.path.join(DOCS_DIR, "screenshots"))


def save_screenshot(page: Page, name: str) -> str:
    """Save a 1920x1080 screenshot to screenshots/ and return its path."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    path = os.path.join(SCREENSHOTS_DIR, f"{name}.png")
    page.screenshot(path=path, full_page=False)
    print(f"\n  [screenshot] {path}")
    return path


def setup_isometric_with_selected_token(page: Page, base_url: str, projection: str = "umap") -> None:
    """
    Sets the viewport to 1920x1080, applies the Isometric preset, selects the projection engine,
    and moves the reticle/camera slightly so a node is actively selected in the right sidebar.
    """
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.goto(base_url, wait_until="networkidle")
    page.wait_for_selector("canvas", timeout=15_000)
    page.wait_for_selector("aside", timeout=10_000)

    # Set projection engine
    page.click("button[aria-label='Projection']")
    page.click(f"[role='listbox'] [data-value='{projection}']")
    page.wait_for_selector("text=Co-ranking", timeout=120_000)

    # Apply Isometric preset
    page.locator("button:has-text('Isometric')").first.click()
    page.wait_for_timeout(2000)

    # Programmatically select a token (e.g., 'document') and typeset MathJax
    page.evaluate("""
        () => {
            if (typeof window.__selectToken === 'function') {
                window.__selectToken('market');
            }
            if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
                window.MathJax.typesetPromise();
            }
        }
    """)
    page.wait_for_timeout(2000)


# ──────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────

class TestScreenshots:
    """Screenshot suite for thesis documentation figures."""

    def test_01_isometric_full_workspace(self, page: Page, base_url: str) -> None:
        """
        Figure 1: Full SMVV workspace in Isometric alignment.
        Shows left controls, 3D manifold canvas with reticle selection, and populated right inspector.
        """
        setup_isometric_with_selected_token(page, base_url, "umap")
        save_screenshot(page, "01_isometric_full_workspace")

    def test_02_projection_umap_isometric(self, page: Page, base_url: str) -> None:
        """Figure 2: UMAP manifold projection in isometric perspective."""
        setup_isometric_with_selected_token(page, base_url, "umap")
        save_screenshot(page, "02_projection_umap_isometric")

    def test_03_projection_pca_isometric(self, page: Page, base_url: str) -> None:
        """Figure 3: PCA linear projection in isometric perspective."""
        setup_isometric_with_selected_token(page, base_url, "pca")
        save_screenshot(page, "03_projection_pca_isometric")

    def test_04_projection_tsne_isometric(self, page: Page, base_url: str) -> None:
        """Figure 4: t-SNE manifold projection in isometric perspective."""
        setup_isometric_with_selected_token(page, base_url, "tsne")
        save_screenshot(page, "04_projection_tsne_isometric")

    def test_05_projection_isomap_isometric(self, page: Page, base_url: str) -> None:
        """Figure 5: Isomap geodesic manifold projection in isometric perspective."""
        setup_isometric_with_selected_token(page, base_url, "isomap")
        save_screenshot(page, "05_projection_isomap_isometric")

    def test_06_sidebar_inspector_detail(self, page: Page, base_url: str) -> None:
        """Figure 6: Detail of the Right Sidebar Vector Inspector with MathJax equation and neighbors."""
        setup_isometric_with_selected_token(page, base_url, "umap")
        aside_right = page.locator("aside").nth(1)
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        aside_right.screenshot(path=os.path.join(SCREENSHOTS_DIR, "06_sidebar_inspector_detail.png"))
        print(f"\n  [screenshot] screenshots/06_sidebar_inspector_detail.png")

    def test_07_sidebar_controls_detail(self, page: Page, base_url: str) -> None:
        """Figure 7: Detail of the Left Sidebar Control Dashboard."""
        setup_isometric_with_selected_token(page, base_url, "umap")
        aside_left = page.locator("aside").first
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        aside_left.screenshot(path=os.path.join(SCREENSHOTS_DIR, "07_sidebar_controls_detail.png"))
        print(f"\n  [screenshot] screenshots/07_sidebar_controls_detail.png")

    def test_08_navigation_hud_detail(self, page: Page, base_url: str) -> None:
        """Figure 8: Detail of the flight navigation keyboard indicator HUD."""
        setup_isometric_with_selected_token(page, base_url, "umap")
        hud = page.locator("div.absolute.bottom-\\[16px\\]").first
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
        if hud.count():
            hud.screenshot(path=os.path.join(SCREENSHOTS_DIR, "08_navigation_hud_detail.png"))
            print(f"\n  [screenshot] screenshots/08_navigation_hud_detail.png")
        else:
            save_screenshot(page, "08_navigation_hud_detail")
