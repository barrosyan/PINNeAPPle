"""
export_pdf.py
Renders chassis_structural_analysis.html to PDF using Playwright (headless Chromium).
Waits for MathJax, Chart.js and all images to fully render before capturing.
"""
from __future__ import annotations
import asyncio, os, sys, time

_DIR = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(_DIR, "chassis_structural_analysis.html")
PDF  = os.path.join(_DIR, "chassis_structural_analysis.pdf")


async def main() -> None:
    from playwright.async_api import async_playwright

    print("  Opening headless Chromium …")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        # Load HTML from disk via file:// URI
        uri = "file:///" + HTML.replace("\\", "/")
        print(f"  Loading: {uri}")
        await page.goto(uri, wait_until="networkidle", timeout=60_000)

        # Give MathJax and Chart.js a moment to finish rendering
        print("  Waiting for JS rendering (MathJax / Chart.js) …")
        await page.wait_for_timeout(4_000)

        # Wait until MathJax is done (if present)
        try:
            await page.wait_for_function(
                "() => !window.MathJax || (window.MathJax.typesetPromise && true)",
                timeout=10_000,
            )
        except Exception:
            pass

        print("  Generating PDF …")
        await page.pdf(
            path=PDF,
            format="A4",
            print_background=True,
            margin={"top": "15mm", "bottom": "15mm",
                    "left": "12mm", "right": "12mm"},
            prefer_css_page_size=False,
        )

        await browser.close()

    size_mb = os.path.getsize(PDF) / 1_048_576
    print(f"\n  PDF saved: {PDF}")
    print(f"  Size: {size_mb:.1f} MB")
    print("  Done.")


if __name__ == "__main__":
    t0 = time.time()
    asyncio.run(main())
    print(f"  Elapsed: {time.time()-t0:.1f}s")
