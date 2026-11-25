from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import sys
import threading
from playwright.sync_api import sync_playwright, expect

root = Path(__file__).resolve().parents[2]
parser = ArgumentParser(description="Exercise the DCR Reform package against a loopback tosu fixture")
parser.add_argument("--release", type=Path, default=root / "data/reform/mma_release")
parser.add_argument("--browser", help="Path to a Chromium binary, or use the Playwright installation")
options = parser.parse_args()
release = options.release.resolve()
plugin = "DCR Reform for tosu"
chart = (root / "data/reform/inference_osu/4961239.osu").read_bytes()

class Host(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(release), **kwargs)
    def do_GET(self):
        if self.path.startswith("/files/beatmap/file"):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(chart)
        else:
            super().do_GET()
    def log_message(self,*args): pass

host = ThreadingHTTPServer(("127.0.0.1",24050), Host)
threading.Thread(target=host.serve_forever, daemon=True).start()
args = [sys.executable, "deploy/mma/cpu_service.py", "--threads", "4",
        "--manifest", str(release/plugin/"reform-data/manifest.json")]
base = "http://127.0.0.1:24050/DCR%20Reform%20for%20tosu/"
with TemporaryDirectory(prefix="dcr-chromium-") as profile, subprocess.Popen(args, cwd=root) as cpu, sync_playwright() as p:
    try:
        calls = []
        def route(r):
            if ":24063/features" in r.request.url: calls.append(r.request.url)
            if r.request.url.startswith(("http://127.0.0.1:24050", "http://127.0.0.1:24063")):
                r.continue_()
            else: r.abort()
        def open_browser():
            context = p.chromium.launch_persistent_context(profile, executable_path=options.browser,
                headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context.route("**/*", route)
            context.route_web_socket(re.compile(".*"), lambda ws: None)
            page = context.pages[0]
            page.on("pageerror", lambda error: print("Page error:", error, flush=True))
            page.goto(base)
            page.wait_for_selector("#reform-rating", state="attached")
            return context, page

        context, page = open_browser()
        def estimate(method, rate, identity="id:4961239", client="stable"):
            page.evaluate("""async ([method,rate,identity,client]) => {
              const {state} = await import('./js/app/appContext.js');
              const {applySettingsPayload} = await import('./js/app/settings.js');
              const {fetchBeatmapFile} = await import('./js/app/analysis.js');
              const {setRecomputeHandler} = await import('./js/app/scheduler.js');
              setRecomputeHandler(null);
              applySettingsPayload({estimatorAlgorithm:method});
              state.wsEndpoint = '127.0.0.1:24050';
              state.client = client;
              state.speedRate = rate;
              state.analysisRate = null;
              state.modSignature = String(rate);
              state.modCodes = rate === 1 ? [] : ['DT'];
              state.lastBeatmapIdentity = identity;
              state.activeSource = 'osu';
              setRecomputeHandler(fetchBeatmapFile);
              await fetchBeatmapFile('Reform integration check');
            }""",[method,rate,identity,client])
        value = page.locator("#reform-value")
        calc = page.locator("#reform-calculate")
        for method in ["Sunny","DCR","CM3P_ext"]:
            estimate(method,1,client="lazer" if method == "CM3P_ext" else "stable")
            expect(value).to_have_text(re.compile(r"Reform Dan.*"+method+r".*\d+\.\d+"))
            print("bundled",value.text_content(),flush=True)
        assert not calls, calls
        estimate("DCR",1.5)
        expect(calc).to_be_visible()
        calc.click()
        expect(value).to_have_text(re.compile(r"Reform Dan.*DCR.*\d+\.\d+"),timeout=130000)
        print("computed",value.text_content(),flush=True)
        assert len([u for u in calls if "method=DCR" in u])==1 and not any("CM3P_ext" in u for u in calls),calls
        count=len(calls)
        context.close()
        context, page = open_browser()
        value = page.locator("#reform-value")
        calc = page.locator("#reform-calculate")
        estimate("DCR",1.5)
        expect(calc).to_be_hidden()
        expect(value).to_have_text(re.compile(r"Reform Dan.*DCR.*\d+\.\d+"))
        assert len(calls)==count
        print("persisted",value.text_content(),flush=True)
        estimate("CM3P_ext",1.5)
        expect(calc).to_be_visible()
        calc.click()
        expect(value).to_have_text(re.compile(r"Reform Dan.*CM3P_ext.*\d+\.\d+"),timeout=175000)
        print("computed",value.text_content(),flush=True)
        estimate("DCR",1,"id:999999999")
        expect(calc).to_be_visible()
        chart += b"\n// updated exact bytes\n"
        estimate("DCR",1)
        expect(calc).to_be_visible()
        print("missing ID, MD5 mismatch and rate correctly reached compute state",flush=True)
        context.close()
    finally:
        cpu.terminate()
        host.shutdown()
