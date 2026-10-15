import asyncio, sys
from playwright.async_api import async_playwright
CASES = [("vector","https://duckdb-mcp.nrp-nautilus.io/tiles/hex/9cdc5393a12340fd/{z}/{x}/{y}.pbf",6),
         ("vector","https://duckdb-mcp.nrp-nautilus.io/tiles/hex/9cdc5393a12340fd/{z}/{x}/{y}.pbf",9),
         ("geojson","http://localhost:8765/hex6.geojson",6),("geojson","http://localhost:8765/hex7.geojson",6),
         ("geojson","http://localhost:8765/hex8.geojson",6),("range","http://localhost:8765/range.geojson",6)]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=swiftshader","--enable-webgl","--ignore-gpu-blocklist"])
        for kind, src, z in CASES:
            ts = []
            for rep in range(3):
                pg = await b.new_page(viewport={"width":1200,"height":800})
                await pg.goto(f"http://localhost:8765/bench.html?kind={kind}&z={z}&src={src}")
                await pg.wait_for_function("window.RESULT !== undefined", timeout=180000)
                ts.append(await pg.evaluate("window.RESULT")); await pg.close()
            print(f"{kind:8s} z{z} {src.split('/')[-1] if kind!='vector' else 'mvt'}: [ms to idle, features rendered] {ts}", flush=True)
        await b.close()
asyncio.run(main())
