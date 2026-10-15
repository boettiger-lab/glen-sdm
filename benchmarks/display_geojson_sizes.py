import duckdb, time, json, gzip, os, sys, h3
d = sys.argv[1]
con = duckdb.connect(); con.sql("LOAD h3")
for res in (8, 7, 6):
    t = time.time()
    rows = con.sql(f"""SELECT h3_cell_to_parent(h8, {res}) c, round(avg(suitability), 3) s FROM read_parquet('{d}/pred.parquet') GROUP BY 1""").fetchall()
    feats = []
    for c, s in rows:
        ring = [[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(h3.int_to_str(c))]
        ring.append(ring[0])
        feats.append('{"type":"Feature","properties":{"s":%s},"geometry":{"type":"Polygon","coordinates":[%s]}}' % (s, json.dumps(ring, separators=(",", ":"))))
    body = '{"type":"FeatureCollection","features":[' + ",".join(feats) + "]}"
    tg = time.time() - t
    p = f"{d}/hex{res}.geojson"; open(p, "w").write(body)
    gz = len(gzip.compress(body.encode(), 6))
    print(f"res {res}: {len(rows):>7} cells  gen {tg:5.2f}s  size {len(body)/1e6:6.1f} MB  gzip {gz/1e6:5.1f} MB")
b = open(f"{d}/range.geojson").read()
print(f"range polygon: size {len(b)/1e6:.2f} MB gzip {len(gzip.compress(b.encode()))/1e6:.2f} MB")
