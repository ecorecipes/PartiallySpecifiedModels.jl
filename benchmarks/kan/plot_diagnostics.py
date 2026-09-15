import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_key(row):
    return row["study"], row["case"], row["model"], row["seed"]


def edge_key(row):
    return int(row["layer"]), int(row["input"]), int(row["output"])


def render_svg(key, edges, coverage):
    columns, panel_width, panel_height = 3, 320, 220
    nrows = math.ceil(len(edges)/columns)
    width, height = columns*panel_width, 115+nrows*panel_height
    title = html.escape(" | ".join(key[:3]) + " | seed " + key[3])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="11" fill="#252525">',
             f'<text x="16" y="23" font-size="17">{title}</text>',
             '<text x="16" y="44">Logical input coordinates; each edge has its own vertical scale.</text>',
             '<text x="16" y="62">Grey: central grid. Green: training input range (clipped to panel), not uncertainty.</text>']
    colors = {"base": "#2563eb", "spline": "#d97706", "total": "#202020"}
    for i, (name, color) in enumerate(colors.items()):
        x = 16+100*i
        parts.append(f'<line x1="{x}" y1="80" x2="{x+20}" y2="80" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x+25}" y="84">{name.capitalize()}</text>')
    for panel, (edge, rows) in enumerate(sorted(edges.items())):
        rows = sorted(rows, key=lambda r: float(r["logical_x"]))
        xs = [float(r["logical_x"]) for r in rows]
        values = {name: [float(r[name]) for r in rows] for name in colors}
        assert all(math.isfinite(v) for v in xs)
        assert all(math.isfinite(v) for ys in values.values() for v in ys)
        assert len(xs) >= 2 and all(a < b for a, b in zip(xs, xs[1:]))
        xmin, xmax = xs[0], xs[-1]
        ymin = min(v for ys in values.values() for v in ys)
        ymax = max(v for ys in values.values() for v in ys)
        margin = 0.08*(ymax-ymin) if ymax > ymin else 0.1*max(abs(ymin), 1.0)
        ymin, ymax = ymin-margin, ymax+margin
        left = (panel % columns)*panel_width+55
        top = 115+(panel//columns)*panel_height
        pw, ph = 240, 155
        px = lambda x: left+(x-xmin)/(xmax-xmin)*pw
        py = lambda y: top+ph-(y-ymin)/(ymax-ymin)*ph
        parts.append(f'<text x="{left}" y="{top-10}" font-weight="bold">L{edge[0]}: input {edge[1]} -&gt; output {edge[2]}</text>')
        parts.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbbbbb"/>')
        regions = [(float(rows[0]["grid_lo"]), float(rows[0]["grid_hi"]), "#ececec")]
        observed = coverage[key+edge[:2]]
        regions.append((float(observed["observed_lo"]), float(observed["observed_hi"]), "#d4ebdb"))
        for lo, hi, color in regions:
            lo, hi = max(lo, xmin), min(hi, xmax)
            if hi >= lo:
                parts.append(f'<rect x="{px(lo):.3f}" y="{top}" width="{px(hi)-px(lo):.3f}" height="{ph}" fill="{color}"/>')
        if ymin <= 0 <= ymax:
            parts.append(f'<line x1="{left}" x2="{left+pw}" y1="{py(0):.3f}" y2="{py(0):.3f}" stroke="#aaaaaa"/>')
        for name, color in colors.items():
            points = " ".join(f"{px(x):.3f},{py(y):.3f}" for x, y in zip(xs, values[name]))
            thickness = 2 if name == "total" else 1.3
            dash = ' stroke-dasharray="4 3"' if name == "base" else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{thickness}"{dash}/>')
        for x in (xmin, (xmin+xmax)/2, xmax):
            parts.append(f'<text x="{px(x):.3f}" y="{top+ph+17}" text-anchor="middle">{x:.3g}</text>')
        for y in (ymin+margin, ymax-margin):
            parts.append(f'<text x="{left-5}" y="{py(y)+4:.3f}" text-anchor="end">{y:.3g}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("benchmarks/kan/results/kan-diagnostics"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    curves = read_rows(args.input/"edge_curves.csv")
    coverage = {model_key(r)+(int(r["layer"]), int(r["input"])): r
                for r in read_rows(args.input/"coverage.csv") if r["partition"] == "training"}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in curves:
        grouped[model_key(row)][edge_key(row)].append(row)
    output = args.output or args.input/"plots"
    output.mkdir(parents=True, exist_ok=False)
    names = []
    for key, edges in sorted(grouped.items()):
        name = "__".join(key)+".svg"
        if Path(name).name != name:
            raise ValueError("plot identifiers must not contain path separators")
        svg = render_svg(key, edges, coverage)
        root = ElementTree.fromstring(svg)
        assert len(root.findall(".//{http://www.w3.org/2000/svg}polyline")) == 3*len(edges)
        (output/name).write_text(svg)
        names.append(name)
    script = Path(__file__).read_bytes()
    (output/"plot-script.py").write_bytes(script)
    (output/"metadata.json").write_text(json.dumps(dict(
        inputs={name: fingerprint(args.input/name) for name in ("edge_curves.csv", "coverage.csv", "metadata.toml")},
        script_sha256=hashlib.sha256(script).hexdigest(),
        plots=names, coordinates="logical", model_evaluation=False,
        interpretation="Representative effective edges; training-range shading is not uncertainty or joint support"),
        indent=2) + "\n")
    print("Rendered", len(names), "edge figures from saved diagnostic data.")


if __name__ == "__main__":
    main()
