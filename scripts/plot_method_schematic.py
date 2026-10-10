import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

W, H = 1280, 464
PT = 0.72
INK = "#1F2937"

STYLE = {
    "mask": ("#EEF0F3", "#7A8594"),
    "teacher": ("#E8EEF7", "#5B7DB1"),
    "learned": ("#FBEFD5", "#C9A227"),
    "fill": ("#DDF0EC", "#2F7F73"),
    "out_value": ("#E3F1E3", "#4E9A4E"),
    "out_weighted": ("#FCE9D7", "#D27A2C"),
    "out_zero": ("#ECEFF2", "#8A94A0"),
}


class Canvas:
    def __init__(self):
        self.fig = plt.figure(figsize=(W / 100, H / 100))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=15, weight="normal", ha="center", va="center", style="normal", color="#000000"):
        self.ax.text(x, y, s, fontsize=size * PT, fontweight=weight, ha=ha, va=va, fontstyle=style, color=color)

    def rect(self, x, y, w, h, fill, stroke, lw=2.2, dashed=False, r=9):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fill, edgecolor=stroke, linewidth=lw * PT,
            linestyle=(0, (5, 3)) if dashed else "-", zorder=2,
        ))

    def box(self, x, y, w, h, kind, lines, size=18, sub=None, sub_size=13.5, inset=0):
        fill, stroke = STYLE[kind]
        self.rect(x, y, w, h, fill, stroke)
        n = len(lines) + (1 if sub else 0)
        step = 23 if sub is None else 21
        top = y + h / 2 - (n - 1) * step / 2
        cx = x + inset + (w - inset) / 2
        for i, s in enumerate(lines):
            self.text(cx, top + i * step, s, size=size, weight="bold")
        if sub:
            self.text(cx, top + len(lines) * step, sub, size=sub_size)

    def side_box(self, x, y, w, h, kind, title, sub):
        fill, stroke = STYLE[kind]
        self.rect(x, y, w, h, fill, stroke, lw=2, r=8)
        self.text(x + 16, y + h / 2 - 8, title, size=16, weight="bold", ha="left")
        self.text(x + 16, y + h / 2 + 10, sub, size=13.5, ha="left")

    def stack(self, x, y, w, h, title, sub, top=None):
        for k in (2, 1, 0):
            self.rect(x - 6 * k, y - 6 * k, w, h, "#FFFFFF", INK, lw=2)
        if top:
            self.text(x + w / 2, y + h / 2 - 22, top, size=19, style="italic")
            self.text(x + w / 2, y + h / 2 + 2, title, size=16, weight="bold")
            self.text(x + w / 2, y + h / 2 + 23, sub, size=13.5)
        else:
            self.text(x + w / 2, y + h / 2 - 10, title, size=17, weight="bold")
            self.text(x + w / 2, y + h / 2 + 13, sub, size=13.5)

    def arrow(self, x, y, length=30):
        a, b, h = x, x + length, 7
        pts = [(a, y - h / 2), (b - 11, y - h / 2), (b - 11, y - h), (b, y), (b - 11, y + h), (b - 11, y + h / 2), (a, y + h / 2)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor=INK, linewidth=1.8 * PT, joinstyle="round", zorder=3))

    def caption(self, x, y, lines):
        for i, s in enumerate(lines):
            self.text(x, y + i * 19, s, size=15)

    def phase(self, x, y, label, note):
        t = self.ax.text(x, y, label, fontsize=13.5 * PT, fontweight="bold", ha="left", va="center")
        self.fig.canvas.draw()
        bb = t.get_window_extent().transformed(self.ax.transData.inverted())
        self.text(bb.x1 + 6, y, "·  " + note, size=13.5, ha="left", style="italic")

    def polyline(self, pts, lw=2.2):
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
        self.ax.add_patch(PathPatch(MPath(pts, codes), facecolor="none", edgecolor=INK, linewidth=lw * PT, joinstyle="round", zorder=1))

    def grid_icon(self, cx, cy, mode):
        s, g = 9, 2
        filled = {(0, 1), (1, 3), (2, 0), (2, 2), (0, 3)}
        x0, y0 = cx - (4 * s + 3 * g) / 2, cy - (3 * s + 2 * g) / 2
        for r in range(3):
            for c in range(4):
                x, y = x0 + c * (s + g), y0 + r * (s + g)
                if (r, c) in filled:
                    kw = dict(facecolor="#4B5563", edgecolor="#4B5563", linewidth=0.5)
                elif mode == "mask" and (r, c) == (1, 1):
                    kw = dict(facecolor="#FFFFFF", edgecolor="#7C3AED", linewidth=1.3, linestyle=(0, (1.5, 1)))
                elif mode == "fill" and (r, c) in {(0, 0), (1, 0)}:
                    kw = dict(facecolor="#2F7F73", edgecolor="#2F7F73", linewidth=0.5)
                else:
                    kw = dict(facecolor="#FFFFFF", edgecolor="#9CA3AF", linewidth=0.7)
                self.ax.add_patch(Rectangle((x, y), s, s, zorder=3, **kw))

    def teachers_icon(self, cx, cy):
        for i, c in enumerate(["#93A8CC", "#6F8FC0", "#5B7DB1", "#46699E", "#2F5487"]):
            self.ax.add_patch(Circle((cx - 28 + 14 * i, cy), 5.5, facecolor=c, edgecolor="none", zorder=3))

    def tree_icon(self, cx, cy):
        pts = [(cx, cy - 10), (cx - 9, cy), (cx + 9, cy), (cx - 14, cy + 10), (cx - 4, cy + 10), (cx + 14, cy + 10)]
        for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]:
            self.ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#8A6D12", linewidth=1.1, zorder=3)
        for x, y in pts:
            self.ax.add_patch(Circle((x, y), 3.3, facecolor="#C9A227", edgecolor="none", zorder=4))

    def network_icon(self, cx, cy):
        layers = [[-9, 0, 9], [-12, -4, 4, 12], [0]]
        xs = [cx - 14, cx, cx + 14]
        for li in range(2):
            for ya in layers[li]:
                for yb in layers[li + 1]:
                    self.ax.plot([xs[li], xs[li + 1]], [cy + ya, cy + yb], color="#B9A35A", linewidth=0.6, zorder=3)
        for li, ys in enumerate(layers):
            for yy in ys:
                self.ax.add_patch(Circle((xs[li], cy + yy), 2.9, facecolor="#C9A227", edgecolor="none", zorder=4))

    def rank_icon(self, cx, cy):
        for i, h in enumerate([22, 16, 12, 8, 4]):
            self.ax.add_patch(Rectangle((cx - 21 + 9 * i, cy + 11 - h), 6.5, h, facecolor="#2F7F73", alpha=1 - 0.14 * i, edgecolor="none", zorder=3))

    def legend(self, y):
        items = [("mask", "Masking"), ("teacher", "Imputation method"), ("learned", "Learned from hidden counts"),
                 ("fill", "Ranking and filling"), ("data", "Data"), ("option", "Option")]
        widths = [len(label) * 8.2 + 48 for _, label in items]
        x = (W - sum(widths)) / 2
        for (key, label), w in zip(items, widths):
            if key == "data":
                self.rect(x - 3, y - 11, 16, 16, "#FFFFFF", INK, lw=1.6, r=3)
                self.rect(x + 1, y - 7, 16, 16, "#FFFFFF", INK, lw=1.6, r=3)
            elif key == "option":
                self.rect(x, y - 9, 18, 18, "#FFFFFF", "#8A94A0", lw=1.8, dashed=True, r=3)
            else:
                fill, stroke = STYLE[key]
                self.rect(x, y - 9, 18, 18, fill, stroke, lw=1.8, r=3)
            self.text(x + 26, y, label, size=15, ha="left")
            x += w


def build():
    c = Canvas()
    c.phase(40, 26, "PHASE 1 · LEARN", "once per dataset")

    by, bh = 66, 86
    mid = by + bh / 2
    c.stack(54, by, 150, bh, "Count Matrix", "cells × genes")
    c.caption(129, 186, ["Most entries are zero"])
    c.arrow(210, mid)

    c.grid_icon(343, 44, "mask")
    c.box(246, by, 196, bh, "mask", ["1. Hide 10% of", "Nonzero Counts"])
    c.caption(344, 176, ["Hidden counts", "become labels"])
    c.arrow(448, mid)

    c.teachers_icon(590, 46)
    c.box(484, by, 212, bh, "teacher", ["2. Five Teachers", "Propose Counts"])
    c.caption(590, 176, ["Gene median, SVD,", "kNN, MAGIC, scVI"])
    c.arrow(702, mid)

    sx, sw = 740, 292
    c.box(sx, 50, sw, 54, "learned", ["3. Fuse Proposals → μ"], size=17, sub="boosted regression", inset=44)
    c.tree_icon(sx + 28, 77)
    c.box(sx, 114, sw, 54, "learned", ["4. Train Selector → s"], size=17, sub="hidden counts vs recorded zeros", inset=44)
    c.network_icon(sx + 28, 141)
    c.caption(sx + sw / 2, 188, ["Fitted on training cells only"])
    c.arrow(1038, mid)

    c.stack(1086, by, 150, bh, "Value, Score", "for each zero", top="μ,  s")

    gy, rx = 222, 382
    ry, rh = 288, 86
    rmid = ry + rh / 2
    c.polyline([(1236, mid), (1260, mid), (1260, gy), (rx, gy), (rx, rmid), (404, rmid)])
    c.ax.add_patch(Polygon([(402, rmid - 7), (418, rmid), (402, rmid + 7)], closed=True, facecolor=INK, edgecolor=INK, zorder=3))
    c.text(820, gy - 11, "No fitting step sees a hidden count of a test cell", size=14.5, style="italic")

    c.phase(40, 250, "PHASE 2 · FILL", "test cells, chosen fill fraction")

    oy, oh = 274, 112
    c.rect(40, oy, 302, oh, "#FFFFFF", "#8A94A0", lw=2, dashed=True)
    c.text(56, oy + 20, "Options", size=16, weight="bold", ha="left")
    c.text(56, oy + 44, "Donor or perturbation labels", size=14, weight="bold", ha="left")
    c.text(56, oy + 62, "guide kNN, scVI and the selector", size=13.5, ha="left")
    c.text(56, oy + 84, "Transductive or inductive", size=14, weight="bold", ha="left")
    c.text(56, oy + 102, "teachers see all or training cells", size=13.5, ha="left")

    c.rank_icon(526, 268)
    c.box(420, ry, 212, rh, "fill", ["5. Rank Candidate", "Zeros by s"])
    c.caption(526, 394, ["Highest score first"])
    c.arrow(638, rmid)

    c.grid_icon(778, 270, "fill")
    c.box(674, ry, 208, rh, "fill", ["6. Fill the Top", "Fraction"])
    c.caption(778, 394, ["Nonzero counts", "never change"])
    c.arrow(888, rmid)

    ox, ow = 926, 314
    c.side_box(ox, 264, ow, 42, "out_value", "Fused value μ", "Restores hidden counts")
    c.side_box(ox, 311, ow, 42, "out_weighted", "Detection-weighted d·μ", "For downstream analyses")
    c.side_box(ox, 358, ow, 42, "out_zero", "Unselected zeros", "Stay zero")

    c.legend(446)
    return c.fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_evidence/figures"))
    parser.add_argument("--stem", default="safe_fusion_method")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build()
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(args.output_dir / f"{args.stem}.{ext}", facecolor="white", **kw)


if __name__ == "__main__":
    main()
