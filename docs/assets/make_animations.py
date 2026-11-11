#!/usr/bin/env python3
"""Generate the animated SVGs in the READMEs.

    docs/assets/flow.svg       how infrathon-build turns a definition into a task (main README)
    examples/kvzip/run.svg     one real run, hour by hour, with both trials' scores (example README)

Plain SVG with CSS keyframes and SMIL motion: GitHub plays both in an <img>, needs no script, and
the files follow the viewer's light or dark theme. Run this again after changing the data or design:

    python3 docs/assets/make_animations.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEME = """
  :root {
    --bg: #ffffff; --card: #f6f8fa; --line: #d0d7de; --text: #1f2328; --muted: #59636e; --faint: #eaeef2;
    --builder: #6f42c1; --tool: #0a7f73; --agent: #c4570c; --judge: #0969da; --ok: #1a7f37; --bad: #cf222e;
    --builder-soft: #f3eefc; --tool-soft: #e6f6f4; --agent-soft: #fdf1e7; --judge-soft: #e8f1fd;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117; --card: #161b22; --line: #30363d; --text: #e6edf3; --muted: #9198a1; --faint: #21262d;
      --builder: #a371f7; --tool: #3fb9a8; --agent: #f0883e; --judge: #4493f8; --ok: #3fb950; --bad: #f85149;
      --builder-soft: #2a2140; --tool-soft: #12302d; --agent-soft: #3a2414; --judge-soft: #132a45;
    }
  }
  text { font-family: %s; fill: var(--text); }
  .mono { font-family: %s; }
  .muted { fill: var(--muted); }
  .card { fill: var(--card); stroke: var(--line); stroke-width: 1.2; }
  .line { stroke: var(--line); stroke-width: 1.6; fill: none; }
  .arrow { stroke: var(--muted); stroke-width: 1.6; fill: none; marker-end: url(#head); }
""" % (FONT, MONO)


class Timeline:
    """CSS keyframes on one shared cycle, written in seconds."""

    def __init__(self, cycle: float, prefix: str) -> None:
        self.cycle = cycle
        self.prefix = prefix
        self.rules: list[str] = []
        self.count = 0

    def pct(self, seconds: float) -> str:
        return f"{max(0.0, min(100.0, 100.0 * seconds / self.cycle)):.3f}%"

    def anim(self, frames: list[tuple[float, str]], extra: str = "") -> str:
        """frames: (seconds, 'prop: value; ...'). Returns a class name to put on an element."""
        self.count += 1
        name = f"{self.prefix}{self.count}"
        body = " ".join(f"{self.pct(t)} {{ {css} }}" for t, css in frames)
        self.rules.append(f"@keyframes {name}k {{ {body} }}")
        self.rules.append(f".{name} {{ animation: {name}k {self.cycle}s linear infinite; {extra} }}")
        return name

    def css(self) -> str:
        return "\n  ".join(self.rules)


def window(t: Timeline, start: float, end: float, on: str, off: str, fade: float = 0.25) -> list[tuple[float, str]]:
    """Off, then on between start and end, then off."""
    return [(0, off), (max(0, start - 0.001), off), (start + fade, on), (end, on), (min(t.cycle, end + fade), off),
            (t.cycle, off)]


def hold_from(t: Timeline, start: float, on: str, off: str, fade: float = 0.25) -> list[tuple[float, str]]:
    """Off until start, then on to the end of the cycle (a short reset just before it loops)."""
    return [(0, off), (max(0, start - 0.001), off), (start + fade, on), (t.cycle - 0.35, on), (t.cycle, off)]


def head_marker() -> str:
    return ('<marker id="head" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill: var(--muted)"/></marker>')


def svg(width: int, height: int, title: str, css: str, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{title}">\n<title>{title}</title>\n<defs>{head_marker()}</defs>\n'
            f"<style>{THEME}\n  {css}\n</style>\n"
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="16" '
            f'style="fill: var(--bg); stroke: var(--line)"/>\n{body}\n</svg>\n')


# ---- icons (24 x 24, drawn around 0,0 as the top-left) ------------------------------------------

def icon(kind: str, x: float, y: float, color: str) -> str:
    s = f'style="fill:none; stroke: var({color}); stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round"'
    shapes = {
        "doc": '<path d="M6 2h8l5 5v15H6z"/><path d="M14 2v5h5"/><path d="M9 12h7M9 16h7"/>',
        "repo": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
        "build": '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z"/>',
        "check": '<path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
        "agent": '<rect x="3" y="4" width="18" height="14" rx="2"/><path d="M7 9l3 2.5L7 14M12 14h5"/>',
        "review": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/><path d="M8 10.5l2 2 3.5-4"/>',
        "deliver": '<path d="M3 7l9-4 9 4v10l-9 4-9-4z"/><path d="M3 7l9 4 9-4M12 11v10"/>',
        "shield": '<path d="M12 2l8 3v6c0 5-3.4 9.4-8 11-4.6-1.6-8-6-8-11V5z"/>',
        "gpu": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 6V3M10 6V3M14 6V3M18 6V3M6 21v-3M10 21v-3M14 21v-3M18 21v-3"/>',
    }
    return f'<g transform="translate({x},{y})" {s}>{shapes[kind]}</g>'


# ---- 1. the flow ------------------------------------------------------------------------------

def flow() -> str:
    W, H, T = 1100, 594, 16.0
    tl = Timeline(T, "f")
    out: list[str] = []

    out.append(f'<text x="28" y="44" style="font-size: 21px; font-weight: 650">infrathon-build</text>')
    out.append(f'<text x="214" y="44" class="muted" style="font-size: 15px">from an expert\'s definition to a '
               f'qualified Infrathon task</text>')

    # Inputs.
    inputs = [("doc", "Expert definition", "τ = (W, P, E, R, T)", 112), ("repo", "Infrathon", "example tasks", 182)]
    for kind, name, sub, y in inputs:
        cls = tl.anim([(0, "opacity: 0.25"), (0.2, "opacity: 0.25"), (1.0, "opacity: 1"), (T, "opacity: 1")])
        out.append(f'<g class="{cls}"><rect x="28" y="{y}" width="168" height="56" rx="10" class="card"/>'
                   f'{icon(kind, 40, y + 16, "--muted")}'
                   f'<text x="74" y="{y + 25}" style="font-size: 13.5px; font-weight: 600">{name}</text>'
                   f'<text x="74" y="{y + 42}" class="muted" style="font-size: 12px">{sub}</text></g>')
    out.append('<path d="M196 140 C 212 140, 210 178, 224 178" class="arrow"/>')
    out.append('<path d="M196 210 C 212 210, 210 178, 224 178" class="arrow"/>')

    # Stages: name, actor, color, icon, two lines, start and end of its active window.
    stages = [
        ("build", "builder agent", "builder", "build", ["writes the task and", "tests it on the GPUs"], 2.2, 4.0),
        ("check", "tool", "tool", "check", ["clean clone: images,", "tests, baselines"], 4.6, 5.6),
        ("trial", "trial agent", "agent", "agent", ["hours on the task,", "submits to the Judge"], 6.2, 11.6),
        ("review", "builder agent", "builder", "review", ["reads the trial and", "fixes the task"], 12.2, 13.4),
        ("deliver", "tool", "tool", "deliver", ["patches, bundle", "and the report"], 14.8, T - 0.4),
    ]
    xs = [225, 400, 575, 750, 925]
    top, h, w = 120, 116, 150
    cy = top + h / 2

    # Connectors under the boxes, then the travelling token (also under them, so it shows on the gaps).
    for a, b in zip(xs, xs[1:]):
        out.append(f'<path d="M{a + w} {cy} H{b - 2}" class="arrow"/>')
    centers = [x + w / 2 for x in xs]
    start_x, end_x = 196, centers[-1]
    length = end_x - start_x
    stops = [(0, start_x), (1.2, start_x)]
    for (name, *_rest, s, e), c in zip(stages, centers):
        stops += [(s, c), (e if name != "trial" else e, c)]
    stops[-1] = (T, end_x)
    key_points = ";".join(f"{(x - start_x) / length:.4f}" for _, x in stops)
    key_times = ";".join(f"{t / T:.4f}" for t, _ in stops)
    out.append(f'<circle r="6" style="fill: var(--agent)"><animateMotion dur="{T}s" repeatCount="indefinite" '
               f'calcMode="linear" keyPoints="{key_points}" keyTimes="{key_times}" '
               f'path="M{start_x} {cy} H{end_x}"/></circle>')

    # The retrial loop above the row.
    loop = f"M{centers[3]} {top - 2} V{top - 34} H{centers[1]} V{top - 4}"
    out.append(f'<path d="{loop}" class="line" style="stroke-dasharray: 4 4; marker-end: url(#head)"/>')
    flow_cls = tl.anim(window(tl, 13.4, 14.4, "opacity: 1; stroke-dashoffset: -60", "opacity: 0; stroke-dashoffset: 0", 0.2))
    out.append(f'<path d="{loop}" class="{flow_cls}" style="fill: none; stroke: var(--builder); stroke-width: 2.6; '
               f'stroke-dasharray: 8 6"/>')
    pill = tl.anim(window(tl, 13.4, 14.4, "opacity: 1", "opacity: 0.55", 0.2))
    out.append(f'<g class="{pill}"><rect x="{(centers[1] + centers[3]) / 2 - 92}" y="{top - 45}" width="184" height="22" '
               f'rx="11" style="fill: var(--bg); stroke: var(--line)"/>'
               f'<text x="{(centers[1] + centers[3]) / 2}" y="{top - 30}" text-anchor="middle" class="muted" '
               f'style="font-size: 12px">retrial, if the review asks</text></g>')

    for (name, actor, color, ic, lines, s, e), x in zip(stages, xs):
        soft = f"--{color}-soft"
        halo = tl.anim(window(tl, s, e, "opacity: 1", "opacity: 0"))
        done = tl.anim(hold_from(tl, e, "opacity: 1", "opacity: 0"))
        lift = tl.anim(window(tl, s, e, "transform: translateY(-3px)", "transform: translateY(0)"),
                       "transform-box: fill-box;")
        out.append(f'<g class="{lift}">')
        out.append(f'<rect x="{x - 4}" y="{top - 4}" width="{w + 8}" height="{h + 8}" rx="15" class="{halo}" '
                   f'style="fill: none; stroke: var(--{color}); stroke-width: 2; stroke-opacity: 0.35"/>')
        out.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="12" class="card"/>')
        out.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="12" class="{halo}" '
                   f'style="fill: var({soft}); stroke: var(--{color}); stroke-width: 2"/>')
        out.append(f'<rect x="{x + 12}" y="{top + 12}" width="{len(actor) * 6.4 + 16}" height="18" rx="9" '
                   f'style="fill: var({soft}); stroke: var(--{color}); stroke-opacity: 0.5"/>')
        out.append(f'<text x="{x + 20}" y="{top + 25}" style="font-size: 10.5px; font-weight: 600; '
                   f'fill: var(--{color})">{actor}</text>')
        out.append(icon(ic, x + w - 36, top + 9, f"--{color}"))
        out.append(f'<text x="{x + 14}" y="{top + 58}" style="font-size: 16px; font-weight: 650">{name}</text>')
        for i, line in enumerate(lines):
            out.append(f'<text x="{x + 14}" y="{top + 80 + 16 * i}" class="muted" style="font-size: 12px">{line}</text>')
        out.append(f'<g class="{done}"><circle cx="{x + w - 4}" cy="{top + 4}" r="9" style="fill: var(--{color})"/>'
                   f'<path d="M{x + w - 8.5} {top + 4} l3 3 l5.5 -6" style="fill: none; stroke: var(--bg); '
                   f'stroke-width: 2.2; stroke-linecap: round"/></g>')
        out.append('</g>')

        if name == "check":
            note = tl.anim(window(tl, 4.6, 5.6, "opacity: 1", "opacity: 0.6"))
            out.append(f'<text x="{x + w / 2}" y="{top + h + 20}" text-anchor="middle" class="muted {note}" '
                       f'style="font-size: 11.5px">fails → the builder fixes it</text>')
        if name == "deliver":
            for i, chip in enumerate(["task.patch", "task.bundle", "REPORT.md"]):
                cls = tl.anim(hold_from(tl, s + 0.25 * i, "opacity: 1; transform: translateY(0)",
                                        "opacity: 0; transform: translateY(6px)"), "transform-box: fill-box;")
                out.append(f'<g class="{cls}"><rect x="{x + 18}" y="{top + h + 9 + 23 * i}" width="{w - 36}" '
                           f'height="19" rx="6" style="fill: var(--tool-soft); stroke: var(--tool); stroke-opacity: .5"/>'
                           f'<text x="{x + w / 2}" y="{top + h + 22.5 + 23 * i}" text-anchor="middle" class="mono" '
                           f'style="font-size: 11.5px; fill: var(--tool)">{chip}</text></g>')

    # ---- inside a trial --------------------------------------------------------------------
    py, ph = 330, 236
    trial_center = centers[2]
    panel = tl.anim(window(tl, 6.2, 11.6, "opacity: 1", "opacity: 0.55"))
    out.append(f'<path d="M{trial_center} {top + h} V{py}" class="line" style="stroke-dasharray: 3 4"/>')
    out.append(f'<g class="{panel}">')
    out.append(f'<rect x="28" y="{py}" width="1044" height="{ph}" rx="14" style="fill: none; stroke: var(--agent); '
               f'stroke-opacity: 0.45; stroke-width: 1.5; stroke-dasharray: 6 5"/>')
    out.append(f'<text x="48" y="{py + 28}" style="font-size: 14px; font-weight: 650">Inside a trial</text>')
    out.append(f'<text x="152" y="{py + 28}" class="muted" style="font-size: 12.5px">the trial agent works exactly as '
               f'in a benchmark run</text>')
    out.append('</g>')

    wx, wy, ww, wh = 48, py + 52, 236, 132          # Work container
    cx_, cw = 360, 170                               # Controller
    jx, jw = 606, 196                                # Judge
    gx, gw = 860, 196                                # score chart
    by, bh = wy + 14, 104                            # card row
    cards_at = len(out)                              # arrows and moving dots go in here, under the cards
    out.append(f'<rect x="{wx}" y="{wy}" width="{ww}" height="{wh}" rx="10" class="card"/>')
    out.append(f'<path d="M{wx} {wy + 24} H{wx + ww}" class="line"/>')
    for i, c in enumerate(["--bad", "--agent", "--ok"]):
        out.append(f'<circle cx="{wx + 14 + 12 * i}" cy="{wy + 12}" r="4" style="fill: var({c}); opacity: .75"/>')
    out.append(f'<text x="{wx + 54}" y="{wy + 16}" class="muted" style="font-size: 11.5px">Work container · trial agent</text>')
    widths = [150, 112, 176, 96, 140, 70]
    for i, width in enumerate(widths):
        frames: list[tuple[float, str]] = [(0, "transform: scaleX(0)")]
        for cycle_start in (6.2, 8.0, 9.8):
            begin = cycle_start + 0.12 * i
            frames += [(begin - 0.001, "transform: scaleX(0)" if cycle_start == 6.2 else "transform: scaleX(1)"),
                       (begin + 0.25, "transform: scaleX(1)")]
        frames += [(11.6, "transform: scaleX(1)"), (11.9, "transform: scaleX(0)"), (T, "transform: scaleX(0)")]
        cls = tl.anim(frames, "transform-box: fill-box; transform-origin: left center;")
        color = ["--agent", "--muted", "--judge", "--muted", "--builder", "--muted"][i]
        out.append(f'<rect x="{wx + 16 + (12 if i in (2, 3) else 0)}" y="{wy + 36 + 15 * i}" width="{width}" height="6" '
                   f'rx="3" class="{cls}" style="fill: var({color}); opacity: 0.55"/>')

    out.append(f'<rect x="{cx_}" y="{by}" width="{cw}" height="{bh}" rx="10" class="card"/>')
    out.append(icon("shield", cx_ + cw / 2 - 12, by + 14, "--judge"))
    out.append(f'<text x="{cx_ + cw / 2}" y="{by + 62}" text-anchor="middle" style="font-size: 13.5px; font-weight: 650">'
               f'Controller</text>')
    out.append(f'<text x="{cx_ + cw / 2}" y="{by + 80}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'LLM integrity review</text>')
    ok_cls = tl.anim([(0, "opacity: 0"), (6.65, "opacity: 0"), (6.75, "opacity: 1"), (7.5, "opacity: 1"),
                      (7.7, "opacity: 0"), (10.25, "opacity: 0"), (10.35, "opacity: 1"), (11.1, "opacity: 1"),
                      (11.3, "opacity: 0"), (T, "opacity: 0")])
    out.append(f'<g class="{ok_cls}"><circle cx="{cx_ + cw - 18}" cy="{by + 18}" r="10" style="fill: var(--ok)"/>'
               f'<path d="M{cx_ + cw - 23} {by + 18} l3.5 3.5 l6 -7" style="fill: none; stroke: var(--bg); '
               f'stroke-width: 2.4; stroke-linecap: round"/></g>')
    bad_cls = tl.anim([(0, "opacity: 0"), (8.5, "opacity: 0"), (8.6, "opacity: 1"), (9.5, "opacity: 1"),
                       (9.7, "opacity: 0"), (T, "opacity: 0")])
    out.append(f'<g class="{bad_cls}"><circle cx="{cx_ + cw - 18}" cy="{by + 18}" r="10" style="fill: var(--bad)"/>'
               f'<path d="M{cx_ + cw - 22} {by + 14} l8 8 M{cx_ + cw - 14} {by + 14} l-8 8" style="fill: none; '
               f'stroke: var(--bg); stroke-width: 2.4; stroke-linecap: round"/>'
               f'<text x="{cx_ + cw / 2}" y="{by + bh + 18}" text-anchor="middle" style="font-size: 11.5px; '
               f'fill: var(--bad)">breaks a rule: rejected</text></g>')

    out.append(f'<rect x="{jx}" y="{by}" width="{jw}" height="{bh}" rx="10" class="card"/>')
    out.append(f'<text x="{jx + jw / 2}" y="{by + 26}" text-anchor="middle" style="font-size: 13.5px; font-weight: 650">'
               f'Judge</text>')
    out.append(f'<text x="{jx + jw / 2}" y="{by + 44}" text-anchor="middle" class="muted" style="font-size: 11.5px">'
               f'private tests on its own GPUs</text>')
    for i in range(7):
        frames = [(0, "opacity: 0.18")]
        for start in (6.9, 10.5):
            t0 = start + 0.06 * i
            frames += [(t0, "opacity: 0.18"), (t0 + 0.12, "opacity: 1"), (start + 0.75, "opacity: 1"),
                       (start + 0.95, "opacity: 0.18")]
        frames.append((T, "opacity: 0.18"))
        cls = tl.anim(frames)
        out.append(f'<rect x="{jx + 22 + 22.5 * i}" y="{by + 60}" width="17" height="26" rx="3" class="{cls}" '
                   f'style="fill: var(--judge)"/>')

    # Submissions travel under the cards: out to the Controller and the Judge, and the result back.
    lane_out, lane_pass, lane_fail = by + bh / 2, wy + wh + 18, wy + wh + 6
    pass_path = (f"M{wx + ww} {lane_out} H{jx + jw / 2} V{lane_pass} H{wx + ww * 0.62} V{wy + wh - 2}")
    fail_path = f"M{wx + ww} {lane_out} H{cx_ + cw / 2} V{lane_fail} H{wx + ww * 0.82} V{wy + wh - 2}"
    under: list[str] = []
    under.append(f'<path d="M{wx + ww} {lane_out} H{cx_ - 2}" class="arrow"/>')
    under.append(f'<path d="M{cx_ + cw} {lane_out} H{jx - 2}" class="arrow"/>')
    under.append(f'<path d="M{jx + jw / 2} {by + bh} V{lane_pass} H{wx + ww * 0.62} V{wy + wh + 2}" class="arrow" '
                 f'style="stroke-dasharray: 3 4; opacity: 0.55"/>')
    under.append(f'<text x="{(jx + wx + ww) / 2 + 40}" y="{lane_pass + 15}" text-anchor="middle" class="muted" '
                 f'style="font-size: 11.5px">valid or not, the score and the allowed feedback</text>')

    def dot(path: str, legs: list[tuple[float, float]], color_frames: str) -> str:
        """legs: (seconds, fraction along the path); invisible outside its own cycle."""
        first, last = legs[0][0], legs[-1][0]
        frames = [(0.0, 0.0)] + [(first - 0.001, 0.0)] + legs + [(last + 0.001, 1.0), (T, 1.0)]
        key_times = ";".join(f"{t / T:.4f}" for t, _ in frames)
        key_points = ";".join(f"{p:.4f}" for _, p in frames)
        vis = [(0, 0), (first - 0.001, 0), (first + 0.05, 1), (last - 0.05, 1), (last, 0), (T, 0)]
        vis_times = ";".join(f"{t / T:.4f}" for t, _ in vis)
        vis_values = ";".join(str(v) for _, v in vis)
        return (f'<circle r="6.5" style="stroke: var(--bg); stroke-width: 2">'
                f'<animateMotion dur="{T}s" repeatCount="indefinite" calcMode="linear" keyTimes="{key_times}" '
                f'keyPoints="{key_points}" path="{path}"/>'
                f'<animate attributeName="opacity" dur="{T}s" repeatCount="indefinite" keyTimes="{vis_times}" '
                f'values="{vis_values}"/>{color_frames}</circle>')

    def path_fraction(points: list[tuple[float, float]]) -> list[float]:
        lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in zip(points, points[1:])]
        total, run, out_ = sum(lengths), 0.0, [0.0]
        for length in lengths:
            run += length
            out_.append(run / total)
        return out_

    pp = [(wx + ww, lane_out), (cx_, lane_out), (cx_ + cw, lane_out), (jx, lane_out), (jx + jw / 2, lane_out),
          (jx + jw / 2, lane_pass), (wx + ww * 0.62, lane_pass), (wx + ww * 0.62, wy + wh - 2)]
    f = path_fraction(pp)
    for start in (6.2, 9.8):
        legs = [(start, f[0]), (start + 0.25, f[1]), (start + 0.55, f[2]), (start + 0.75, f[3]), (start + 1.0, f[4]),
                (start + 1.2, f[5]), (start + 1.65, f[6]), (start + 1.8, f[7])]
        color = (f'<animate attributeName="fill" dur="{T}s" repeatCount="indefinite" calcMode="discrete" '
                 f'keyTimes="0;{(start + 1.05) / T:.4f}" values="#c4570c;#1a7f37"/>')
        under.append(dot(pass_path, legs, color))
    fp = [(wx + ww, lane_out), (cx_, lane_out), (cx_ + cw / 2, lane_out), (cx_ + cw / 2, lane_fail),
          (wx + ww * 0.82, lane_fail), (wx + ww * 0.82, wy + wh - 2)]
    f = path_fraction(fp)
    legs = [(8.0, f[0]), (8.25, f[1]), (8.55, f[2]), (9.05, f[2]), (9.25, f[3]), (9.65, f[4]), (9.8, f[5])]
    color = (f'<animate attributeName="fill" dur="{T}s" repeatCount="indefinite" calcMode="discrete" '
             f'keyTimes="0;{8.6 / T:.4f}" values="#c4570c;#cf222e"/>')
    under.append(dot(fail_path, legs, color))
    out[cards_at:cards_at] = under

    # Score chart: the best score so far over a whole trial, drawn as the trial runs.
    out.append(f'<rect x="{gx}" y="{wy}" width="{gw}" height="{wh}" rx="10" class="card"/>')
    out.append(f'<text x="{gx + 14}" y="{wy + 20}" style="font-size: 12.5px; font-weight: 650">best score so far</text>')
    trial2 = json.loads((ROOT / "examples/kvzip/run/trials/round-2.json").read_text())
    best, points = 0.0, []
    for s_ in trial2["submissions"]:
        if s_.get("valid") and isinstance(s_.get("score"), (int, float)):
            best = max(best, s_["score"])
        points.append(best)
    x0, x1, y0, y1 = gx + 18, gx + gw - 16, wy + wh - 16, wy + 34
    def sx(i): return x0 + (x1 - x0) * i / (len(points) - 1)
    def sy(v): return y0 - (y0 - y1) * max(0.0, v - 40) / 60
    out.append(f'<path d="M{x0} {y0} H{x1}" class="line"/>')
    d = "M" + " L".join(f"{sx(i):.1f} {sy(v):.1f}" for i, v in enumerate(points))
    draw = tl.anim([(0, "stroke-dashoffset: 400"), (6.2, "stroke-dashoffset: 400"), (11.4, "stroke-dashoffset: 0"),
                    (T - 0.35, "stroke-dashoffset: 0"), (T, "stroke-dashoffset: 400")])
    out.append(f'<path d="{d}" class="{draw}" style="fill: none; stroke: var(--ok); stroke-width: 2.4; '
               f'stroke-linejoin: round; stroke-dasharray: 400"/>')
    label = tl.anim(hold_from(tl, 11.2, "opacity: 1", "opacity: 0"))
    out.append(f'<g class="{label}"><circle cx="{sx(len(points) - 1)}" cy="{sy(points[-1])}" r="4" '
               f'style="fill: var(--ok)"/><text x="{x1}" y="{sy(points[-1]) + 20}" text-anchor="end" '
               f'style="font-size: 12px; font-weight: 650; fill: var(--ok)">{points[-1]:.1f}</text></g>')

    # A thin progress bar along the bottom.
    bar = tl.anim([(0, "transform: scaleX(0)"), (T, "transform: scaleX(1)")], "transform-box: fill-box; transform-origin: left center;")
    out.append(f'<rect x="16" y="{H - 6}" width="{W - 32}" height="2.5" rx="1.25" class="{bar}" '
               f'style="fill: var(--line)"/>')
    return svg(W, H, "infrathon-build: from an expert's definition to a qualified Infrathon task", tl.css(), "\n".join(out))


# ---- 2. one real run, hour by hour ---------------------------------------------------------------

def steps_from_report(report: Path) -> list[tuple[str, float, float]]:
    """(step, start hour, end hour) from the report's Steps table."""
    rows: list[tuple[str, float, float]] = []
    for line in report.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] and cells[0] != "step" and not set(cells[0]) <= {"-"}:
            try:
                done = float(cells[2])
            except ValueError:
                continue
            rows.append((cells[0], rows[-1][2] if rows else 0.0, done))   # steps run back to back
        if rows and not line.startswith("|"):
            break
    return rows


def run_story() -> str:
    W, H, T = 1100, 646, 20.0
    play = 15.0                                      # seconds the playhead takes over the whole run
    tl = Timeline(T, "r")
    example = ROOT / "examples/kvzip/run"
    steps = steps_from_report(example / "REPORT.md")
    total = max(end for _, _, end in steps)
    trials = [json.loads((example / f"trials/round-{n}.json").read_text()) for n in (1, 2)]
    x0, x1 = 150, 1066
    def hx(hour): return x0 + (x1 - x0) * hour / total
    def at(hour): return play * hour / total         # when the playhead reaches that hour
    out: list[str] = []

    out.append('<text x="28" y="44" style="font-size: 21px; font-weight: 650">KVzip KV Scoring</text>')
    out.append(f'<text x="250" y="44" class="muted" style="font-size: 15px">one run of infrathon-build · '
               f'{total:.0f} hours · eight H100s</text>')
    for lx, label, color in ((770, "builder agent", "builder"), (880, "tool", "tool"), (936, "trial agent", "agent")):
        out.append(f'<rect x="{lx}" y="31" width="12" height="12" rx="3" style="fill: var(--{color})"/>'
                   f'<text x="{lx + 18}" y="41.5" class="muted" style="font-size: 12px">{label}</text>')

    lanes = {"builder": 92, "tool": 128, "agent": 164}
    lane_of = {"build": "builder", "review-1": "builder", "review-2": "builder", "setup": "tool", "check-1": "tool",
               "check-2": "tool", "check-final": "tool", "trial-1": "agent", "trial-2": "agent"}
    for name, y in (("Builder agent", 92), ("Tool", 128), ("Trial agent", 164)):
        out.append(f'<text x="28" y="{y + 15}" class="muted" style="font-size: 12.5px">{name}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="22" rx="5" style="fill: var(--faint)"/>')
    for hour in range(0, int(total) + 1, 3):
        out.append(f'<path d="M{hx(hour)} 192 V198" class="line"/>'
                   f'<text x="{hx(hour)}" y="215" text-anchor="middle" class="muted" style="font-size: 11px">{hour} h</text>')

    # Bars revealed by a clip that grows with the playhead.
    out.append(f'<clipPath id="reveal"><rect x="{x0}" y="80" height="120" width="0">'
               f'<animate attributeName="width" dur="{T}s" repeatCount="indefinite" '
               f'keyTimes="0;{play / T:.4f};{(T - 0.4) / T:.4f};1" values="0;{x1 - x0};{x1 - x0};0"/></rect></clipPath>')
    out.append('<g clip-path="url(#reveal)">')
    for name, start, end in steps:
        color = lane_of.get(name, "tool")
        y = lanes[color]
        width = max(hx(end) - hx(start), 3)
        out.append(f'<rect x="{hx(start)}" y="{y}" width="{width}" height="22" rx="5" style="fill: var(--{color})"/>')
        label = {"trial-1": "trial 1", "trial-2": "trial 2", "review-1": "review", "review-2": "review"}.get(name, name)
        if width > 7.5 * len(label) + 8:
            out.append(f'<text x="{hx(start) + width / 2}" y="{y + 15}" text-anchor="middle" '
                       f'style="font-size: 11.5px; font-weight: 600; fill: var(--bg)">{label}</text>')
    out.append('</g>')
    out.append(f'<line x1="{x0}" y1="84" x2="{x0}" y2="194" style="stroke: var(--text); stroke-width: 1.5; opacity: .7">'
               f'<animate attributeName="x1" dur="{T}s" repeatCount="indefinite" keyTimes="0;{play / T:.4f};1" '
               f'values="{x0};{x1};{x1}"/><animate attributeName="x2" dur="{T}s" repeatCount="indefinite" '
               f'keyTimes="0;{play / T:.4f};1" values="{x0};{x1};{x1}"/></line>')
    out.append(f'<circle cx="{x0}" cy="84" r="5" style="fill: var(--text)"><animate attributeName="cx" dur="{T}s" '
               f'repeatCount="indefinite" keyTimes="0;{play / T:.4f};1" values="{x0};{x1};{x1}"/></circle>')

    # What happened, as the playhead reaches it.
    end_of = {name: end for name, _, end in steps}
    best = [t["best"] for t in trials]
    events = [
        (end_of["build"], "builder", "Built", ["task, Judge and suites;", "18 planted bad", "submissions all rejected"]),
        (end_of["check-1"], "tool", "Checked", ["clean clone: images,", "public tests, KNorm 24.3", "and Fast KVzip 89.3"]),
        (end_of["trial-1"], "agent", f"Trial 1: {best[0]:.2f}", ["but the Work image", "named the hidden", "workloads"]),
        (end_of["review-1"], "builder", "Review 1: 7 fixes", ["leak removed, weights", "check, deadline and", "brief corrected"]),
        (end_of["trial-2"], "agent", f"Trial 2: {best[1]:.2f}", ["no leak this time;", "an unstated memory", "rule hit 3 honest ones"]),
        (end_of["check-final"], "tool", "Delivered", ["review 2: 4 more fixes;", "final check passed;", "patches and report"]),
    ]
    cw_, gap, cy0 = 160, 16.8, 250
    for i, (hour, color, title, lines) in enumerate(events):
        cx = 28 + i * (cw_ + gap)
        cls = tl.anim(hold_from(tl, at(hour), "opacity: 1; transform: translateY(0)", "opacity: 0; transform: translateY(8px)",
                                0.35), "transform-box: fill-box;")
        out.append(f'<g class="{cls}">')
        elbow = cy0 - 26 + 12 * (i % 2)                # alternate heights so neighbouring leaders never overlap
        out.append(f'<path d="M{hx(hour)} 196 V{elbow} H{cx + cw_ / 2} V{cy0}" style="fill: none; stroke: var(--{color}); '
                   f'stroke-width: 1.3; stroke-dasharray: 3 3"/>')
        out.append(f'<circle cx="{hx(hour)}" cy="196" r="3.5" style="fill: var(--{color})"/>')
        out.append(f'<rect x="{cx}" y="{cy0}" width="{cw_}" height="84" rx="10" class="card"/>')
        out.append(f'<rect x="{cx}" y="{cy0 + 10}" width="4" height="64" rx="2" style="fill: var(--{color})"/>')
        out.append(f'<text x="{cx + 14}" y="{cy0 + 22}" style="font-size: 13.5px; font-weight: 650">{title}</text>')
        for j, line in enumerate(lines):
            out.append(f'<text x="{cx + 14}" y="{cy0 + 41 + 15 * j}" class="muted" style="font-size: 11.5px">{line}</text>')
        out.append('</g>')

    # Both trials' scores, each bar rising as the playhead passes its submission.
    for n, trial in enumerate(trials):
        name = f"trial-{n + 1}"
        start = next(s for s_, s, _ in steps if s_ == name)
        end = end_of[name]
        px, py, pw, ph = 28 + n * 532, 356, 512, 266
        title = ["Trial 1 · the hidden workload names were visible", "Trial 2 · after review 1's fixes"][n]
        out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="12" class="card"/>')
        out.append(f'<text x="{px + 16}" y="{py + 26}" style="font-size: 13.5px; font-weight: 650">{title}</text>')
        out.append(f'<text x="{px + pw - 16}" y="{py + 26}" text-anchor="end" class="muted" style="font-size: 12px">'
                   f'{len(trial["submissions"])} submissions</text>')
        gx0, gx1, gy0, gy1 = px + 44, px + pw - 96, py + ph - 30, py + 68
        def gy(v): return gy0 - (gy0 - gy1) * v / 100
        for v in (0, 25, 50, 75, 100):
            out.append(f'<path d="M{gx0} {gy(v)} H{gx1}" style="stroke: var(--faint); stroke-width: 1"/>'
                       f'<text x="{gx0 - 8}" y="{gy(v) + 4}" text-anchor="end" class="muted" style="font-size: 10.5px">{v}</text>')
        for value, label in ((89.25, "Fast KVzip 89.3"), (24.29, "KNorm 24.3")):
            out.append(f'<path d="M{gx0} {gy(value)} H{gx1}" style="stroke: var(--judge); stroke-width: 1.2; '
                       f'stroke-dasharray: 5 4; opacity: .8"/><text x="{gx1 + 8}" y="{gy(value) + 4}" '
                       f'style="font-size: 11px; fill: var(--judge)">{label}</text>')
        subs = trial["submissions"]
        slot = (gx1 - gx0) / len(subs)
        best_so_far, peak_index = 0.0, 0
        for i, s_ in enumerate(subs):
            hour = start + (end - start) * (i + 0.8) / len(subs)
            bx = gx0 + slot * i + slot * 0.18
            bw = slot * 0.64
            appear = hold_from(tl, at(hour), "transform: scaleY(1); opacity: 1", "transform: scaleY(0); opacity: 0", 0.3)
            cls = tl.anim(appear, "transform-box: fill-box; transform-origin: center bottom;")
            if s_.get("valid"):
                score = float(s_["score"])
                if score > best_so_far:
                    best_so_far, peak_index = score, i
                out.append(f'<rect x="{bx:.1f}" y="{gy(score):.1f}" width="{bw:.1f}" height="{gy0 - gy(score):.1f}" rx="2" '
                           f'class="{cls}" style="fill: var(--agent); opacity: .85"/>')
            else:
                out.append(f'<g class="{cls}"><rect x="{bx:.1f}" y="{gy0 - 7}" width="{bw:.1f}" height="7" rx="2" '
                           f'style="fill: var(--bad)"/><text x="{bx + bw / 2:.1f}" y="{gy0 - 12}" text-anchor="middle" '
                           f'style="font-size: 11px; font-weight: 700; fill: var(--bad)">×</text></g>')
        label = tl.anim(hold_from(tl, at(end), "opacity: 1", "opacity: 0", 0.3))
        bx = gx0 + slot * peak_index + slot / 2
        out.append(f'<g class="{label}"><path d="M{bx - 5:.1f} {gy(best_so_far) - 9:.1f} h10 l-5 6 z" style="fill: var(--agent)"/>'
                   f'<text x="{min(bx, gx1 - 34):.1f}" y="{gy(best_so_far) - 13:.1f}" text-anchor="middle" '
                   f'style="font-size: 12.5px; font-weight: 700; fill: var(--agent)">best {best_so_far:.2f}</text></g>')
        out.append(f'<text x="{(gx0 + gx1) / 2}" y="{py + ph - 9}" text-anchor="middle" class="muted" '
                   f'style="font-size: 11px">submissions in order · × rejected by a faulty check, fixed in the next review</text>')

    return svg(W, H, "KVzip KV Scoring: one run of infrathon-build, hour by hour", tl.css(), "\n".join(out))


if __name__ == "__main__":
    for target, make in ((ROOT / "docs/assets/flow.svg", flow), (ROOT / "examples/kvzip/run.svg", run_story)):
        target.write_text(make(), encoding="utf-8")
        print(f"wrote {target}")
