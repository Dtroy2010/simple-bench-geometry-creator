
# -*- coding: utf-8 -*-
# Author: DRoy
# Date: 2024-12-04
"""
Simple Bench Geometry Creator — Streamlit App
==============================================
Generates a 2D stepped bench-slope geometry from user-defined segments.
Each segment can have its own bench height, face angle, stack height, and
inter-ramp angle. Segments are chained automatically.

Run with:
    streamlit run "Simple Bench Geometry Creator_streamlit.py"
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json

# ===========================================================================
#  FUNCTIONS  (no need to edit below this line)
# ===========================================================================

def calculate_bench_width(inter_ramp_angle: float, bench_height: float, bench_face_angle: float) -> float:
    """Return the catch-berm width required to achieve the target inter-ramp angle."""
    return (bench_height / np.tan(np.radians(inter_ramp_angle))
            - bench_height / np.tan(np.radians(bench_face_angle)))


def generate_full_slope(
    bench_width: float,
    stack_height: float,
    bench_face_angle: float,
    bench_height: float,
    overall_height: float,
    start_coords: tuple = (0, 0),
    direction: str = "right",
    berm_width: float = 0.0,
) -> np.ndarray:
    """Generate vertex coordinates for one slope segment."""
    num_full_stacks = int(overall_height // stack_height)
    partial_stack_height = overall_height % stack_height

    vertices = [start_coords]
    x, y = start_coords
    sign = 1 if direction == "right" else -1

    total_stacks = num_full_stacks + (1 if partial_stack_height > 0 else 0)
    for stack in range(total_stacks):
        current_stack_height = stack_height if stack < num_full_stacks else partial_stack_height
        num_benches = int(current_stack_height // bench_height)

        for _ in range(num_benches):
            x += sign * bench_height / np.tan(np.radians(bench_face_angle))
            y += bench_height
            vertices.append((x, y))
            x += sign * bench_width
            vertices.append((x, y))

        remainder = current_stack_height % bench_height
        if remainder > 0:
            x += sign * remainder / np.tan(np.radians(bench_face_angle))
            y += remainder
            vertices.append((x, y))
            x += sign * bench_width
            vertices.append((x, y))

        # Inter-stack berm: wide flat platform after each complete stack except the last
        if berm_width > 0 and stack < total_stacks - 1:
            x += sign * berm_width
            vertices.append((x, y))

    return np.array(vertices)


def generate_combined_slope(slope_segments: list, direction: str = "right") -> tuple:
    """Chain all segments and return combined vertices and per-segment info."""
    combined_vertices = []
    current_start = (0.0, 0.0)
    segment_info = []

    for i, seg in enumerate(slope_segments):
        bench_width = seg.get("bench_width")
        ira = seg["inter_ramp_angle"]

        if bench_width is None:
            bench_width = calculate_bench_width(ira, seg["bench_height"], seg["bench_face_angle"])

        # Validate bench_height divides stack_height evenly
        if seg["stack_height"] % seg["bench_height"] != 0:
            print(f"  Warning: bench_height ({seg['bench_height']} m) does not divide "
                  f"stack_height ({seg['stack_height']} m) evenly in segment {i + 1}.")

        label = seg.get("label", f"Segment {i + 1}")
        print(f"{label}: IRA={ira}°  BFA={seg['bench_face_angle']}°  "
              f"bench_width={bench_width:.2f} m  height={seg['overall_height']} m")

        vertices = generate_full_slope(
            bench_width=bench_width,
            stack_height=seg["stack_height"],
            bench_face_angle=seg["bench_face_angle"],
            bench_height=seg["bench_height"],
            overall_height=seg["overall_height"],
            start_coords=current_start,
            direction=direction,
            berm_width=seg.get("berm_width", 0.0),
        )
        combined_vertices.append(vertices)
        segment_info.append({
            "label": label,
            "x_start": float(vertices[0, 0]),
            "y_start": float(vertices[0, 1]),
            "x_end": float(vertices[-1, 0]),
            "y_end": float(vertices[-1, 1]),
            "bench_face_angle": seg["bench_face_angle"],
            "inter_ramp_angle": ira,
        })

        # Flat road/access bench between this segment and the next
        road_width = seg.get("road_width")
        not_last = i < len(slope_segments) - 1
        if road_width is not None and not_last:
            sign = 1 if direction == "right" else -1
            lx, ly = float(vertices[-1, 0]), float(vertices[-1, 1])
            combined_vertices.append(np.array([[lx + sign * road_width, ly]]))
            current_start = (lx + sign * road_width, ly)
        else:
            current_start = tuple(vertices[-1])

    return np.vstack(combined_vertices), segment_info


# ===========================================================================
#  STREAMLIT UI
# ===========================================================================

def _default_segment(n: int, units: str = "m") -> dict:
    bh = 40.0  if units == "ft" else 12.0
    sh = 160.0 if units == "ft" else 48.0
    rw = 65.0  if units == "ft" else 20.0
    return {
        "label": f"Segment {n}",
        "inter_ramp_angle": 37,
        "bench_face_angle": 65,
        "bench_height": bh,
        "stack_height": sh,
        "overall_height": sh,
        "road_width": rw,
        "berm_width": 0.0,
    }


def build_geometry(
    segments: list[dict],
    direction: str,
    toe_width: float,
    crest_width: float,
    depth_below_toe: float,
    unit_label: str = "m",
    toe_anchor: tuple | None = None,
) -> tuple[np.ndarray, list[dict], list[str]]:
    """Build the full slope geometry from the UI segment definitions.

    Returns:
        Tuple of (all_vertices, segment_info, warnings).
    """
    slope_segments = []
    warnings = []
    for i, s in enumerate(segments):
        if s["stack_height"] % s["bench_height"] != 0:
            warnings.append(
                f"{s['label']}: bench_height ({s['bench_height']} {unit_label}) does not "
                f"divide stack_height ({s['stack_height']} {unit_label}) evenly."
            )
        slope_segments.append({
            "inter_ramp_angle": s["inter_ramp_angle"],
            "bench_face_angle": s["bench_face_angle"],
            "bench_height": s["bench_height"],
            "stack_height": s["stack_height"],
            "overall_height": s["overall_height"],
            "road_width": s["road_width"] if i < len(segments) - 1 else None,
            "label": s["label"],
            "bench_width": calculate_bench_width(s["inter_ramp_angle"], s["bench_height"], s["bench_face_angle"]),
            "berm_width": s.get("berm_width", 0.0),
        })

    all_vertices, segment_info = generate_combined_slope(slope_segments, direction=direction)

    # Carry bench_width into segment_info for display in the plot
    for info, seg in zip(segment_info, slope_segments):
        info["bench_width"] = seg["bench_width"]
    sign = 1 if direction == "right" else -1

    if toe_width > 0:
        toe_pt = np.array([[all_vertices[0, 0] - sign * toe_width, all_vertices[0, 1]]])
        all_vertices = np.vstack([toe_pt, all_vertices])
    if crest_width > 0:
        crest_pt = np.array([[all_vertices[-1, 0] + sign * crest_width, all_vertices[-1, 1]]])
        all_vertices = np.vstack([all_vertices, crest_pt])

    x_shift = all_vertices[:, 0].min()
    y_shift = all_vertices[:, 1].min()
    all_vertices[:, 0] -= x_shift
    all_vertices[:, 1] -= y_shift
    for info in segment_info:
        info["x_start"] -= x_shift
        info["y_start"] -= y_shift
        info["x_end"] -= x_shift
        info["y_end"] -= y_shift

    # Per-segment crest and toe points
    # Crest: top of last bench face = x_end stepped back one bench_width
    # Toe:   first vertex of the segment (start of first bench face)
    segment_crests = [
        np.array([info["x_end"] - sign * info["bench_width"], info["y_end"]])
        for info in segment_info
    ]
    segment_toes = [
        np.array([info["x_start"], info["y_start"]])
        for info in segment_info
    ]

    if toe_anchor is not None:
        dx = toe_anchor[0] - float(segment_toes[0][0])
        dy = toe_anchor[1] - float(segment_toes[0][1])
        all_vertices[:, 0] += dx
        all_vertices[:, 1] += dy
        for info in segment_info:
            info["x_start"] += dx
            info["y_start"] += dy
            info["x_end"] += dx
            info["y_end"] += dy
        segment_crests = [np.array([c[0] + dx, c[1] + dy]) for c in segment_crests]
        segment_toes = [np.array([t[0] + dx, t[1] + dy]) for t in segment_toes]

    x_first, y_first = float(all_vertices[0, 0]), float(all_vertices[0, 1])
    x_last = float(all_vertices[-1, 0])
    closing_pts = np.array([
        [x_last,  y_first - depth_below_toe],
        [x_first, y_first - depth_below_toe],
        [x_first, y_first],
    ])
    all_vertices = np.vstack([all_vertices, closing_pts])
    return all_vertices, segment_info, warnings, segment_toes, segment_crests



def make_figure(
    all_vertices: np.ndarray,
    segment_info: list[dict],
    direction: str,
    unit_label: str = "m",
    crest_points: list | None = None,
    toe_points: list | None = None,
    setback_points: list | None = None,
    toe_offset_points: list | None = None,
    boundary_point: np.ndarray | None = None,
    damage_line_pts: list | None = None,
) -> plt.Figure:
    """Create and return the matplotlib figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill(all_vertices[:, 0], all_vertices[:, 1], color="darkorange", alpha=0.12, zorder=0)
    ax.plot(all_vertices[:, 0], all_vertices[:, 1], color="darkorange", linewidth=1.5)

    for info in segment_info:
        ax.axvline(x=info["x_end"], color="steelblue", linestyle="--", linewidth=0.8, alpha=0.6)
        label = f"{info['label']}\nBFA: {info['bench_face_angle']}°\nIRA: {info['inter_ramp_angle']}°\nBW: {info['bench_width']:.1f} {unit_label}"
        nudge = 20
        # Place label well inside the slope body from the dashed line
        if direction == "left":
            label_x = info["x_end"] - nudge
            ha = "right"
        else:
            label_x = info["x_end"] + nudge
            ha = "left"
        label_y = info["y_end"]
        ax.text(
            label_x,
            label_y,
            label,
            color="steelblue",
            fontsize=8,
            va="top",
            ha=ha,
        )

    if crest_points:
        for pt in crest_points:
            ax.plot(pt[0], pt[1], "o", color="red", markersize=8, zorder=3)
    if toe_points:
        for pt in toe_points:
            ax.plot(pt[0], pt[1], "o", color="red", markersize=8, zorder=3)
    if setback_points:
        for pt in setback_points:
            ax.plot(pt[0], pt[1], "o", color="steelblue", markersize=8, zorder=3)
    if toe_offset_points:
        for pt in toe_offset_points:
            ax.plot(pt[0], pt[1], "o", color="green", markersize=8, zorder=3)
    if boundary_point is not None:
        ax.plot(boundary_point[0], boundary_point[1], "o", color="darkorange", markersize=8, zorder=3, markeredgecolor="black", markeredgewidth=0.8)
    if damage_line_pts:
        dlx = [pt[0] for pt in damage_line_pts]
        dly = [pt[1] for pt in damage_line_pts]
        ax.plot(dlx, dly, color="red", linewidth=1.5, linestyle="--", zorder=2)

    ax.set_aspect("equal")
    ax.set_xlabel(f"Horizontal Distance ({unit_label})")
    ax.set_ylabel(f"Elevation ({unit_label})")
    _dir_label = "Left to Right" if direction == "left" else "Right to Left"
    ax.set_title(f"Bench Slope Geometry — Failure Direction: {_dir_label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Bench Geometry Creator", layout="wide")
    st.title("Simple Bench Geometry Creator")
    st.caption("Define slope segments from bottom to top and generate a 2D stepped bench geometry.")

    # ── Session state: segment list ─────────────────────────────────────────
    if "segments" not in st.session_state:
        st.session_state.segments = [_default_segment(1)]

    # ── Sidebar: global settings ────────────────────────────────────────────
    with st.sidebar:
        st.header("Global Settings")
        units = st.selectbox("Units", ["m", "ft"], index=0, key="units_select")
        u = units

        # When units change, reset segments and sidebar inputs to sensible defaults
        if st.session_state.get("_last_units") != units:
            st.session_state["_last_units"] = units
            st.session_state.segments = [_default_segment(1, units)]
            for _k in ["toe_width", "crest_width", "depth_below_toe", "toe_anchor_x", "toe_anchor_y"]:
                st.session_state.pop(_k, None)
            st.rerun()

        _failure_dir = st.selectbox("Failure Direction", ["Left to Right", "Right to Left"], index=0, key="failure_dir")
        direction = "left" if _failure_dir == "Left to Right" else "right"
        toe_width = st.number_input(
            f"Toe Platform Width ({u})", min_value=0.0,
            value=40.0 if units == "m" else 130.0, step=1.0, key="toe_width"
        )
        crest_width = st.number_input(
            f"Crest Platform Width ({u})", min_value=0.0,
            value=40.0 if units == "m" else 130.0, step=1.0, key="crest_width"
        )
        depth_below_toe = st.number_input(
            f"Depth Below Toe ({u})", min_value=0.1,
            value=50.3 if units == "m" else 165.0, step=0.1,
            key="depth_below_toe",
            help="Closes the polygon at this depth below the toe elevation."
        )
        st.divider()
        use_toe_anchor = st.checkbox(
            "Pin Segment 1 Toe to Coordinates",
            value=False,
            key="use_toe_anchor",
            help="Anchor the Segment 1 toe at specific X, Y coordinates. The entire section will be positioned relative to this point.",
        )
        if use_toe_anchor:
            toe_anchor_x = st.number_input(f"Toe X ({u})", value=0.0, step=1.0, key="toe_anchor_x", format="%.2f")
            toe_anchor_y = st.number_input(f"Toe Y ({u})", value=0.0, step=1.0, key="toe_anchor_y", format="%.2f")
            toe_anchor = (toe_anchor_x, toe_anchor_y)
        else:
            toe_anchor = None
        st.divider()
        if st.button("➕ Add Segment", use_container_width=True):
            n = len(st.session_state.segments) + 1
            st.session_state.segments.append(_default_segment(n, units))
            st.rerun()
        st.divider()
        st.subheader("Damage Region Line")
        show_damage_line = st.checkbox("Show Damage Region Line", value=False, key="show_damage_line",
                                       help="Dashed red polyline: Crest Setback → Toe Offset Points → Boundary Point.")
        if show_damage_line or st.session_state.get("show_setback", False):
            crest_setback = st.number_input(
                f"Crest Setback ({u})",
                min_value=0.0,
                value=15.0 if units == "m" else 50.0,
                step=0.5,
                key="crest_setback",
                help="Horizontal distance set back from each segment crest, away from the slope face.",
            )
        else:
            crest_setback = 0.0
        st.divider()
        st.subheader("Reference Points")
        show_damage = st.checkbox("Show Crest Point", value=False, key="show_damage")
        show_toe = st.checkbox("Show Toe Point", value=False, key="show_toe")
        show_setback = st.checkbox("Show Crest Setback Point", value=False, key="show_setback")
        show_toe_offset = st.checkbox("Show Toe Offset Point", value=False, key="show_toe_offset",
                                      help="Point offset 0.3H from each segment toe, perpendicular to the overall slope line into the slope.")
        show_boundary_pt = st.checkbox("Show Boundary Point", value=False, key="show_boundary_pt",
                                       help="Point at the same elevation as the lowest toe offset point, extended to the far edge of the model boundary.")
        st.divider()
        st.subheader("Save / Load Configuration")
        # ── Save ──
        _config_to_save = {
            "version": 1,
            "units": units,
            "failure_direction": st.session_state.get("failure_dir", "Left to Right"),
            "toe_width": float(st.session_state.get("toe_width", 40.0)),
            "crest_width": float(st.session_state.get("crest_width", 40.0)),
            "depth_below_toe": float(st.session_state.get("depth_below_toe", 50.3)),
            "use_toe_anchor": bool(st.session_state.get("use_toe_anchor", False)),
            "toe_anchor_x": float(toe_anchor[0]) if toe_anchor is not None else float(st.session_state.get("toe_anchor_x", 0.0)),
            "toe_anchor_y": float(toe_anchor[1]) if toe_anchor is not None else float(st.session_state.get("toe_anchor_y", 0.0)),
            "crest_setback": float(st.session_state.get("crest_setback", 15.0 if units == "m" else 50.0)),
            "segments": [
                {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in seg.items()}
                for seg in st.session_state.segments
            ],
        }
        st.download_button(
            label="💾 Save Configuration",
            data=json.dumps(_config_to_save, indent=2),
            file_name="bench_geometry_config.json",
            mime="application/json",
            use_container_width=True,
        )
        # ── Load ──
        _uploaded_config = st.file_uploader(
            "Load Configuration (.json)", type=["json"], key="config_upload",
        )
        if _uploaded_config is not None:
            _content = _uploaded_config.read()
            _content_hash = hash(_content)
            if st.session_state.get("_last_config_hash") != _content_hash:
                try:
                    _cfg = json.loads(_content)
                    st.session_state["_last_config_hash"] = _content_hash
                    st.session_state.segments = _cfg["segments"]
                    _new_units = _cfg.get("units", "m")
                    st.session_state["units_select"] = _new_units
                    st.session_state["_last_units"] = _new_units
                    st.session_state["failure_dir"] = _cfg.get("failure_direction", "Left to Right")
                    st.session_state["toe_width"] = _cfg.get("toe_width", 40.0)
                    st.session_state["crest_width"] = _cfg.get("crest_width", 40.0)
                    st.session_state["depth_below_toe"] = _cfg.get("depth_below_toe", 50.3)
                    st.session_state["use_toe_anchor"] = _cfg.get("use_toe_anchor", False)
                    st.session_state["toe_anchor_x"] = _cfg.get("toe_anchor_x", 0.0)
                    st.session_state["toe_anchor_y"] = _cfg.get("toe_anchor_y", 0.0)
                    st.session_state["crest_setback"] = _cfg.get("crest_setback", 15.0)
                    st.rerun()
                except Exception as _load_err:
                    st.error(f"Failed to load configuration: {_load_err}")

    # ── Two-column layout ───────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ── LEFT: Segment editors ───────────────────────────────────────────────
    with left_col:
        st.subheader("Slope Segments (Bottom → Top)")
        to_delete = None
        for i, seg in enumerate(st.session_state.segments):
            with st.expander(f"**{seg['label']}**", expanded=True):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    seg["label"] = st.text_input("Label", value=seg["label"], key=f"label_{i}")
                    seg["inter_ramp_angle"] = st.number_input(
                        "Inter-Ramp Angle (°)", min_value=1.0, max_value=89.0,
                        value=float(seg["inter_ramp_angle"]), step=0.5, key=f"ira_{i}"
                    )
                    seg["bench_face_angle"] = st.number_input(
                        "Bench Face Angle (°)", min_value=1.0, max_value=89.0,
                        value=float(seg["bench_face_angle"]), step=0.5, key=f"bfa_{i}"
                    )
                with col2:
                    seg["bench_height"] = st.number_input(
                        f"Bench Height ({u})", min_value=0.1,
                        value=float(seg["bench_height"]), step=0.5, key=f"bh_{i}"
                    )
                    seg["stack_height"] = st.number_input(
                        f"Stack Height ({u})", min_value=0.1,
                        value=float(seg["stack_height"]), step=0.5, key=f"sh_{i}"
                    )
                    seg["berm_width"] = st.number_input(
                        f"Inter-Stack Berm Width ({u})", min_value=0.0,
                        value=float(seg.get("berm_width", 0.0)), step=0.5, key=f"bw_{i}",
                        help="Wide flat platform inserted at each stack boundary within this segment (0 = none)."
                    )
                    seg["overall_height"] = st.number_input(
                        f"Overall Height ({u})", min_value=0.1,
                        value=float(seg["overall_height"]), step=0.5, key=f"oh_{i}"
                    )
                with col3:
                    if i < len(st.session_state.segments) - 1:
                        seg["road_width"] = st.number_input(
                            f"Ramp Width ({u})", min_value=0.0,
                            value=float(seg.get("road_width") or 20.0),
                            step=0.5, key=f"rw_{i}",
                            help="Flat access ramp to next segment."
                        )
                    if len(st.session_state.segments) > 1:
                        if st.button("🗑 Remove", key=f"del_{i}"):
                            to_delete = i

        if to_delete is not None:
            st.session_state.segments.pop(to_delete)
            st.rerun()

    # ── RIGHT: Live plot ────────────────────────────────────────────────────
    with right_col:
        st.subheader("Preview")
        crest_points = []
        toe_points = []
        setback_points = []
        toe_offset_points = []
        boundary_point = None
        damage_line_pts = None
        try:
            all_vertices, segment_info, warnings, segment_toes, segment_crests = build_geometry(
                st.session_state.segments, direction, toe_width, crest_width,
                depth_below_toe, unit_label=u, toe_anchor=toe_anchor
            )
            for w in warnings:
                st.warning(w)

            total_height = all_vertices[:, 1].max()
            total_width = abs(all_vertices[:, 0].max() - all_vertices[:, 0].min())

            # Overall slope angle: from lowest toe to highest crest
            osa_toe = segment_toes[0]
            osa_crest = segment_crests[-1]
            _dx = abs(osa_crest[0] - osa_toe[0])
            _dy = osa_crest[1] - osa_toe[1]
            overall_slope_angle = np.degrees(np.arctan2(_dy, _dx)) if _dx > 0 else 90.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Vertices", len(all_vertices))
            m2.metric(f"Height ({u})", f"{total_height:.1f}")
            m3.metric(f"Width ({u})", f"{total_width:.1f}")
            m4.metric("Overall Slope (°)", f"{overall_slope_angle:.1f}")

            lines = [f"{x:.4f}\t{y:.4f}" for x, y in all_vertices]
            output_text = "\n".join(lines)
            escaped = output_text.replace("\\", "\\\\").replace("`", "\\`")
            st.components.v1.html(
                f"""
                <button onclick="
                    navigator.clipboard.writeText(`{escaped}`)
                        .then(() => {{ this.innerText = '✅ Copied!'; setTimeout(() => this.innerText = '📋 Copy Coordinates', 2000); }})
                        .catch(() => {{ this.innerText = '❌ Failed'; setTimeout(() => this.innerText = '📋 Copy Coordinates', 2000); }});
                " style="
                    width:100%; height:38px; cursor:pointer;
                    background-color:#262730; color:white;
                    border:1px solid rgba(250,250,250,0.2); border-radius:6px;
                    font-size:14px; font-family:sans-serif;
                ">📋 Copy Coordinates</button>
                """,
                height=50,
            )

            if show_damage:
                crest_points = segment_crests
                all_cp_text = "\n".join(f"{pt[0]:.4f}\t{pt[1]:.4f}" for pt in crest_points)
                cp_escaped = all_cp_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{cp_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Crest Points', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Crest Points', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#1a3a5c; color:white;
                        border:1px solid rgba(100,160,220,0.4); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Crest Points</button>
                    """,
                    height=50,
                )

            if show_toe:
                toe_points = segment_toes
                all_tp_text = "\n".join(f"{pt[0]:.4f}\t{pt[1]:.4f}" for pt in toe_points)
                tp_escaped = all_tp_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{tp_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Toe Points', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Toe Points', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#1a3a5c; color:white;
                        border:1px solid rgba(100,160,220,0.4); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Toe Points</button>
                    """,
                    height=50,
                )

            if show_setback:
                sign = 1 if direction == "right" else -1
                top_crest = segment_crests[-1]
                setback_points = [np.array([top_crest[0] + sign * crest_setback, top_crest[1]])]
                sp_text = f"{setback_points[0][0]:.4f}\t{setback_points[0][1]:.4f}"
                sp_escaped = sp_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{sp_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Crest Setback Point', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Crest Setback Point', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#1a3a5c; color:white;
                        border:1px solid rgba(100,160,220,0.4); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Crest Setback Point</button>
                    """,
                    height=50,
                )

            if show_toe_offset:
                sign = 1 if direction == "right" else -1
                theta_rad = np.radians(overall_slope_angle)
                # Unit vector perpendicular to overall slope, pointing into the slope body
                perp = np.array([sign * np.sin(theta_rad), -np.cos(theta_rad)])
                toe_offset_points = [
                    segment_toes[i] + 0.3 * st.session_state.segments[i]["overall_height"] * perp
                    for i in range(len(segment_toes))
                ]
                all_to_text = "\n".join(f"{pt[0]:.4f}\t{pt[1]:.4f}" for pt in toe_offset_points)
                to_escaped = all_to_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{to_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Toe Offset Points', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Toe Offset Points', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#1a4a2c; color:white;
                        border:1px solid rgba(100,220,140,0.4); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Toe Offset Points</button>
                    """,
                    height=50,
                )

            if show_boundary_pt and toe_offset_points:
                # Same Y as lowest toe offset point, X at the far model edge
                by = float(toe_offset_points[0][1])
                bx = float(all_vertices[:, 0].max()) if direction == "left" else float(all_vertices[:, 0].min())
                boundary_point = np.array([bx, by])
                bp_text = f"{boundary_point[0]:.4f}\t{boundary_point[1]:.4f}"
                bp_escaped = bp_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{bp_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Boundary Point', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Boundary Point', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#3a2a00; color:white;
                        border:1px solid rgba(220,160,50,0.4); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Boundary Point</button>
                    """,
                    height=50,
                )

            if show_damage_line:
                _sign = 1 if direction == "right" else -1
                _theta_rad = np.radians(overall_slope_angle)
                _perp = np.array([_sign * np.sin(_theta_rad), -np.cos(_theta_rad)])
                _dl_setback = np.array([segment_crests[-1][0] + _sign * crest_setback, segment_crests[-1][1]])
                _dl_toe_offsets = [
                    segment_toes[i] + 0.3 * st.session_state.segments[i]["overall_height"] * _perp
                    for i in range(len(segment_toes))
                ]
                _dl_by = float(_dl_toe_offsets[0][1])
                _dl_bx = float(all_vertices[:, 0].max()) if direction == "left" else float(all_vertices[:, 0].min())
                _dl_boundary = np.array([_dl_bx, _dl_by])
                # setback → top-to-bottom toe offsets → boundary
                damage_line_pts = [_dl_setback] + list(reversed(_dl_toe_offsets)) + [_dl_boundary]
                dl_text = "\n".join(f"{pt[0]:.4f}\t{pt[1]:.4f}" for pt in damage_line_pts)
                dl_escaped = dl_text.replace("\\", "\\\\").replace("`", "\\`")
                st.components.v1.html(
                    f"""
                    <button onclick="
                        navigator.clipboard.writeText(`{dl_escaped}`)
                            .then(() => {{ this.innerText = '\u2705 Copied!'; setTimeout(() => this.innerText = '\U0001f4cb Copy Damage Line', 2000); }})
                            .catch(() => {{ this.innerText = '\u274c Failed'; setTimeout(() => this.innerText = '\U0001f4cb Copy Damage Line', 2000); }});
                    " style="
                        width:100%; height:38px; cursor:pointer;
                        background-color:#4a0000; color:white;
                        border:1px solid rgba(220,80,80,0.5); border-radius:6px;
                        font-size:14px; font-family:sans-serif;
                    ">\U0001f4cb Copy Damage Line</button>
                    """,
                    height=50,
                )

            fig = make_figure(all_vertices, segment_info, direction, unit_label=u,
                              crest_points=crest_points, toe_points=toe_points,
                              setback_points=setback_points, toe_offset_points=toe_offset_points,
                              boundary_point=boundary_point, damage_line_pts=damage_line_pts)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except Exception as e:
            st.info(f"Adjust settings to preview geometry. ({e})")
            all_vertices = None

    # ── Output & download (full width below) ───────────────────────────────
    if all_vertices is not None:
        st.divider()
        lines = [f"{x:.4f}\t{y:.4f}" for x, y in all_vertices]
        output_text = "\n".join(lines)

        with st.expander("Vertex Coordinates"):
            st.code(output_text, language=None)

        st.download_button(
            label="⬇ Download Vertices (.txt)",
            data=output_text.encode(),
            file_name="slope_vertices.txt",
            mime="text/plain",
        )
        if show_damage and crest_points:
            all_cp_text = "\n".join(
                f"{info['label']}\t{pt[0]:.4f}\t{pt[1]:.4f}"
                for info, pt in zip(segment_info, crest_points)
            )
            with st.expander("Crest Point Coordinates"):
                st.code(all_cp_text, language=None)
            st.download_button(
                label="\u2b07 Download Crest Points (.txt)",
                data=all_cp_text.encode(),
                file_name="crest_points.txt",
                mime="text/plain",
                key="dl_crest",
            )
        if show_toe and toe_points:
            all_tp_text = "\n".join(
                f"{info['label']}\t{pt[0]:.4f}\t{pt[1]:.4f}"
                for info, pt in zip(segment_info, toe_points)
            )
            with st.expander("Toe Point Coordinates"):
                st.code(all_tp_text, language=None)
            st.download_button(
                label="\u2b07 Download Toe Points (.txt)",
                data=all_tp_text.encode(),
                file_name="toe_points.txt",
                mime="text/plain",
                key="dl_toe",
            )
        if show_setback and setback_points:
            sp_text = f"{setback_points[0][0]:.4f}\t{setback_points[0][1]:.4f}"
            with st.expander("Crest Setback Point Coordinates"):
                st.code(sp_text, language=None)
            st.download_button(
                label="\u2b07 Download Crest Setback Point (.txt)",
                data=sp_text.encode(),
                file_name="crest_setback_point.txt",
                mime="text/plain",
                key="dl_setback",
            )
        if show_toe_offset and toe_offset_points:
            all_to_text = "\n".join(
                f"{info['label']}\t{pt[0]:.4f}\t{pt[1]:.4f}"
                for info, pt in zip(segment_info, toe_offset_points)
            )
            with st.expander("Toe Offset Point Coordinates"):
                st.code(all_to_text, language=None)
            st.download_button(
                label="\u2b07 Download Toe Offset Points (.txt)",
                data=all_to_text.encode(),
                file_name="toe_offset_points.txt",
                mime="text/plain",
                key="dl_toe_offset",
            )
        if show_boundary_pt and boundary_point is not None:
            bp_text = f"{boundary_point[0]:.4f}\t{boundary_point[1]:.4f}"
            with st.expander("Boundary Point Coordinates"):
                st.code(bp_text, language=None)
            st.download_button(
                label="\u2b07 Download Boundary Point (.txt)",
                data=bp_text.encode(),
                file_name="boundary_point.txt",
                mime="text/plain",
                key="dl_boundary",
            )
        if show_damage_line and damage_line_pts:
            dl_text = "\n".join(f"{pt[0]:.4f}\t{pt[1]:.4f}" for pt in damage_line_pts)
            with st.expander("Damage Line Coordinates"):
                st.code(dl_text, language=None)
            st.download_button(
                label="\u2b07 Download Damage Line (.txt)",
                data=dl_text.encode(),
                file_name="damage_line.txt",
                mime="text/plain",
                key="dl_damage_line",
            )

if __name__ == "__main__":
    main()
