#!/usr/bin/env python3
"""Generate a Gazebo world (SDF) that SHOWS the university track.

Reads university_loop.npy + branch_in.npy and writes
qcar2_worlds/worlds/university_track.sdf:
  * a dark road ribbon along the loop + branch,
  * SOLID white edge lines down both sides of every road,
  * a dashed white centre line,
  * a real ROUNDABOUT (circular ring road + central island) at the top-right
    corner, and
  * a START marker.
Same static-box style as the existing lab_track.sdf, so it loads the same way.

    colcon build --packages-select qcar2_worlds --symlink-install   # once
    python3 scripts/make_world.py                                   # to regenerate
    ros2 launch qcar2_bringup sim_bringup.launch.py world:=university_track \\
        x:=0.0 y:=0.0 yaw:=0.0 headless:=false
"""
import pathlib
import numpy as np

ROAD_W = 1.0          # road width (m)
HALF = ROAD_W / 2.0
LINE_W = 0.10         # solid edge-line width (m) -- visible to the BEV, not blob-prone
CENTER_LINE_W = 0.08  # dashed centre-line width (m)
ROAD_Z = 0.002        # road surface height off the ground
LINE_Z = ROAD_Z + 0.003
SEG = 0.25            # box spacing (m)
RB_C = (4.0, 9.0)     # roundabout centre (top-right corner; track re-origined so START=(0,0))
RB_R = 1.0            # roundabout ring centreline radius (= corner radius)
RB_ISLAND = 0.35      # central island radius; leaves clear space inside lane line
RB_RING_SEGMENTS = 192
RB_OPENING_MARGIN = 0.04
RB_OPENING_FALLBACK_HALF_ANGLE = 0.42
RB_CENTER_JOIN_INSET = 0.005
RB_DASH_COUNT = 24
RB_DASH_FILL = 0.45

GREY = (0.15, 0.15, 0.15)
WHITE = (1.0, 1.0, 1.0)
GREEN = (0.20, 0.55, 0.20)
BLUE = (0.10, 0.50, 0.90)

HEADER = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="university_track">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system"           name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"     name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"           name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system"               name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system"           name="gz::sim::systems::Contact"/>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.3 0.3 0.3 1</specular>
      <attenuation><range>1000</range><constant>0.9</constant><linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
      <direction>-0.3 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.25 0.40 0.25 1</ambient><diffuse>0.25 0.40 0.25 1</diffuse></material></visual>
      </link>
    </model>
"""
FOOTER = "  </world>\n</sdf>\n"


def box(name, x, y, z, yaw, sx, sy, sz, rgb):
    r, g, b = rgb
    return (f'    <model name="{name}"><static>true</static>'
            f'<pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.5f}</pose>'
            f'<link name="l"><visual name="v"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>'
            f'</visual></link></model>\n')


def cylinder(name, x, y, z, radius, h, rgb, collision=False):
    r, g, b = rgb
    collision_xml = ""
    if collision:
        collision_xml = (
            f'<collision name="collision"><geometry><cylinder><radius>{radius:.3f}</radius>'
            f'<length>{h:.3f}</length></cylinder></geometry></collision>'
        )
    return (f'    <model name="{name}"><static>true</static>'
            f'<pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>'
            f'<link name="l">{collision_xml}<visual name="v"><geometry><cylinder><radius>{radius:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material>'
            f'</visual></link></model>\n')


def _dash_ranges(count=18, fill=0.42):
    ranges = []
    step = 2.0 * np.pi / float(count)
    dash = step * float(fill)
    for i in range(count):
        start = i * step
        ranges.append((start, start + dash))
    return ranges


def _angle_diff(a, b):
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _in_angle_gaps(angle, gaps):
    return any(abs(_angle_diff(angle, center)) <= half_width
               for center, half_width in gaps)


def _resample(xy, spacing=SEG):
    d = np.hypot(np.diff(xy[0]), np.diff(xy[1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    t = np.arange(0.0, s[-1], spacing)
    return np.interp(t, s, xy[0]), np.interp(t, s, xy[1])


def _offset(xs, ys, d):
    """Shift a polyline sideways by d (left normal) -- for the edge lines."""
    dx, dy = np.gradient(xs), np.gradient(ys)
    n = np.hypot(dx, dy)
    n[n == 0] = 1.0
    return xs + d * (-dy / n), ys + d * (dx / n)


def _segment_parts_outside_circle(x0, y0, x1, y1, avoid):
    if avoid is None:
        return [(x0, y0, x1, y1)]

    cx, cy, r = avoid
    px, py = x0 - cx, y0 - cy
    vx, vy = x1 - x0, y1 - y0
    a = vx * vx + vy * vy
    if a <= 1e-12:
        return []

    b = 2.0 * (px * vx + py * vy)
    c = px * px + py * py - r * r
    roots = []
    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        q = np.sqrt(max(0.0, disc))
        for t in ((-b - q) / (2.0 * a), (-b + q) / (2.0 * a)):
            if 1e-6 < t < 1.0 - 1e-6:
                roots.append(float(t))

    cuts = [0.0] + sorted(set(round(t, 8) for t in roots)) + [1.0]
    parts = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if hi - lo < 1e-5:
            continue
        mid = 0.5 * (lo + hi)
        mx, my = x0 + mid * vx, y0 + mid * vy
        if np.hypot(mx - cx, my - cy) >= r - 1e-6:
            ax, ay = x0 + lo * vx, y0 + lo * vy
            bx, by = x0 + hi * vx, y0 + hi * vy
            parts.append((ax, ay, bx, by))
    return parts


def _strip(xs, ys, prefix, width, rgb, z, stride=1, length_scale=1.4, avoid=None):
    out = ""
    for i in range(0, len(xs) - 1, stride):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        for j, (ax, ay, bx, by) in enumerate(
            _segment_parts_outside_circle(x0, y0, x1, y1, avoid)
        ):
            length_raw = float(np.hypot(bx - ax, by - ay))
            if length_raw < 0.02:
                continue
            mx, my = (ax + bx) / 2, (ay + by) / 2
            yaw = float(np.arctan2(by - ay, bx - ax))
            length = length_raw * length_scale + 0.02
            out += box(f"{prefix}_{i}_{j}", mx, my, z, yaw, length, width, 0.002, rgb)
    return out


def _strip_in_annulus(xs, ys, prefix, width, rgb, z, r_min, r_max,
                      stride=1, length_scale=1.0):
    cx, cy = RB_C
    out = ""
    for i in range(0, len(xs) - 1, stride):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        d = np.hypot(mx - cx, my - cy)
        if d < r_min or d > r_max:
            continue
        length_raw = float(np.hypot(x1 - x0, y1 - y0))
        if length_raw < 0.02:
            continue
        yaw = float(np.arctan2(y1 - y0, x1 - x0))
        length = length_raw * length_scale + 0.02
        out += box(f"{prefix}_{i}", mx, my, z, yaw, length, width, 0.002, rgb)
    return out


def _polyline_circle_crossings(xs, ys, cx, cy, r):
    angles = []
    for i in range(len(xs) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        px, py = x0 - cx, y0 - cy
        vx, vy = x1 - x0, y1 - y0
        a = vx * vx + vy * vy
        if a <= 1e-12:
            continue
        b = 2.0 * (px * vx + py * vy)
        c = px * px + py * py - r * r
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue
        q = np.sqrt(max(0.0, disc))
        for t in ((-b - q) / (2.0 * a), (-b + q) / (2.0 * a)):
            if 0.0 <= t <= 1.0:
                x = x0 + t * vx
                y = y0 + t * vy
                angle = float(np.arctan2(y - cy, x - cx))
                if not any(abs(_angle_diff(angle, old)) < 0.08 for old in angles):
                    angles.append(angle)
    return angles


def _roundabout_opening_gaps(*paths):
    cx, cy, r = RB_ZONE
    edge_offset = HALF - 0.5 * LINE_W
    gaps = []
    for xy in paths:
        xs, ys = _resample(xy)
        center_angles = _polyline_circle_crossings(xs, ys, cx, cy, r)
        edge_angles = []
        for offset in (edge_offset, -edge_offset):
            ex, ey = _offset(xs, ys, offset)
            edge_angles.extend(_polyline_circle_crossings(ex, ey, cx, cy, r))

        for center_angle in center_angles:
            diffs = [
                _angle_diff(edge_angle, center_angle)
                for edge_angle in edge_angles
                if abs(_angle_diff(edge_angle, center_angle)) < 1.1
            ]
            if len(diffs) >= 2:
                lo, hi = min(diffs), max(diffs)
                gap_center = center_angle + 0.5 * (lo + hi)
                gap_half_width = 0.5 * (hi - lo) + RB_OPENING_MARGIN
            else:
                gap_center = center_angle
                gap_half_width = RB_OPENING_FALLBACK_HALF_ANGLE

            if not any(abs(_angle_diff(gap_center, old_center)) < 0.15
                       for old_center, _ in gaps):
                gaps.append((gap_center, gap_half_width))
    return gaps


# nothing else may draw road/markings inside this circle -- the roundabout owns it
RB_ZONE = (RB_C[0], RB_C[1], RB_R + HALF - 0.5 * LINE_W)


def road(xy, prefix):
    """Grey road + solid white edges on both sides + dashed centre line.

    Everything is clipped out of the roundabout zone so the straight-road
    markings stop at the circle instead of crossing through it.
    """
    xs, ys = _resample(xy)
    out = _strip(xs, ys, prefix, ROAD_W, GREY, ROAD_Z, avoid=RB_ZONE)
    edge_offset = HALF - 0.5 * LINE_W
    lx, ly = _offset(xs, ys, +edge_offset)
    rx, ry = _offset(xs, ys, -edge_offset)
    out += _strip(lx, ly, prefix + "_le", LINE_W, WHITE, LINE_Z, avoid=RB_ZONE)
    out += _strip(rx, ry, prefix + "_re", LINE_W, WHITE, LINE_Z, avoid=RB_ZONE)
    out += _strip(xs, ys, prefix + "_cd", CENTER_LINE_W, WHITE, LINE_Z,
                  stride=3, length_scale=1.0, avoid=RB_ZONE)
    out += _strip_in_annulus(xs, ys, prefix + "_rb_cd_join",
                             CENTER_LINE_W, WHITE, LINE_Z + 0.001,
                             RB_R + RB_CENTER_JOIN_INSET,
                             RB_ZONE[2] - 0.02)
    return out


def circle(cx, cy, r):
    n = max(8, int(2 * np.pi * r / SEG))
    th = np.linspace(0.0, 2 * np.pi, n + 1)
    return cx + r * np.cos(th), cy + r * np.sin(th)


def _circle_strip(prefix, radius, width, rgb, z, segments=RB_RING_SEGMENTS,
                  dashed=False, dash_count=RB_DASH_COUNT,
                  dash_fill=RB_DASH_FILL, gaps=None):
    """Draw a circular line as many short, Gazebo-colored box primitives."""
    cx, cy = RB_C
    out = ""
    for i in range(segments):
        a0 = 2.0 * np.pi * i / segments
        a1 = 2.0 * np.pi * (i + 1) / segments
        am = 0.5 * (a0 + a1)
        if gaps and _in_angle_gaps(am, gaps):
            continue
        if dashed:
            phase = (am % (2.0 * np.pi / dash_count)) / (2.0 * np.pi / dash_count)
            if phase > dash_fill:
                continue
        x0, y0 = cx + radius * np.cos(a0), cy + radius * np.sin(a0)
        x1, y1 = cx + radius * np.cos(a1), cy + radius * np.sin(a1)
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        yaw = float(am + 0.5 * np.pi)
        length = float(np.hypot(x1 - x0, y1 - y0) * 1.08 + 0.003)
        out += box(f"{prefix}_{i}", mx, my, z, yaw, length, width, 0.002, rgb)
    return out


def roundabout(opening_gaps):
    cx, cy = RB_C
    # Native cylinders/boxes keep Gazebo's material colors reliable. The green
    # centre disk masks the road disk, visually forming a clean annulus.
    out = cylinder("roundabout_road", cx, cy, ROAD_Z, RB_R + HALF, 0.002, GREY)
    out += cylinder("roundabout_inner_green", cx, cy, ROAD_Z + 0.002,
                    RB_R - HALF, 0.002, GREEN)
    out += _circle_strip(
        "roundabout_inner_line",
        RB_R - HALF + 0.5 * LINE_W,
        LINE_W,
        WHITE,
        LINE_Z,
    )
    out += _circle_strip(
        "roundabout_outer_line",
        RB_R + HALF - 0.5 * LINE_W,
        LINE_W,
        WHITE,
        LINE_Z,
        gaps=opening_gaps,
    )
    out += _circle_strip(
        "roundabout_center_dash",
        RB_R,
        CENTER_LINE_W,
        WHITE,
        LINE_Z + 0.001,
        dashed=True,
    )
    out += cylinder("roundabout_island", cx, cy, 0.05, RB_ISLAND, 0.10, GREEN, collision=True)
    # give-way line across the entry (where the right straight meets the ring)
    ey = cy - float(np.sqrt((RB_R + HALF - 0.5 * LINE_W) ** 2 - RB_R ** 2))
    for j, xx in enumerate(np.linspace(cx + RB_R - HALF + 0.10, cx + RB_R + HALF - 0.10, 6)):
        out += box(f"giveway_{j}", float(xx), ey, LINE_Z, 0.0,
                   0.10, 0.06, 0.002, WHITE)
    return out


def main():
    ws = pathlib.Path.home() / "rosbot_ws"
    loop = np.load(ws / "university_loop.npy")
    branch = np.load(ws / "branch_in.npy")
    pkg_worlds = ws / "src/qcar2_autonomous_lanes/qcar2_worlds"

    opening_gaps = _roundabout_opening_gaps(loop, branch)
    body = road(loop, "loop") + road(branch, "branch") + roundabout(opening_gaps)
    body += box("start_marker", float(branch[0, 0]), float(branch[1, 0]), 0.01,
                0.0, 0.4, 0.4, 0.02, BLUE)

    sdf = HEADER + body + FOOTER
    out = pkg_worlds / "worlds/university_track.sdf"
    out.write_text(sdf)
    print(f"wrote {out}\n  {len(sdf)} bytes, {sdf.count('<model ')} models")

    # ---- top-down preview ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#bfe3bf")
    cx, cy = RB_C

    def mask_zone(xs, ys):
        xs, ys = xs.astype(float).copy(), ys.astype(float).copy()
        m = np.hypot(xs - cx, ys - cy) < (RB_R + HALF)
        xs[m] = np.nan
        ys[m] = np.nan
        return xs, ys

    for xy in (loop, branch):
        xs, ys = _resample(xy)
        ax.plot(*mask_zone(xs, ys), "-", lw=16, color="#444444",
                solid_capstyle="round")
        for d in (HALF - 0.5 * LINE_W, -HALF + 0.5 * LINE_W):
            ax.plot(*mask_zone(*_offset(xs, ys, d)), "-", lw=1.6, color="white")
        ax.plot(*mask_zone(xs, ys), "--", lw=1.2, color="white", dashes=(5, 6))
    ax.plot(*circle(cx, cy, RB_R), "-", lw=16, color="#444444",
            solid_capstyle="round")
    ax.plot(*circle(cx, cy, RB_R - HALF + LINE_W), "-", lw=1.6, color="white")

    outer_r = RB_R + HALF - LINE_W
    th = np.linspace(0.0, 2.0 * np.pi, 720)
    outer_x = cx + outer_r * np.cos(th)
    outer_y = cy + outer_r * np.sin(th)
    outer_mask = np.array([not _in_angle_gaps(angle, opening_gaps) for angle in th])
    outer_x[~outer_mask] = np.nan
    outer_y[~outer_mask] = np.nan
    ax.plot(outer_x, outer_y, "-", lw=1.6, color="white")

    for xy in (loop, branch):
        xs, ys = _resample(xy)
        cx0, cy0 = RB_C
        d = np.hypot(xs - cx0, ys - cy0)
        m = (d >= RB_R + RB_CENTER_JOIN_INSET) & (d <= RB_ZONE[2] - 0.02)
        jx, jy = xs.astype(float).copy(), ys.astype(float).copy()
        jx[~m] = np.nan
        jy[~m] = np.nan
        ax.plot(jx, jy, "-", lw=1.2, color="white")

    for start, stop in _dash_ranges(RB_DASH_COUNT, RB_DASH_FILL):
        th = np.linspace(start, stop, 20)
        ax.plot(cx + RB_R * np.cos(th), cy + RB_R * np.sin(th),
                "-", lw=1.2, color="white")
    ax.add_patch(plt.Circle((cx, cy), RB_ISLAND, color="#2e8b2e"))
    _ey = cy - np.sqrt((RB_R + HALF - 0.5 * LINE_W) ** 2 - RB_R ** 2)
    ax.plot(np.linspace(cx + RB_R - HALF + 0.10, cx + RB_R + HALF - 0.10, 6),
            [_ey] * 6, "s", color="white", ms=3)
    ax.annotate("roundabout", (cx, cy), color="white", fontweight="bold",
                ha="center", va="center")
    ax.plot(branch[0, 0], branch[1, 0], "s", color="#1f78ff", ms=14)
    ax.annotate("START", (branch[0, 0], branch[1, 0]), color="#1f78ff",
                fontweight="bold", xytext=(8, 8), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_title("university_track.sdf — roundabout + white edge lines")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(ws / "world_preview.png", dpi=110)
    print("saved world_preview.png")


if __name__ == "__main__":
    main()
