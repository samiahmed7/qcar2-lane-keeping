#!/usr/bin/env python3
"""
Generate an oval QCar2 track SDF.

Defaults produce the "normal" track. Pass --help to see all flags.
Example for a wider-lane variant:
    python3 generate_track.py --lo 0.55 --tw 1.6 > worlds/qcar_oval_wide.sdf
"""
import argparse
import math

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--world-name',   default='qcar_oval',
                help='SDF <world name=...> attribute')
ap.add_argument('--s', type=float, default=4.0,
                help='half-length of each straight (m). Total straight = 2·S.')
ap.add_argument('--r', type=float, default=2.0,
                help='corner centreline radius (m)')
ap.add_argument('--tw', type=float, default=1.6,
                help='total track width wall-to-wall (m)')
ap.add_argument('--lo', type=float, default=0.40,
                help='lane-line offset from centreline (each side, m). '
                     'Space between the two solid lane lines = 2·LO.')
ap.add_argument('--dash-len', type=float, default=0.30,
                help='length of each centre dash (m)')
ap.add_argument('--dash-period', type=float, default=0.45,
                help='spacing between successive dash starts (m)')
ap.add_argument('--n', type=int, default=10,
                help='segments per 90° corner (more = smoother)')
args = ap.parse_args()

# ── Track parameters ──────────────────────────────────────────────────────────
S   = args.s          # half-length of each straight
R   = args.r          # corner centreline radius
TW  = args.tw         # total track width (wall-to-wall)
HW  = TW / 2          # half-width
LO  = args.lo         # lane-line offset from centreline (each side)
DASH_LEN    = args.dash_len
DASH_PERIOD = args.dash_period
N   = args.n
D   = (math.pi / 2) / N   # radians per segment

if LO >= HW:
    raise SystemExit(
        f'Invalid geometry: lane-line offset {LO} m >= track half-width {HW} m. '
        f'Increase --tw or decrease --lo.'
    )

# ── Colours ───────────────────────────────────────────────────────────────────
ROAD  = "0.12 0.12 0.12 1"
WALL  = "0.9 0.35 0.0 1"
WHITE = "1 1 1 1"
GREY  = "0.3 0.3 0.3 1"

# ── Model accumulator ─────────────────────────────────────────────────────────
_models = []
_idx = [0]

def _box(px, py, pz, yaw, sx, sy, sz, col, walls=False):
    n = f"m{_idx[0]}"; _idx[0] += 1
    coll = (f"\n        <collision name='c'><geometry><box>"
            f"<size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry></collision>"
            if walls else "")
    _models.append(
        f"    <model name='{n}'><static>true</static>\n"
        f"      <pose>{px:.4f} {py:.4f} {pz:.5f} 0 0 {yaw:.5f}</pose>\n"
        f"      <link name='l'>{coll}\n"
        f"        <visual name='v'><geometry><box>"
        f"<size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>\n"
        f"          <material><ambient>{col}</ambient>"
        f"<diffuse>{col}</diffuse></material>\n"
        f"        </visual></link></model>"
    )

# ── Straight sections ─────────────────────────────────────────────────────────
# Each straight is extended by 0.3 m on each side to cover the corner junction.
SL  = 2 * S + 0.6  # 8.6 m effective box length (avoids gaps at corners)

straights = [
    (0.0,    -(S+R), False),   # bottom  (travels +X)
    (0.0,    +(S+R), False),   # top     (travels −X)
    (+(S+R),  0.0,   True ),   # right   (travels +Y)
    (-(S+R),  0.0,   True ),   # left    (travels −Y)
]

for cx, cy, vert in straights:
    sx_r, sy_r = (SL, TW) if not vert else (TW, SL)

    # Road surface
    _box(cx, cy, 0.0010, 0, sx_r, sy_r, 0.001, ROAD)

    # sign > 0 means the coordinate is positive → outer side is away from 0
    sgn = math.copysign(1, cy if not vert else cx)

    if not vert:   # horizontal straight — offsets in Y
        iw = cy - sgn * HW         # inner wall y  (toward 0)
        ow = cy + sgn * HW         # outer wall y
        il = cy - sgn * LO         # inner lane line y
        ol = cy + sgn * LO         # outer lane line y
        # Walls
        _box(cx, iw, 0.040, 0, SL, 0.05, 0.08, WALL)
        _box(cx, ow, 0.040, 0, SL, 0.05, 0.08, WALL)
        # Lane lines
        _box(cx, il, 0.002, 0, SL, 0.06, 0.001, WHITE)
        _box(cx, ol, 0.002, 0, SL, 0.06, 0.001, WHITE)
        # Centre dashes along X
        dx = -S + DASH_LEN / 2
        while dx <= S - DASH_LEN / 2:
            _box(cx + dx, cy, 0.002, 0, DASH_LEN, 0.06, 0.001, WHITE)
            dx += DASH_PERIOD
    else:           # vertical straight — offsets in X
        iw = cx - sgn * HW
        ow = cx + sgn * HW
        il = cx - sgn * LO
        ol = cx + sgn * LO
        # Walls
        _box(iw, cy, 0.040, 0, 0.05, SL, 0.08, WALL)
        _box(ow, cy, 0.040, 0, 0.05, SL, 0.08, WALL)
        # Lane lines
        _box(il, cy, 0.002, 0, 0.06, SL, 0.001, WHITE)
        _box(ol, cy, 0.002, 0, 0.06, SL, 0.001, WHITE)
        # Centre dashes along Y
        dy = -S + DASH_LEN / 2
        while dy <= S - DASH_LEN / 2:
            _box(cx, cy + dy, 0.002, 0, 0.06, DASH_LEN, 0.001, WHITE)
            dy += DASH_PERIOD

# ── Curved corners ────────────────────────────────────────────────────────────
# Corner arc centres at (±S, ±S) = (±4, ±4)
corners = [
    ( S, -S, -math.pi/2),   # bottom-right
    ( S,  S,  0.0       ),   # top-right
    (-S,  S,  math.pi/2 ),   # top-left
    (-S, -S,  math.pi   ),   # bottom-left
]

OVL = 1.12   # overlap factor for arc segments (avoids inter-segment gaps)

# How many corner segments make up one centre-dash period?
# Arc length per segment = R · D. Round to nearest integer ≥ 1.
ARC_PER_SEG = R * D
SEGS_PER_DASH = max(1, int(round(DASH_PERIOD / ARC_PER_SEG)))

for cx, cy, a0 in corners:
    for i in range(N):
        th  = a0 + (i + 0.5) * D
        yaw = math.pi / 2 + th

        def seg(r):
            return r * D * OVL   # arc-segment length at radius r

        def pt(r):
            return cx + r * math.cos(th), cy + r * math.sin(th)

        # Road surface
        px, py = pt(R)
        _box(px, py, 0.0010, yaw, seg(R), TW,   0.001, ROAD)

        # Inner wall  (R − HW = 1.4)
        ri = R - HW
        px, py = pt(ri)
        _box(px, py, 0.040, yaw, seg(ri), 0.05, 0.08, WALL)

        # Outer wall  (R + HW = 2.6)
        ro = R + HW
        px, py = pt(ro)
        _box(px, py, 0.040, yaw, seg(ro), 0.05, 0.08, WALL)

        # Inner lane line  (R − LO)
        ril = R - LO
        px, py = pt(ril)
        _box(px, py, 0.002, yaw, seg(ril), 0.06, 0.001, WHITE)

        # Outer lane line  (R + LO)
        rol = R + LO
        px, py = pt(rol)
        _box(px, py, 0.002, yaw, seg(rol), 0.06, 0.001, WHITE)

        # Centre dashes — placed every SEGS_PER_DASH segments so the period
        # along the arc roughly matches DASH_PERIOD on the straight sections.
        if i % SEGS_PER_DASH == 0:
            px, py = pt(R)
            # Cap dash length so it doesn't exceed the available segment(s)
            dash_arc = min(DASH_LEN, ARC_PER_SEG * SEGS_PER_DASH * 0.8)
            _box(px, py, 0.002, yaw, dash_arc, 0.06, 0.001, WHITE)

# ── Assemble SDF ──────────────────────────────────────────────────────────────
header = f"""\
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="{args.world_name}">

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
      <attenuation>
        <range>1000</range><constant>0.9</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.3 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

footer = """\
  </world>
</sdf>
"""

out = [header]
for m in _models:
    out.append(m)
out.append(footer)
print("\n".join(out))
