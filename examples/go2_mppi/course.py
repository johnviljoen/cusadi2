"""The parkour course and the eight robots on it: one analytic height function h(x, y), written once
against a tiny backend (CasADi for the rollout function, NumPy for the terrain mesh the video shows).

Radial "summit" layout: a raised square platform in the middle, reached from four sides by

    east  (+x)  a staircase
    north (+y)  a tall (0.2 m) step onto a box, then a 0.1 m step
    west  (-x)  a ramp up to an outer platform, then a gap to the summit
    south (-y)  a ramp

Every vertical face is a tanh step of scale EDGE, so h is smooth and the contact normal of the
penalty model is its gradient. Sizes that had to shrink for MPPI to cross them are noted in the README.
"""
import os

import numpy as np

EDGE = 0.012          # tanh scale of a riser [m] (10-90 % of the rise over 2.2 * EDGE)
S = 1.2               # summit half-size
LANE = 0.9            # half-width of every approach
H_SUMMIT = 0.40
STAIR_N, STAIR_TREAD = 4, 0.30                 # east: STAIR_N risers of H_SUMMIT / STAIR_N
BOX_H, BOX_LEN = 0.20, 1.10                    # north: a tall box ...
NSTEP_H, NSTEP_LEN = 0.30, 0.30                # ... then one 0.1 m step up to the summit (see README: shrunk)
GAP = 0.20                                     # west: gap width (see README: shrunk from 0.4)
PLAT_LEN, WRAMP_LEN = 1.2, 1.5                 # west: outer platform and the ramp up to it
SRAMP_LEN = 1.6                                # south: ramp length
for _kv in os.environ.get("GO2_COURSE", "").split():   # experiments: GO2_COURSE="BOX_H=0.25 GAP=0.3" overrides the sizes
    _k, _v = _kv.split("=")
    assert _k in globals() and _k.isupper(), _k
    globals()[_k] = float(_v)


class NP:
    tanh, fmin, fmax = staticmethod(np.tanh), staticmethod(np.minimum), staticmethod(np.maximum)


def height(x, y, xp=NP):
    """Terrain height at (x, y); xp provides tanh, fmin, fmax (NumPy arrays or CasADi SX)."""
    up = lambda u: 0.5 + 0.5 * xp.tanh(u / EDGE)                    # 0 -> 1 across u = 0
    box = lambda x0, x1, y0, y1: up(x - x0) * up(x1 - x) * up(y - y0) * up(y1 - y)
    clamp01 = lambda u: xp.fmin(xp.fmax(u, 0.0), 1.0)
    h = H_SUMMIT * box(-S, S, -S, S)
    rise = H_SUMMIT / STAIR_N
    for i in range(1, STAIR_N):                                     # east stairs, highest tread next to the summit
        h = xp.fmax(h, (H_SUMMIT - i * rise) * box(S - 0.1, S + i * STAIR_TREAD, -LANE, LANE))
    h = xp.fmax(h, BOX_H * box(-LANE, LANE, S - 0.1, S + BOX_LEN))  # north box and step
    h = xp.fmax(h, NSTEP_H * box(-LANE, LANE, S - 0.1, S + NSTEP_LEN))
    x_gap = -S - GAP                                                # west: outer platform ends here
    x_plat = x_gap - PLAT_LEN
    h = xp.fmax(h, H_SUMMIT * clamp01((x - (x_plat - WRAMP_LEN)) / WRAMP_LEN) * up(x_gap - x) * up(y + LANE) * up(LANE - y))
    h = xp.fmax(h, H_SUMMIT * clamp01((y + S + SRAMP_LEN) / SRAMP_LEN) * up(-S + 0.1 - y) * up(x + LANE) * up(LANE - x))
    return h


def robots():
    """Start pose and parking spot of each robot: (x0, y0, yaw0, goal_x, goal_y, goal_yaw).
    Two robots per approach; the goal heading faces the centre of the summit."""
    R_SPOT, LAT = 0.80, 0.32
    out = []
    #        side yaw-in  start distance, lateral, heading error  (two robots)
    sides = [(0.0,          ((4.3, 0.45, 0.35), (4.9, -0.30, -0.5))),      # east, walking -x
             (np.pi / 2,    ((3.5, -0.40, -0.4), (4.1, 0.35, 0.6))),       # north
             (np.pi,        ((5.3, 0.30, 0.5), (5.9, -0.45, -0.3))),       # west
             (-np.pi / 2,   ((4.0, -0.35, 0.45), (4.6, 0.40, -0.6)))]      # south
    for ang, pair in sides:
        c, s = np.cos(ang), np.sin(ang)
        for k, (dist, lat, dyaw) in enumerate(pair):
            spot_lat = LAT if k == 0 else -LAT
            rot = lambda r, l: (c * r - s * l, s * r + c * l)
            x0, y0 = rot(dist, lat)
            gx, gy = rot(R_SPOT, spot_lat)
            yaw_in = np.arctan2(-s, -c)                              # facing the centre
            out.append((x0, y0, yaw_in + dyaw, gx, gy, yaw_in))
    return np.array(out)


def flat_robots(n=8, same=False):
    """Milestone 0: eight robots far from the course, walking 3 m on flat ground in eight directions
    (`same`: all in one direction, for parameter sweeps)."""
    out = []
    for i in range(n):
        ang = 0.0 if same else 2 * np.pi * i / n
        x0, y0 = 20.0 + 3.0 * (i % 4), 20.0 + 3.0 * (i // 4)
        out.append((x0, y0, ang + 0.4, x0 + 3.0 * np.cos(ang), y0 + 3.0 * np.sin(ang), ang))
    return np.array(out)


def side_robots(side, n=8):
    """One approach in isolation: n copies of a robot starting just outside that feature (robots do
    not interact, so they can share a start), for per-robot parameter sweeps."""
    ang, dist = {"east": (0.0, 2.9), "north": (np.pi / 2, 3.1), "west": (np.pi, 4.9), "south": (-np.pi / 2, 3.4)}[side]
    c, s = np.cos(ang), np.sin(ang)
    yaw_in = np.arctan2(-s, -c)
    return np.array([(c * dist - s * 0.3, s * dist + c * 0.3, yaw_in, c * 0.8 - s * 0.3, s * 0.8 + c * 0.3, yaw_in)] * n)
