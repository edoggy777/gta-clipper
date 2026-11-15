#!/usr/bin/env python3
"""eases.py - 12 measured easing curves, usable from Python and inside ffmpeg filters.

An ease turns "how far through the move am I" (0 to 1) into "how far has it travelled"
(0 to 1). A straight line looks robotic; these curves were fitted to real motion design.

The curve maths and frame counts are ported from the motionmaxxing project
(https://github.com/Tejashmakwana/motionmaxxing, runtime/motion.js, Apache-2.0).

  ease("softLand", 0.5)       -> 0.84   (plain Python number)
  expr("softLand", "t/0.4")   -> text you can paste into an ffmpeg expression
  seconds("softLand")         -> 0.4    (how long the curve is meant to last)

Run this file to print a table of every curve:
  python3 eases.py
"""
import math

# name: (kind, numbers, frames at 30 fps, what it is for)
CURVES = {
    "crashIn":    ("snapOver", (2.0, 0.19, 0.218), 10, "hits and slightly passes its mark at once"),
    "popOver":    ("spring",   (2.67, 4.36),       15, "slow, late overshoot"),
    "bounceHard": ("spring",   (1.34, 5.07),       22, "big overshoot, use sparingly"),
    "snapSettle": ("snapRamp", (0.942, 0.118),     11, "over half the move in the first tenth of the time"),
    "softLand":   ("expoOut",  (3.41,),            12, "firm slow-down, the workhorse for text"),
    "whip":       ("powerOut", (2.52,),             7, "hard slow-down on a big move"),
    "glide":      ("powerOut", (1.7,),             10, "gentle slow-down"),
    "cruise":     ("linear",   (),                  9, "straight line: fades and slow drifts"),
    "softInOut":  ("smooth",   (1.42,),             9, "even S-curve for fades and blur"),
    "gentleIn":   ("powerIn",  (2.07,),             9, "mild speed-up"),
    "accelExit":  ("powerIn",  (2.6,),             13, "the leaving curve"),
    "crashOut":   ("powerIn",  (3.97,),            17, "hard speed-up into a cut"),
}
NAMES = list(CURVES)


def _raw(kind, p):
    """The curve before it is scaled to end exactly on 1."""
    if kind == "linear":
        return lambda t: t
    if kind == "powerOut":
        return lambda t: 1 - (1 - t) ** p[0]
    if kind == "powerIn":
        return lambda t: t ** p[0]
    if kind == "expoOut":
        return lambda t: 1 - math.exp(-p[0] * t)
    if kind == "snapRamp":
        w, tau = p
        return lambda t: w * (1 - math.exp(-t / tau)) + (1 - w) * t
    if kind == "snapOver":
        a, ta, td = p
        return lambda t: (1 + a * math.exp(-t / td)) * (1 - math.exp(-t / ta))
    if kind == "spring":
        c, om = p
        return lambda t: 1 - math.exp(-c * t) * (math.cos(om * t) + (c / om) * math.sin(om * t))
    if kind == "smooth":
        return lambda t: t ** p[0] / (t ** p[0] + (1 - t) ** p[0])
    raise ValueError(kind)


def ease(name, t):
    """Value of the curve at t. t below 0 gives 0, t above 1 gives 1."""
    kind, p, _, _ = CURVES[name]
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    f = _raw(kind, p)
    return f(t) / f(1.0)


def seconds(name):
    """How long the curve is meant to last, in seconds."""
    return CURVES[name][2] / 30.0


def expr(name, x):
    """The same curve as an ffmpeg expression. x is an ffmpeg expression for the 0-to-1 progress,
    for example "t/0.4". The result contains commas, so keep it inside '...' in the filter."""
    kind, p, _, _ = CURVES[name]
    u = f"clip({x},0,1)"
    n = _raw(kind, p)(1.0)   # scale so the curve ends exactly on 1
    if kind == "linear":
        return u
    if kind == "powerOut":
        return f"(1-pow(1-{u},{p[0]}))"
    if kind == "powerIn":
        return f"pow({u},{p[0]})"
    if kind == "expoOut":
        return f"((1-exp(-{p[0]}*{u}))/{n:.6f})"
    if kind == "snapRamp":
        w, tau = p
        return f"(({w}*(1-exp(-{u}/{tau}))+{1 - w:.6f}*{u})/{n:.6f})"
    if kind == "snapOver":
        a, ta, td = p
        return f"((1+{a}*exp(-{u}/{td}))*(1-exp(-{u}/{ta}))/{n:.6f})"
    if kind == "spring":
        c, om = p
        return f"((1-exp(-{c}*{u})*(cos({om}*{u})+{c / om:.6f}*sin({om}*{u})))/{n:.6f})"
    if kind == "smooth":
        return f"(pow({u},{p[0]})/(pow({u},{p[0]})+pow(1-{u},{p[0]})))"
    raise ValueError(kind)


if __name__ == "__main__":
    print(f"{'name':<11} {'secs':>5}  " + "  ".join(f"{t:>4.2f}" for t in (0.1, 0.25, 0.5, 0.75, 0.9)) + "   use")
    for nm in NAMES:
        print(f"{nm:<11} {seconds(nm):>5.2f}  " + "  ".join(f"{ease(nm, t):>4.2f}" for t in (0.1, 0.25, 0.5, 0.75, 0.9))
              + "   " + CURVES[nm][3])
