#!/usr/bin/env python3
"""
Fit the inset/rotation anchor constants of filmic RGB's "AgX" color science
(v8) — see doc/filmic-agx.md and filmic_agx_prepare_bracket() in
src/iop/filmicrgb.c, whose PROVISIONAL constants this script is meant to replace.

Model (mirrors the C pixel path, working profile = linear Rec2020):
  work RGB -> inset matrix -> per-channel [log2 encoding -> sigmoid spline ->
  hardness power] -> exact-inverse outset -> measure chroma / hue in Kirk Yrg.

Objectives:
  1. purity-vs-exposure: colors on the working-gamut boundary must reach
     achromatic (chroma ratio ~ 0) at the white end of the curve, monotonically;
  2. hue drift: dh/dEV in Yrg matches a chosen uniform target across the wheel
     (0 for a neutral character, or a uniform warm bias);
  3. hard constraints as penalties: displaced primaries stay inside the working
     gamut triangle with a margin (positivity of the bracket) and away from
     degeneracy (conditioning).

Usage:
  python3 tools/derive_filmic_agx_primaries.py [--drift-target DEG_PER_EV]

Prints the C arrays to paste into filmic_agx_prepare_bracket(). Requires
numpy + scipy. Constants below are copied verbatim from the C code so the model
matches the pipeline bit-for-bit in spirit (float64 here, float32 there).
"""

import argparse
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- constants
# copied from src/common/chromatic_adaptation.h and colorspaces_inline_conversions.h

XYZ_D50_to_D65_CAT16 = np.array([
    [9.89466254e-01, -4.00304626e-02, 4.40530317e-02],
    [-5.40518733e-03, 1.00666069e+00, -1.75551955e-03],
    [-4.03920992e-04, 1.50768030e-02, 1.30210211e+00]])

XYZ_D65_to_LMS_2006 = np.array([
    [0.257085, 0.859943, -0.031061],
    [-0.394427, 1.175800, 0.106423],
    [0.064856, -0.076250, 0.559067]])

filmlightRGB_to_LMS = np.array([
    [0.95, 0.38, 0.00],
    [0.05, 0.62, 0.03],
    [0.00, 0.00, 0.97]])
LMS_to_filmlightRGB = np.linalg.inv(filmlightRGB_to_LMS)

Y_2006_COEFFS = np.array([0.68990272, 0.34832189, 0.0])

# linear Rec2020, D50-adapted (Bradford), as computed by the LCMS path for the
# working profile. Close enough for anchor fitting; regenerate from the pipeline
# if exactness matters.
REC2020_TO_XYZ_D50 = np.array([
    [0.6734241, 0.1656411, 0.1251286],
    [0.2790177, 0.6753402, 0.0456377],
    [-0.0019300, 0.0299784, 0.7973330]])

# filmic defaults the anchors are fitted against — keep in sync with the C
# $DEFAULT values. The curve model is shared with the appearance-match harness
# (C-exact v3 node geometry + perceptual-sigmoid segments).
from derive_filmic_default_curve import curve_factory, GREY  # noqa: E402

BLACK_EV, WHITE_EV = -8.0, 4.0
CURVE_DEFAULTS = (1.18, 10.0, 0.0, 1.5, 1.5)  # contrast, latitude %, balance %, toe power, (shoulder is slope-matched)
CURVE = curve_factory(BLACK_EV, WHITE_EV, *CURVE_DEFAULTS, shoulder_slope_matched=True)

# ---------------------------------------------------------------- color helpers

def xyz_D50_to_Yrg(xyz):
    lms = XYZ_D65_to_LMS_2006 @ (XYZ_D50_to_D65_CAT16 @ xyz)
    Y = Y_2006_COEFFS @ lms
    a = lms.sum()
    rgb = LMS_to_filmlightRGB @ (lms / a if a != 0 else lms)
    return np.array([Y, rgb[0], rgb[1]])

def rgb_work_to_Yrg(rgb):
    return xyz_D50_to_Yrg(REC2020_TO_XYZ_D50 @ rgb)

WHITE_YRG = rgb_work_to_Yrg(np.ones(3))

def chroma_hue(Yrg):
    r, g = Yrg[1] - WHITE_YRG[1], Yrg[2] - WHITE_YRG[2]
    return np.hypot(r, g), np.arctan2(g, r)

# ---------------------------------------------------------------- curve model

def tone_map(rgb):
    dr = WHITE_EV - BLACK_EV
    out = np.empty(3)
    for c in range(3):
        x = (np.log2(max(rgb[c], 1e-10) / GREY) - BLACK_EV) / dr
        out[c] = CURVE(x)
    return out

# ---------------------------------------------------------------- bracket model

def bracket_matrices(insets, rotations):
    """inset matrix M (work->rendering) and its exact inverse, exactly as the C code."""
    white_xyz = REC2020_TO_XYZ_D50 @ np.ones(3)
    P_prime = np.empty((3, 3))
    for i in range(3):
        p_yrg = xyz_D50_to_Yrg(REC2020_TO_XYZ_D50[:, i])
        d = p_yrg[1:] - WHITE_YRG[1:]
        s = 1.0 - insets[i]
        ca, sa = np.cos(rotations[i]), np.sin(rotations[i])
        rot = np.array([[ca, -sa], [sa, ca]])
        rg = WHITE_YRG[1:] + s * (rot @ d)
        # Yrg -> XYZ D50 (inverse path of xyz_D50_to_Yrg)
        rgbn = np.array([rg[0], rg[1], 1.0 - rg[0] - rg[1]])
        lms = filmlightRGB_to_LMS @ rgbn
        lms *= p_yrg[0] / (Y_2006_COEFFS @ lms)
        P_prime[:, i] = np.linalg.inv(XYZ_D50_to_D65_CAT16) @ \
                        np.linalg.inv(XYZ_D65_to_LMS_2006) @ lms
    scale = np.linalg.solve(P_prime, white_xyz)
    P_inset = P_prime * scale[None, :]
    M = np.linalg.inv(REC2020_TO_XYZ_D50) @ P_inset
    return M, np.linalg.inv(M)

# ---------------------------------------------------------------- objectives

EVS = np.arange(-2.0, WHITE_EV + 0.51, 0.5)
HUE_STEPS = 24

def boundary_colors():
    """saturated colors on the working-gamut boundary, one channel at zero"""
    out = []
    for k in range(HUE_STEPS):
        h = 2 * np.pi * k / HUE_STEPS
        rgb = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)])
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # in [0,1], min channel 0
        out.append(rgb)
    return out

SHOULDER_EV = BLACK_EV + CURVE.sh_x * (WHITE_EV - BLACK_EV)  # where compression starts

# ---------------------------------------------------------------- priority colors
# The colors whose fidelity is non-negotiable : human skin tones (database from
# src/common/color_vocabulary.c, CIE Lab under D65, avg ± 1.5 std corners) and
# in-gamut diffuse reflectances, both swept over the tonal placements a
# photographer may give them. Used to fit the outset recovery (see --fit-outset).

_SKIN_LAB = [  # L avg,std, a avg,std, b avg,std — see color_vocabulary.c for sources
    (60.9, 3.4, 7.0, 1.7, 15.0, 1.8), (61.9, 3.7, 7.1, 1.7, 17.4, 2.0),
    (60.6, 4.8, 6.5, 1.6, 16.4, 2.3), (63.0, 5.5, 5.6, 1.9, 14.0, 2.9),
    (56.4, 3.2, 11.7, 2.1, 16.3, 1.4), (56.8, 4.1, 11.6, 2.2, 17.7, 1.8),
    (56.1, 4.5, 11.3, 2.1, 16.4, 2.2), (59.2, 5.1, 11.6, 2.8, 15.1, 2.3),
    (44.0, 2.0, 14.0, 1.0, 19.0, 1.0), (58.0, 1.0, 15.0, 1.0, 21.0, 1.0),
    (58.9, 3.1, 11.4, 2.1, 14.2, 1.5), (60.7, 4.0, 10.5, 2.3, 17.2, 2.1),
    (58.0, 4.4, 11.7, 2.3, 15.8, 2.1), (59.6, 5.5, 11.8, 3.1, 14.6, 2.6),
    (48.0, 1.0, 15.0, 1.0, 20.0, 1.0), (63.0, 1.0, 16.0, 1.0, 21.0, 1.0)]

_D65_WHITE = np.array([0.95047, 1.0, 1.08883])

def _lab_d65_to_work(L, a, b):
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    f = lambda t: t**3 if t**3 > 0.008856 else (t - 16.0 / 116.0) / 7.787
    xyz_d65 = _D65_WHITE * np.array([f(fx), f(fy), f(fz)])
    xyz_d50 = np.linalg.inv(XYZ_D50_to_D65_CAT16) @ xyz_d65
    return np.linalg.inv(REC2020_TO_XYZ_D50) @ xyz_d50

def priority_samples():
    """skin corners + diffuse hue circle, each swept over tonal placements"""
    y_row = REC2020_TO_XYZ_D50[1]
    def place(rgb, evs):
        lum = y_row @ rgb
        return [rgb * (GREY * 2.0**e / lum) for e in evs]
    out = []
    for (L, Ls, a, As, b, Bs) in _SKIN_LAB:
        for dL in (-1.5, 0.0, 1.5):
            for da in (-1.5, 1.5):
                for db in (-1.5, 1.5):
                    rgb = _lab_d65_to_work(L + dL * Ls, a + da * As, b + db * Bs)
                    if rgb.min() > 0:
                        out += place(rgb, (-1.5, -0.75, 0.0, 0.75, 1.5))
    for k in range(12):
        h = 2 * np.pi * k / 12
        base = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)])
        base = (base - base.min()) / (base.max() - base.min())
        for p in (0.3, 0.5, 0.7):
            out += place(p * base + (1 - p) * 0.5, (-2, -1, 0, 1, 2, 2.5))
    return out

def chroma_trajectory(sample, M, M_inv):
    """chroma ratio and hue drift vs exposure, through the bracketed curve"""
    c_in, h_in = chroma_hue(rgb_work_to_Yrg(sample))
    ratios, drifts = [], []
    for ev in EVS:
        rgb = sample * GREY * 2.0 ** ev
        mapped = M_inv @ tone_map(M @ rgb)
        c_out, h_out = chroma_hue(rgb_work_to_Yrg(np.maximum(mapped, 1e-10)))
        ratios.append(c_out / max(c_in, 1e-6))
        drifts.append(np.remainder(h_out - h_in + np.pi, 2 * np.pi) - np.pi)
    return np.array(ratios), np.array(drifts)

IDENTITY = np.eye(3)
BASELINES = {tuple(s): chroma_trajectory(s, IDENTITY, IDENTITY)[0] for s in boundary_colors()}

def target_profile(baseline):
    """Design target for the bleaching : transparent (baseline) below the shoulder,
    then a smooth ramp to full achromatic exactly at the white endpoint. This pins
    the bleach *rate* through the shoulder — without it the fit has no optimum and
    rails the insets (endpoint-only objectives are monotone in the inset amount)."""
    t = np.clip((EVS - SHOULDER_EV) / (WHITE_EV - SHOULDER_EV), 0.0, 1.0)
    ramp = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep, 1 -> 0
    return baseline * ramp

def residuals(params, drift_target_rad_per_ev):
    insets, rotations = params[:3], params[3:]
    M, M_inv = bracket_matrices(insets, rotations)

    res = []
    # hard constraints as strong penalties : positivity with margin, over the whole
    # user ray t in [1, 2] (the runtime slider scales insets and rotations together)
    M2, _ = bracket_matrices(np.clip(2.0 * insets, 0.0, 0.9), 2.0 * rotations)
    res.append(500.0 * np.maximum(0.0, 0.005 - M).sum())
    res.append(500.0 * np.maximum(0.0, 0.005 - M2).sum())
    res.append(10.0 * max(0.0, np.linalg.cond(M) - 8.0))            # conditioning
    for sample in boundary_colors():
        ratios, drifts = chroma_trajectory(sample, M, M_inv)
        target = target_profile(BASELINES[tuple(sample)])
        res.extend(3.0 * (ratios - target))                          # purity-vs-exposure profile
        res.extend(2.0 * np.maximum(0.0, np.diff(ratios)))           # monotone bleaching
        d = np.gradient(drifts, EVS)
        res.extend(0.5 * (d - drift_target_rad_per_ev))              # uniform drift
    return np.array(res, dtype=float)

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-target", type=float, default=0.0,
                    help="uniform hue drift target in degrees per EV of compression "
                         "(0 = neutral/hue-stable character, e.g. -0.5 for a warm bias)")
    ap.add_argument("--inset", type=float, default=0.25,
                    help="uniform inset anchor. NOT fitted: bleach depth saturates with the "
                         "inset (the exact-inverse outset re-expands what the curve did not "
                         "equalize, and the Yrg gamut mapper owns the very endpoint), so "
                         "endpoint objectives rail it against any bound. 0.25 sits at the "
                         "knee of diminishing returns with ~0.96 midtone transparency at "
                         "+1 EV and leaves conditioning headroom for the user ray t <= 2.")
    ap.add_argument("--fit-insets", action="store_true",
                    help="fit all 6 parameters anyway (diagnostic; expect railed insets)")
    ap.add_argument("--fit-outset", action="store_true",
                    help="fit the outset recovery factor kappa. The outset is the inverse of "
                         "the bracket built with kappa-scaled insets (kappa = 1 : exact "
                         "inverse). An exact inverse mandatorily bleaches every color the "
                         "curve touches — with the low-latitude sigmoid default that is the "
                         "whole tonal range, washing out valid midtone colors (skin tones). "
                         "kappa > 1 over-expands so that priority colors (skin database + "
                         "diffuse reflectances, over their tonal placements) reach the "
                         "output-chroma-equals-input clamp : the clamp then trims recovery "
                         "to exactly 1.0 per pixel, tone-adaptively, which is what makes one "
                         "fixed kappa portable across curves and dynamic ranges (verified "
                         "6.5-16 EV). Then re-fits the inset-blue and outset rotations for the "
                         "RECOVERED-chroma regime (the kappa recovery re-exposes drift the "
                         "bleaching used to hide), under the maintainer's priority ordering. "
                         "This is the CURRENT production fit ; it supersedes --minimax, which "
                         "was correct only for the earlier bleached (exact-inverse) regime.")
    ap.add_argument("--minimax", action="store_true",
                    help="fit rotations for the WORST-CASE hue drift (Chebyshev) over "
                         "EV <= +3.5 instead of zero-mean L2. This is the right objective "
                         "when the hue is NOT anchored in Ych afterwards (see "
                         "FILMIC_AGX_UNANCHORED_HUE_TEST in filmicrgb.c) : without a "
                         "backstop, only the worst case matters. Floor is ~28.5° — the "
                         "drift field reverses direction between toe and shoulder, which "
                         "no constant matrix pair can serve on both sides. SUPERSEDED by "
                         "--fit-outset : minimax was fit for the bleached exact-inverse regime "
                         "and its blue rotation now CAUSES purple on recovered blues.")
    ap.add_argument("--fit-priority", action="store_true",
                    help="CURRENT production fit. Joint Nelder-Mead fine-tune of the whole "
                         "bracket (per-primary inset chroma + rotation, per-primary outset "
                         "chroma + rotation = 12 params) started from the uniform-0.25 / "
                         "kappa=2 config, minimizing hue drift over the PRIORITY set (skin "
                         "database + diffuse reflectances) at preserved chroma. Supersedes "
                         "--fit-outset : the outset is now a per-primary expansion, not a "
                         "scalar kappa. Hard constraints (barriers the optimizer cannot "
                         "cross) : skin red-ward drift <= -1.5 deg, skin chroma >= 92%, "
                         "diffuse recovery p5 >= 0.97 (bleaching cannot game accuracy), inset "
                         "positivity >= 0.004, conditioning <= 6.5. Inset capped at 0.35 to "
                         "keep rendering character (the free optimum runs 0.55 for a sub-JND "
                         "gain at a visible highlight-desaturation cost). sRGB blue at high EV "
                         "is not targeted — a structural DoF limit, left to the Ych recovery.")
    args = ap.parse_args()

    if args.fit_priority:
        from scipy.optimize import minimize
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0 ** e / lum) for e in evs]

        # PRIORITY set : reflective/memory colors (already EV-spread) + skin database
        refl = priority_samples()
        skin_base = [_lab_d65_to_work(L, a + da * As, b + db * Bs)
                     for (L, Ls, a, As, b, Bs) in _SKIN_LAB
                     for da in (-1, 1) for db in (-1, 1)]
        skin_base = [s for s in skin_base if s.min() > 0]
        skin = [s for w in skin_base for s in place(w, (-1.5, -0.5, 0.5, 1.5))]

        def brk(p):
            M, _ = bracket_matrices(p[0:3], p[3:6])
            Mo, _ = bracket_matrices(p[6:9], p[9:12])
            return M, np.linalg.inv(Mo)

        def meas(rgb, M, Mo):
            c_o, h_o = chroma_hue(rgb_work_to_Yrg(rgb))
            x = (np.log2(np.maximum(M @ rgb, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            o = Mo @ np.array([CURVE(v) for v in x])
            c_f, h_f = chroma_hue(rgb_work_to_Yrg(np.maximum(o, 1e-10)))
            return min(c_f / max(c_o, 1e-9), 1.0), np.rad2deg(np.remainder(h_f - h_o + np.pi, 2 * np.pi) - np.pi)

        BIG = 1e5

        def objective(p):
            if any(p[i] < 0.20 or p[i] > 0.35 for i in range(3)):  # inset cap : keep character
                return BIG
            M, Mo = brk(p)
            if M.min() < 0.004:
                return BIG - M.min() * 1e4
            if max(np.linalg.cond(M), np.linalg.cond(Mo)) > 6.5:
                return BIG
            sdh = []
            for s in skin:
                cr, dh = meas(s, M, Mo)
                if dh < -1.5 or cr < 0.92:            # skin red-ward veto + chroma floor
                    return BIG + abs(dh) + (1 - cr) * 100
                sdh.append(abs(dh))
            rd = [meas(s, M, Mo) for s in refl]
            if np.percentile([c for c, d in rd], 5) < 0.97:   # no bleaching
                return BIG
            rdh = [abs(d) for c, d in rd if c > 0.5]
            return np.mean(rdh) + 0.2 * np.max(rdh) + 1.5 * np.mean(sdh)

        p0 = [0.25, 0.25, 0.25, np.deg2rad(-2.50), np.deg2rad(9.29), np.deg2rad(3.11),
              0.50, 0.50, 0.50, np.deg2rad(-2.00), np.deg2rad(11.93), np.deg2rad(5.20)]
        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"xatol": 2e-5, "fatol": 2e-4, "maxiter": 8000, "maxfev": 12000})
        p = res.x
        M, Mo = brk(p)
        sd = [meas(s, M, Mo) for s in skin]
        rd = [meas(s, M, Mo) for s in refl]
        sdh = [d for c, d in sd]
        rdh = [abs(d) for c, d in rd if c > 0.5]
        print("// fitted by tools/derive_filmic_agx_primaries.py --fit-priority (obj %.3f)" % res.fun)
        print("// priority set : skin |mean| %.1f deg [%+.1f..%+.1f] ; reflective mean %.1f max %.1f ; cond %.1f/%.1f"
              % (np.mean(np.abs(sdh)), min(sdh), max(sdh), np.mean(rdh), np.max(rdh),
                 np.linalg.cond(M), np.linalg.cond(Mo)))
        print("static const float inset_anchor[3]    = { %.6ff, %.6ff, %.6ff };" % (p[0], p[1], p[2]))
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[3], p[4], p[5], *np.rad2deg(p[3:6])))
        print("static const float outset_anchor[3]   = { %.6ff, %.6ff, %.6ff };" % (p[6], p[7], p[8]))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[9], p[10], p[11], *np.rad2deg(p[9:12])))
        print("// kappa-equivalent per-primary outset/inset ratios : %.3f %.3f %.3f"
              % (p[6] / p[0], p[7] / p[1], p[8] / p[2]))
        return

    insets0 = np.full(3, args.inset)
    rot0 = np.zeros(3)
    if args.fit_outset:
        # production rotations (from --minimax) ; keep in sync with the C anchors
        rot = np.array([-0.0436910, 0.1621254, 0.4199733])
        M, _ = bracket_matrices(insets0, rot)
        samples = priority_samples()

        def p5_ratio(kappa, curve=CURVE, black=BLACK_EV, white=WHITE_EV):
            Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), rot)
            M_out = np.linalg.inv(Mk)
            ratios = []
            for s in samples:
                c_o, _ = chroma_hue(rgb_work_to_Yrg(s))
                x = (np.log2(np.maximum(M @ s, 1e-10) / GREY) - black) / (white - black)
                out = M_out @ np.array([curve(v) for v in x])
                c_f, _ = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                ratios.append(c_f / max(c_o, 1e-9))
            return np.percentile(ratios, 5)

        lo, hi = 1.0, 2.5
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if p5_ratio(mid) < 1.0 else (lo, mid)
        kappa = 0.5 * (lo + hi)
        print(f"// OUTSET_RECOVERY (kappa) = {kappa:.3f} : smallest outset over-expansion")
        print(f"// whose 5th-percentile pre-clamp chroma ratio on the priority set reaches 1")

        # ---- rotations for the RECOVERED-CHROMA regime ----------------------
        # kappa recovery keeps the drifting pixels saturated, so the drift is now
        # visible where it used to be bleached invisible. Crucially, the old
        # minimax blue inset rotation (+24 deg, tuned to counter BLEACHED-pixel
        # drift) is measured to CAUSE most of the purple on RECOVERED saturated
        # blues (isolation : +13.7 deg with it, +1.5 without, at ~95% chroma).
        # So re-fit for the recovered regime. Structure that works (a free 6-param
        # search wanders into skin-wrecking minima) : keep inset red/green at their
        # diffuse-validated values (they protect skin, sunset and foliage — zeroing
        # them sends green to -25 deg and sunsets to +37 deg), fit inset BLUE plus
        # the three outset deltas. Priority weights, from the maintainer's stated
        # ordering (skin/reflective = absolute priority ; blue LED purple = the
        # worst real complaint ; sunset should drift yellow not red) :
        from scipy.optimize import minimize
        M_srgb = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                           [0.0193, 0.1192, 0.9505]])
        M_2020 = np.array([[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593],
                           [0.0000, 0.0281, 1.0610]])
        TO2020 = np.linalg.inv(M_2020) @ M_srgb
        blue_srgb = np.maximum(TO2020 @ np.array([0.0, 0.0, 1.0]), 1e-6)
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0**e / lum) for e in evs]
        skin_set, sunset_set = [], []
        for (L, Ls, a, As, b, Bs) in _SKIN_LAB:
            for da in (-1.5, 1.5):
                for db in (-1.5, 1.5):
                    rgb = _lab_d65_to_work(L, a + da * As, b + db * Bs)
                    if rgb.min() > 0:
                        skin_set += place(rgb, (-1.5, 0.0, 1.5))
        for base in ([1, .45, .12], [1, .6, .2]):
            for pu in (.6, .9):
                sunset_set += place(np.maximum(TO2020 @ (pu * np.array(base) + (1 - pu) * .5), 1e-6),
                                    (0.0, 1.5, 3.0))

        def run_px(rgb, M_in, M_out):
            c_o, h_o = chroma_hue(rgb_work_to_Yrg(rgb))
            x = (np.log2(np.maximum(M_in @ rgb, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            out = M_out @ np.array([CURVE(v) for v in x])
            c_f, h_f = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
            dh = np.rad2deg(np.remainder(h_f - h_o + np.pi, 2 * np.pi) - np.pi)
            return min(c_f, c_o) / max(c_o, 1e-9), dh, c_f / max(c_o, 1e-9)

        # sign convention : drift > 0 is yellow-ward for warm hues (skin/sunset),
        # drift < 0 is red-ward. Red-ward on skin/sunset is the maintainer's veto.
        def objective(p):
            irot = np.array([rot[0], rot[1], p[0]])       # inset red/green fixed, blue free
            orot = irot + p[1:]                           # outset = inset + deltas
            M_in, _ = bracket_matrices(insets0, irot)
            Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), orot)
            M_out = np.linalg.inv(Mk)
            blue = max(abs(run_px(blue_srgb * GREY * 2.0**ev, M_in, M_out)[1]) for ev in (3.5, 4.0, 4.5))
            skd = [run_px(s, M_in, M_out)[1] for s in skin_set]
            snd = [run_px(s, M_in, M_out)[1] for s in sunset_set]
            p5 = np.percentile([run_px(s, M_in, M_out)[2] for s in samples[::4]], 5)
            posmin = M_in.min()
            return (blue                                        # #1 : blue purple
                    + 30.0 * max(0.0, -min(skd) - 1.5)          # skin red-ward veto (cap 1.5)
                    + 10.0 * max(0.0, max(skd) - 4.0)           # skin excess yellow
                    + 10.0 * max(0.0, -min(snd) - 1.0)          # sunset must not go red
                    + 100.0 * max(0.0, 0.9 - p5)                # diffuse recovery preserved
                    + 1e4 * max(0.0, 0.005 - posmin))           # inset positivity margin

        x0 = np.deg2rad([3.0, 0.4, 3.1, 4.5])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"xatol": 2e-4, "fatol": 5e-3, "maxiter": 500})
        irot = np.array([rot[0], rot[1], res.x[0]])
        orot = irot + res.x[1:]
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (*irot, *np.rad2deg(irot)))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (*orot, *np.rad2deg(orot)))
        M_in, _ = bracket_matrices(insets0, irot)
        Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), orot)
        M_out = np.linalg.inv(Mk)
        b4 = run_px(blue_srgb * GREY * 16.0, M_in, M_out)
        skd = [run_px(s, M_in, M_out)[1] for s in skin_set]
        snd = [run_px(s, M_in, M_out)[1] for s in sunset_set]
        print("// sRGB blue @+4EV drift %+.1f deg (%.0f%% chroma) ; skin [%+.1f..%+.1f] ; sunset [%+.1f..%+.1f]"
              % (b4[1], 100 * b4[0], min(skd), max(skd), min(snd), max(snd)))
        print("// inset positivity min %.4f ; portability (post-clamp p5) :" % M_in.min())
        for (bl, wh, name) in ((-4.0, 2.5, "studio 6.5EV"), (-8.0, 4.0, "default 12EV"), (-10.0, 6.0, "HDR 16EV")):
            c = curve_factory(bl, wh, *CURVE_DEFAULTS, shoulder_slope_matched=True)
            ratios = []
            for s in samples:
                c_o, _ = chroma_hue(rgb_work_to_Yrg(s))
                x = (np.log2(np.maximum(M_in @ s, 1e-10) / GREY) - bl) / (wh - bl)
                out = M_out @ np.array([c(v) for v in x])
                c_f, _ = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                ratios.append(min(c_f / max(c_o, 1e-9), 1.0))
            print("//   %-13s p5 %.3f" % (name, np.percentile(ratios, 5)))
        return
    if args.minimax:
        from scipy.optimize import minimize
        evs = np.arange(-6.0, 3.51, 0.5)  # visible range : beyond, the gamut mapper crushes chroma

        def worst_drift(r):
            M, M_inv = bracket_matrices(insets0, r)
            M2, _ = bracket_matrices(np.clip(2.0 * insets0, 0.0, 0.9), 2.0 * r)
            pen = 1e4 * (np.maximum(0.0, 0.005 - M).sum() + np.maximum(0.0, 0.005 - M2).sum())
            worst = 0.0
            for s in boundary_colors():
                c_in, h_in = chroma_hue(rgb_work_to_Yrg(s))
                for ev in evs:
                    out = M_inv @ tone_map(M @ (s * GREY * 2.0 ** ev))
                    _, h = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                    worst = max(worst, abs(np.remainder(h - h_in + np.pi, 2 * np.pi) - np.pi))
            return np.rad2deg(worst) + pen

        res = minimize(worst_drift, np.deg2rad([-2.0, 4.0, 11.0]), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 400})
        insets, rotations = insets0, res.x
        fit_note = f"minimax, worst-case drift {res.fun:.2f} deg over EV <= +3.5"
    elif args.fit_insets:
        x0 = np.concatenate([insets0, rot0])
        bounds = ([0.02] * 3 + [np.deg2rad(-15)] * 3,
                  [0.45] * 3 + [np.deg2rad(15)] * 3)
        fit = least_squares(residuals, x0, bounds=bounds,
                            args=(np.deg2rad(args.drift_target),), verbose=1)
        insets, rotations = fit.x[:3], fit.x[3:]
    else:
        # rotations only : well-posed 3-parameter fit for the hue drift objective
        fit = least_squares(lambda r, tgt: residuals(np.concatenate([insets0, r]), tgt),
                            rot0, bounds=(np.deg2rad(-15), np.deg2rad(15)),
                            args=(np.deg2rad(args.drift_target),), verbose=1)
        insets, rotations = insets0, fit.x
    if not args.minimax:
        fit_note = f"drift target {args.drift_target} deg/EV, cost {fit.cost:.4f}"
    print(f"\n// fitted by tools/derive_filmic_agx_primaries.py ({fit_note})")
    print("static const float inset_anchor[3] = { %.6ff, %.6ff, %.6ff };" % tuple(insets))
    print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
          % (tuple(rotations) + tuple(np.rad2deg(rotations))))
    M, _ = bracket_matrices(insets, rotations)
    print("// inset matrix (work RGB -> rendering), rows sum to 1:")
    for row in M:
        print("//   [ %+.6f %+.6f %+.6f ]  (sum %.6f)" % (*row, row.sum()))

if __name__ == "__main__":
    main()
