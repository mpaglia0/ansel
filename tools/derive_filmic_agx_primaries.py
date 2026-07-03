#!/usr/bin/env python3
"""
Fit the inset/rotation anchor constants of filmic RGB's "AgX-like" color science
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
# (C-exact v3 geometry + spline v4 sigmoid).
from derive_filmic_default_curve import curve_factory, GREY  # noqa: E402

BLACK_EV, WHITE_EV = -8.0, 4.0
CURVE_DEFAULTS = (1.18, 10.0, 0.0, 1.5, 9.0)  # contrast, latitude %, balance %, "safe" powers
CURVE = curve_factory(BLACK_EV, WHITE_EV, *CURVE_DEFAULTS)

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
    ap.add_argument("--minimax", action="store_true",
                    help="fit rotations for the WORST-CASE hue drift (Chebyshev) over "
                         "EV <= +3.5 instead of zero-mean L2. This is the right objective "
                         "when the hue is NOT anchored in Ych afterwards (see "
                         "FILMIC_AGX_UNANCHORED_HUE_TEST in filmicrgb.c) : without a "
                         "backstop, only the worst case matters. Floor is ~28.5° — the "
                         "drift field reverses direction between toe and shoulder, which "
                         "no constant matrix pair can serve on both sides.")
    args = ap.parse_args()

    insets0 = np.full(3, args.inset)
    rot0 = np.zeros(3)
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
