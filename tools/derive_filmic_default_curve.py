#!/usr/bin/env python3
"""
Derive the default filmic RGB tone curve from an appearance match instead of taste.

Principle: the tone curve is the luminance mapping between two *known* viewing
states, so its default is a solvable problem, not an opinion:

  scene state    : diffuse white ~1000 cd/m2 outdoors, observer adapted to it,
                   average surround, no flare (the scene is the reference);
  display state  : SDR monitor ~100 cd/m2 white, dim surround, veiling flare
                   0.1% of white (demanding viewing : dim room, good display —
                   office-flare fits crush near-blacks, see --flare help).

For each scene exposure (EV around mid-gray) we compute the perceived lightness
J under the scene state with CIECAM16 (achromatic path), then ask which display
luminance produces the same J under the display state — flare included. The
unconstrained match cannot fit the display range, so we least-squares the
filmic curve family onto it with two weights:

  - a content-mass prior (where photographs hold detail), and
  - a JND-visibility smoothness term: the rendering may not introduce
    perceptual-lightness curvature (d2J/dEV2) sharper than the appearance
    match itself contains anywhere. Without it the toe/shoulder powers rail
    ("hold the match, clip hard") because errors at extreme EVs are cheap.

The curve model below replicates the C v3 geometry EXACTLY
(filmic_v3_compute_geometry / _nodes_from_legacy + the spline v4 sigmoid
segments), so the fitted parameters are directly the module's user parameters.
In particular the user 'contrast' is normalized by DR/8 and gamma-compensated,
as in the C code.

Usage:
  python3.12 tools/derive_filmic_default_curve.py
  ... --scene-white 5000 --display-white 200 --flare 0.01   # variants

Changing shipped defaults from these numbers is a product decision:
run, look, then decide.
"""

import argparse
import numpy as np
from scipy.optimize import least_squares

GREY = 0.1845
SAFETY_MARGIN = 0.01                # C: SAFETY_MARGIN
BLACK_TARGET = 0.01517634 / 100.0   # C: black_point_target default, linear
WHITE_TARGET = 1.0                  # C: white_point_target default, linear

# ------------------------------------------------------------- CIECAM16 (achromatic)

def ciecam16_J(L, Lw, La, surround):
    """Perceived lightness J of an achromatic stimulus of luminance L (cd/m2)
    seen against a ~20% background under a white of Lw, adaptation La."""
    c = {"average": 0.69, "dim": 0.59, "dark": 0.525}[surround]
    k = 1.0 / (5.0 * La + 1.0)
    F_L = 0.2 * k**4 * 5.0 * La + 0.1 * (1.0 - k**4) ** 2 * (5.0 * La) ** (1.0 / 3.0)
    n = 0.2  # background/white ratio
    Nbb = 0.725 * (1.0 / n) ** 0.2
    z = 1.48 + np.sqrt(n)

    def achromatic(Y_rel):
        t = (F_L * np.maximum(Y_rel, 0.0) / 100.0) ** 0.42
        Ra = 400.0 * t / (t + 27.13) + 0.1
        return (3.05 * Ra - 0.305) * Nbb

    return 100.0 * (achromatic(100.0 * L / Lw) / achromatic(100.0)) ** (c * z)

# ------------------------------------------------------------- filmic curve model
# exact port of filmic_v3_compute_geometry + filmic_v3_compute_nodes_from_legacy
# + the spline v4 sigmoid solver (src/iop/filmicrgb.c)

def _sigmoid_scale(limit_x, limit_y, tx, ty, slope, power):
    projected = slope * max(1e-6, limit_x - tx)
    actual = max(1e-6, limit_y - ty)
    base = max(1e-6, actual ** -power - projected ** -power)
    return min(1e9, base ** (-1.0 / power))

def curve_factory(black_ev, white_ev, contrast_param, latitude_pct, balance_pct,
                  toe_power, shoulder_power):
    """Returns curve(x): normalized log input -> display-linear output.
    Parameters are the module's user parameters, v3 geometry, spline v4."""
    dr = white_ev - black_ev
    grey_log = abs(black_ev) / dr
    output_power = np.log(GREY) / np.log(grey_log)  # auto-hardness, as in C
    grey_display = GREY ** (1.0 / output_power)
    black_display = np.clip(BLACK_TARGET, 0.0, GREY) ** (1.0 / output_power)
    white_display = max(WHITE_TARGET, GREY) ** (1.0 / output_power)

    # user contrast -> spline slope : DR normalization + gamma compensation
    slope = contrast_param * dr / 8.0
    contrast = slope / (output_power * grey_display ** (output_power - 1.0))
    min_contrast = max(1.0,
                       (white_display - grey_display) / (1.0 - grey_log),
                       (grey_display - black_display) / grey_log) + SAFETY_MARGIN
    contrast = np.clip(contrast, min_contrast, 100.0)

    icpt = grey_display - contrast * grey_log
    margin = SAFETY_MARGIN * (white_display - black_display)
    xmin = (black_display + margin - icpt) / contrast
    xmax = (white_display - margin - icpt) / contrast

    # latitude between grey and the slope-line intersections, balance = translation
    lat = np.clip(latitude_pct, 0.0, 100.0) / 100.0
    bal = np.clip(balance_pct, -50.0, 50.0) / 100.0
    toe_x = (1.0 - lat) * grey_log + lat * xmin
    sh_x = (1.0 - lat) * grey_log + lat * xmax
    corr = 2.0 * bal * ((sh_x - grey_log) if bal > 0.0 else (grey_log - toe_x))
    toe_x = max(toe_x - corr, xmin)
    sh_x = min(sh_x - corr, xmax)
    toe_y = toe_x * contrast + icpt
    sh_y = sh_x * contrast + icpt

    # sigmoid segments + degenerate fallbacks, as in the C spline v4 solver
    toe_s = -_sigmoid_scale(1.0, 1.0 - black_display, 1.0 - toe_x, 1.0 - toe_y,
                            contrast, toe_power)
    sh_s = _sigmoid_scale(1.0, white_display, sh_x, sh_y, contrast, shoulder_power)
    toe_dx, toe_dy = max(1e-6, toe_x), max(1e-6, toe_y - black_display)
    sh_dx, sh_dy = max(1e-6, 1.0 - sh_x), max(1e-6, white_display - sh_y)
    toe_convex = toe_dy / toe_dx > contrast
    sh_concave = sh_dy / sh_dx > contrast
    toe_fb_p = contrast * toe_dx / toe_dy
    toe_fb_c = toe_dy / toe_dx ** toe_fb_p
    sh_fb_p = contrast * sh_dx / sh_dy
    sh_fb_c = sh_dy / sh_dx ** sh_fb_p

    def curve(x):
        x = np.clip(x, 0.0, 1.0)
        if x < toe_x:
            if toe_convex:
                y = black_display + max(0.0, toe_fb_c * x ** toe_fb_p)
            else:
                u = contrast * (x - toe_x) / toe_s
                y = toe_s * (u / (1.0 + u ** toe_power) ** (1.0 / toe_power)) + toe_y
        elif x > sh_x:
            if sh_concave:
                y = white_display - max(0.0, sh_fb_c * (1.0 - x) ** sh_fb_p)
            else:
                u = contrast * (x - sh_x) / sh_s
                y = sh_s * (u / (1.0 + u ** shoulder_power) ** (1.0 / shoulder_power)) + sh_y
        else:
            y = contrast * x + icpt
        return np.clip(y, black_display, white_display) ** output_power

    # metadata for other harnesses (anchor fit needs the compression zone)
    curve.grey_log, curve.toe_x, curve.sh_x = grey_log, toe_x, sh_x
    curve.spline_contrast, curve.output_power = contrast, output_power
    return curve

# ------------------------------------------------------------- the match

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-white", type=float, default=1000.0,
                    help="scene diffuse white, cd/m2 (outdoor overcast ~1000, sunny ~5000)")
    ap.add_argument("--display-white", type=float, default=100.0,
                    help="display white, cd/m2 (SDR reference 100)")
    ap.add_argument("--flare", type=float, default=0.001,
                    help="display veiling flare as a fraction of display white. Default 0.1%% : fit "
                         "for demanding viewing (dim room, good display) so the default toe "
                         "does not crush for anyone ; 0.5%%-flare fits were reported as "
                         "black-crushing in visual testing.")
    ap.add_argument("--surround", default="dim", choices=["average", "dim", "dark"])
    ap.add_argument("--black-ev", type=float, default=-8.0)
    ap.add_argument("--white-ev", type=float, default=4.0)
    ap.add_argument("--content-center", type=float, default=-0.5,
                    help="EV center of the content-mass weighting")
    ap.add_argument("--content-sigma", type=float, default=2.5)
    ap.add_argument("--jnd-weight", type=float, default=1.0,
                    help="weight of the excess perceptual-curvature (JND) term")
    ap.add_argument("--fix-geometry", action="store_true",
                    help="keep latitude/balance at the legacy shipped values (33 %%, 0) and "
                         "fit [contrast, powers] only. NOT the default : under spline v4 the "
                         "latitude is a *tension* control — small latitude hands the range to "
                         "the sigmoids (soft transitions), large latitude forces short, hard "
                         "turns — so it participates in the transition strength exactly like "
                         "the sigmoid powers and belongs in the fit. The CIECAM16 match has "
                         "no linear segment (cone compression is smooth everywhere), hence "
                         "the fit is started from a near-zero latitude.")
    args = ap.parse_args()

    dr = args.white_ev - args.black_ev
    evs = np.linspace(args.black_ev, args.white_ev, 121)
    dev = evs[1] - evs[0]
    xs = (evs - args.black_ev) / dr

    # scene appearance : the reference
    L_scene = args.scene_white * (GREY * 2.0 ** evs)  # diffuse white = 1.0
    J_scene = ciecam16_J(L_scene, args.scene_white, 0.2 * args.scene_white, "average")

    flare = args.flare * args.display_white
    La_disp = 0.2 * args.display_white

    def J_display(y):
        return ciecam16_J(y * args.display_white + flare, args.display_white + flare,
                          La_disp, args.surround)

    # content-mass weighting : where photographs actually hold detail
    w = 0.05 + np.exp(-0.5 * ((evs - args.content_center) / args.content_sigma) ** 2)

    # JND-visibility tolerance : LOCAL, not global. The rendering may not introduce
    # perceptual curvature sharper than the appearance match itself has AT THE SAME
    # exposure (plus a 1 J/EV^2 margin). A global max would license toe turns as
    # sharp as the scene's own deep-shadow compression — the fit then holds the
    # slope deep into the shadows and crushes near-blacks abruptly, unpenalized
    # (observed : toe-side curvature 5 vs global tau 22, the term never bound —
    # reported as crushed near-blacks in visual testing).
    tau = np.abs(np.diff(J_scene, 2)) / dev**2 + 1.0

    def eval_params(p):
        if args.fix_geometry:
            return np.array([p[0], 33.0, 0.0, p[1], p[2]])
        return p  # contrast, latitude, balance, toe_p, sh_p

    def residuals(p):
        curve = curve_factory(args.black_ev, args.white_ev, *eval_params(p))
        J_out = J_display(np.array([curve(x) for x in xs]))
        match = np.sqrt(w) * (J_out - J_scene)
        curvature = np.abs(np.diff(J_out, 2)) / dev**2
        jnd = args.jnd_weight * np.maximum(0.0, curvature - tau)
        return np.concatenate([match, jnd])

    if args.fix_geometry:
        p0 = np.array([1.18, 1.5, 3.3])
        bounds = ([0.5, 1.05, 1.05], [3.0, 16.0, 16.0])
    else:
        # latitude-as-tension : start the sweep with the latitude very close to zero
        # (the sigmoids own the whole transition) and let the fit pull it up only if
        # the appearance match asks for it. NB : the balance loses leverage as the
        # latitude shrinks (its translation is proportional to the latitude span),
        # so check the identifiability of the fitted balance before shipping it.
        p0 = np.array([1.18, 1.0, 0.0, 1.5, 3.3])
        bounds = ([0.5, 0.5, -50.0, 1.05, 1.05], [3.0, 99.0, 50.0, 16.0, 16.0])
    fit = least_squares(residuals, p0, bounds=bounds, verbose=1)
    contrast, latitude, balance, toe_p, sh_p = eval_params(fit.x)

    curve = curve_factory(args.black_ev, args.white_ev, *eval_params(fit.x))
    eps = 0.05
    x_g = abs(args.black_ev) / dr
    slope_loglog = (np.log10(curve(x_g + eps)) - np.log10(curve(x_g - eps))) \
                   / (2 * eps * dr * np.log10(2.0))

    print(f"\n// fitted by tools/derive_filmic_default_curve.py")
    print(f"// scene {args.scene_white:.0f} cd/m2 avg surround -> display "
          f"{args.display_white:.0f} cd/m2 {args.surround} surround, flare {args.flare*100:.1f}%")
    print(f"// DR [{args.black_ev:+.1f}, {args.white_ev:+.1f}] EV, "
          f"JND weight {args.jnd_weight}, residual RMS {np.sqrt(2*fit.cost/len(evs)):.2f} J units")
    print(f"contrast        = {contrast:.3f}    (shipped default 1.180)")
    print(f"latitude        = {latitude:.1f} %   (shipped default 33.0 %)")
    print(f"balance         = {balance:+.1f} %   (shipped default +0.0 %)")
    print(f"toe power       = {toe_p:.2f}    (spline v4 'safe' 1.50)")
    print(f"shoulder power  = {sh_p:.2f}    (spline v4 'safe' 3.30)")
    print(f"spline slope    = {curve.spline_contrast:.3f}, hardness = {curve.output_power:.3f}")
    print(f"end-to-end log-log midtone slope = {slope_loglog:.3f} "
          f"(Rec.709 ~1.2, cinema ~1.5)")

if __name__ == "__main__":
    main()
