#!/usr/bin/env python3.12
"""Offline orchestrator for the filmic-AgX bleach ladder — run this PLUGGED IN.

It fits the four non-anchor variants in dependency order via successive BISECTION of the
no-bleach / extra-bleach space, so every step is confined between its neighbours (monotone,
even per-hue apparent-brightness AND hue-drift steps), and auto-patches BOTH sources:
  - SHIPPED_VARIANTS in tools/derive_filmic_agx_primaries.py   (single source of truth)
  - the DT_FILMIC_COLORSCIENCE_V6..V10 case blocks in src/iop/filmicrgb.c

no-bleach is NOT re-fit here (it is the settled bottom anchor, --min-bleach --ab-pull 200).

Order:
  1. extra-bleach  = --fit-extra-bleach --ab-stabilize AB_STAB   (stabilise reds/magentas so
                     bleaching does not over-brighten them ; apparent brightness stays ~put)
  2. medium-bleach = bisect(no-bleach, extra-bleach)
  3. low-bleach    = bisect(no-bleach, medium-bleach)
  4. high-bleach   = bisect(medium-bleach, extra-bleach)

Then it writes tools/agx_diagnose.log + tools/agx_report.md and prints the rebuild command.

Tunables (edit + re-run if --diagnose shows a step off the ramp): AB_STAB below, and W_AB /
W_HD inside the --fit-bisect branch of derive_filmic_agx_primaries.py.

Usage:  python3.12 tools/fit_agx_ladder.py
Then:   cmake --build build --target filmicrgb   # verify constants + build
"""
import os, re, subprocess, sys

AB_STAB = 70.0                                  # extra-bleach apparent-brightness UNIFORMITY weight (target-free)
AB_LEVEL = 10.0                                 # gentle pull of the MEAN AB toward the target level (kept LOW:
                                                # high level-weight makes the fit hyper-sensitive to the exact
                                                # ab target — a 0.003 change flips blue into a distorting basin)
BLEACH_NUGDE = 0.5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "tools", "derive_filmic_agx_primaries.py")
C = os.path.join(ROOT, "src", "iop", "filmicrgb.c")
DERIVE = [sys.executable, "-u", PY]

# (variant name, C enum suffix, fit flags, human-readable fit string for the SHIPPED entry)
STEPS = [
    ("extra-bleach",  "V10", ["--fit-extra-bleach", "--ab-stabilize", str(AB_STAB), "--ab-level", str(AB_LEVEL), "--bleach-nudge", str(BLEACH_NUGDE)],
     "--fit-extra-bleach --ab-stabilize %g --ab-level %g --bleach-nudge %g" % (AB_STAB, AB_LEVEL, BLEACH_NUGDE)),
    ("medium-bleach", "V8",  ["--fit-bisect", "no-bleach", "extra-bleach"],
     "--fit-bisect no-bleach extra-bleach"),
    ("low-bleach",    "V7",  ["--fit-bisect", "no-bleach", "medium-bleach"],
     "--fit-bisect no-bleach medium-bleach"),
    ("high-bleach",   "V9",  ["--fit-bisect", "medium-bleach", "extra-bleach"],
     "--fit-bisect medium-bleach extra-bleach"),
]


def run(flags):
    r = subprocess.run(DERIVE + flags, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(flags), r.stdout + r.stderr))
    return r.stdout


def parse_params(out):
    """Pull inset/irot/outset/orot float lists from the 'paste this into SHIPPED_VARIANTS' block."""
    if "paste this into SHIPPED_VARIANTS" not in out:
        sys.exit("no SHIPPED block in fit output:\n" + out[-2000:])
    blk = out.split("paste this into SHIPPED_VARIANTS", 1)[1]
    def arr(key):
        m = re.search(key + r"=\[([^\]]+)\]", blk)
        return [float(x) for x in m.group(1).split(",")]
    return arr("inset"), arr("irot"), arr("outset"), arr("orot")


def patch_shipped(text, name, fit, ins, iro, outs, oro):
    entry = ('    "%s": dict(\n'
             '        fit="%s",\n'
             '        inset=[%s],\n'
             '        irot=[%s],\n'
             '        outset=[%s],\n'
             '        orot=[%s]\n'
             '    ),' % (name, fit,
                         ", ".join("%.7f" % v for v in ins),
                         ", ".join("%.7f" % v for v in iro),
                         ", ".join("%.6f" % v for v in outs),
                         ", ".join("%.7f" % v for v in oro)))
    pat = re.compile(r'    "%s":\s*dict\(.*?\n    \),' % re.escape(name), re.S)
    new, n = pat.subn(lambda m: entry, text, count=1)
    if n != 1:
        sys.exit("could not patch SHIPPED_VARIANTS['%s']" % name)
    return new


def patch_c(text, vnum, ins, iro, outs, oro):
    block = ('      inset_anchor[0] = %+.7ff; inset_anchor[1] = %+.7ff; inset_anchor[2] = %+.7ff;\n'
             '      rotation_anchor[0] = %+.7ff; rotation_anchor[1] = %+.7ff; rotation_anchor[2] = %+.7ff;\n'
             '      outset_anchor[0]   = %.6ff; outset_anchor[1] = %.6ff; outset_anchor[2] = %.6ff;\n'
             '      outset_rotation[0] = %+.7ff; outset_rotation[1] = %+.7ff; outset_rotation[2] = %+.7ff;'
             % (ins[0], ins[1], ins[2], iro[0], iro[1], iro[2],
                outs[0], outs[1], outs[2], oro[0], oro[1], oro[2]))
    pat = re.compile(
        r'(case DT_FILMIC_COLORSCIENCE_%s:.*?)'
        r'      inset_anchor\[0\][^\n]*\n'
        r'      rotation_anchor\[0\][^\n]*\n'
        r'      outset_anchor\[0\][^\n]*\n'
        r'      outset_rotation\[0\][^\n]*;' % vnum, re.S)
    new, n = pat.subn(lambda m: m.group(1) + block, text, count=1)
    if n != 1:
        sys.exit("could not patch C case %s" % vnum)
    return new


def main():
    for name, vnum, flags, fitstr in STEPS:
        print("=== fitting %-14s (%s) : %s" % (name, vnum, " ".join(flags)))
        out = run(flags)
        for ln in out.splitlines():
            if ln.startswith("// ") and ("bisect(" in ln or "extra best" in ln):
                print("   " + ln)
        ins, iro, outs, oro = parse_params(out)
        with open(PY) as f:
            py = f.read()
        with open(PY, "w") as f:
            f.write(patch_shipped(py, name, fitstr, ins, iro, outs, oro))
        with open(C) as f:
            cc = f.read()
        with open(C, "w") as f:
            f.write(patch_c(cc, vnum, ins, iro, outs, oro))
        print("   patched SHIPPED_VARIANTS['%s'] + C %s  inset=[%s]"
              % (name, vnum, ", ".join("%.4f" % v for v in ins)))

    diag = run(["--diagnose"])
    rep = run(["--report"])
    with open(os.path.join(ROOT, "tools", "agx_diagnose.log"), "w") as f:
        f.write(diag)
    with open(os.path.join(ROOT, "tools", "agx_report.md"), "w") as f:
        f.write(rep)
    print("\n=== per-hue continuity (tools/agx_diagnose.log) ===")
    print(diag)
    print("Wrote tools/agx_report.md . Now build + verify:\n"
          "  cmake --build build --target filmicrgb")


if __name__ == "__main__":
    main()
