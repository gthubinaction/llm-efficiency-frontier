"""
verify_manuscript.py — checks every numeric claim in the manuscript against the
pipeline's results.json, and checks that repeated claims agree with each other.

    python verify_manuscript.py Manuscript.docx results.json

Why this exists. Across v8, v9 and v10 the algebraic core was repeatedly verified
correct while the presentation layer drifted: text corrected but figures stale,
cross-references pointing at renumbered sections, and — in v10 — a ratio stated
correctly in the results section and inverted in the abstract, discussion and
conclusions. Manual propagation of a corrected number into four places has a high
per-pass error rate that does not fall with care. This script makes the checks
mechanical.

Three classes of check:

  VALUE      a number in the text must match the pipeline within tolerance
  DIRECTION  a ratio must point the way the pipeline says it points
  ECHO       a quantity stated in more than one place must agree everywhere

Exit status is non-zero if any check fails, so it can gate a build.
"""
import json
import re
import sys
import unicodedata
from docx import Document

TOL = 0.02          # 2% relative tolerance on values quoted to 2-3 significant figures

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")

def text_of(path):
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    t = unicodedata.normalize("NFKC", "\n".join(parts))
    return re.sub(r"[ \t]+", " ", t)

def to_float(s):
    """Parse 2.35×10¹⁹, 4.1x10^8, 5.8e9, 0.85, 15%."""
    s = s.strip().replace(",", "")
    s = s.translate(SUP)
    m = re.match(r"^([\d.]+)\s*[×x]\s*10\s*\^?\s*(-?\d+)$", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return float(s.rstrip("%"))

def find_all(text, pattern):
    return [m for m in re.finditer(pattern, text)]

class Report:
    def __init__(self):
        self.rows = []
    def add(self, kind, name, ok, detail):
        self.rows.append((kind, name, ok, detail))
    def failed(self):
        return [r for r in self.rows if not r[2]]
    def show(self):
        width = max(len(r[1]) for r in self.rows)
        for kind, name, ok, detail in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {kind:9s} {name:<{width}s}  {detail}")
        bad = self.failed()
        print("\n" + ("-" * 70))
        print(f"{len(self.rows) - len(bad)} passed, {len(bad)} failed")
        return len(bad)

def check_value(rep, text, name, pattern, expected, tol=TOL, unit=""):
    hits = find_all(text, pattern)
    if not hits:
        rep.add("VALUE", name, False, "claim not found in manuscript")
        return
    vals = [to_float(h.group(1)) for h in hits]
    bad = [v for v in vals if expected == 0 or abs(v - expected) / abs(expected) > tol]
    ok = not bad
    rep.add("VALUE", name, ok,
            f"found {len(vals)}x: {[f'{v:.4g}' for v in vals]} vs pipeline {expected:.4g}{unit}"
            + ("" if ok else "  <-- MISMATCH"))

def check_echo(rep, text, name, pattern, expected_count=None):
    """A quantity repeated across sections must be identical everywhere."""
    vals = [to_float(m.group(1)) for m in find_all(text, pattern)]
    if not vals:
        rep.add("ECHO", name, False, "no occurrence found")
        return
    ok = len(set(round(v, 6) for v in vals)) == 1
    if expected_count and len(vals) != expected_count:
        ok = False
    rep.add("ECHO", name, ok, f"{len(vals)} occurrence(s): {[f'{v:.4g}' for v in vals]}"
            + ("" if ok else "  <-- DISAGREE"))

def check_direction(rep, text, name, phrases_ok, phrases_bad):
    """Ratios must point the way the pipeline says. Any 'bad' phrasing fails."""
    found_bad = [p for p in phrases_bad if re.search(p, text)]
    found_ok = [p for p in phrases_ok if re.search(p, text)]
    ok = not found_bad and bool(found_ok)
    detail = f"correct phrasings: {len(found_ok)}; inverted phrasings: {len(found_bad)}"
    if found_bad:
        detail += f"  <-- INVERTED: {found_bad}"
    rep.add("DIRECTION", name, ok, detail)

def check_absent(rep, text, name, pattern, why):
    hits = find_all(text, pattern)
    rep.add("ABSENT", name, not hits, (why if hits else "clean")
            + (f"  ({len(hits)} occurrence(s))" if hits else ""))

def main(doc_path, results_path):
    text = text_of(doc_path)
    R = json.load(open(results_path))
    rep = Report()

    # ---------- frozen parameters and derived geometry ----------
    fz = R["frozen_parameters"]
    check_value(rep, text, "alpha", r"α = ([\d.]+)", fz["alpha"], tol=0.005)
    check_value(rep, text, "beta", r"β = ([\d.]+)", fz["beta"], tol=0.005)
    check_value(rep, text, "compute exponent",
                r"C ∝ N\^?([\d.]+)", R["compute_exponent"], tol=0.01)
    check_value(rep, text, "N-doubling factor",
                r"([\d.]+)-fold (?:increase in compute|resource increase)",
                R["N_doubling_compute_factor"], tol=0.01)

    # ---------- marginal gains ----------
    g = R["gains_per_compute_doubling"]
    check_value(rep, text, "gain at 1e8", r"from ([\d.]+)% at N = 10", g["1e+08"] * 100)
    check_value(rep, text, "gain at 1e10", r"to (1\.7\d?)% at 10", g["1e+10"] * 100, tol=0.03)

    # ---------- thresholds ----------
    th = R["thresholds"]
    check_value(rep, text, "threshold 4%", r"(4\.1[x×]10\^?8)", th["4%"]["N"], tol=0.03)
    check_echo(rep, text, "threshold 4% echoed", r"(4\.1)[x×]10")
    check_echo(rep, text, "threshold 2% echoed", r"(5\.8)[x×]10")

    # ---------- envelope comparison: value AND direction ----------
    v = R["empirical_validation"]["thresholds_model_vs_empirical"]
    ratios = [d["compute_ratio"] for d in v.values()]
    r_mean = sum(ratios) / len(ratios)
    check_value(rep, text, "path/envelope ratio",
                r"([\d.]+) times the envelope's", r_mean, tol=0.02)
    check_echo(rep, text, "ratio stated consistently", r"(0\.8\d) times the envelope")
    # the pipeline says path < envelope; any phrasing that says the reverse is an error
    check_direction(
        rep, text, "ratio direction",
        phrases_ok=[r"path's budget is 0\.8\d times the envelope",
                    r"envelope[^.]{0,80}(larger|higher|more) (compute|budget)",
                    r"model stops [\d.]+% earlier in budget"],
        phrases_bad=[r"0\.8\d times the model's",
                     r"stopping budget at 0\.8\d times the model",
                     r"within a factor 0\.8\d"])

    # ---------- functional form ----------
    ec = R["empirical_validation"]["envelope_compute_matched"]
    check_value(rep, text, "power LOOCV", r"leave-one-out RMSE of ([\d.]+)",
                ec["power"]["LOOCV"], tol=0.05)
    check_value(rep, text, "log LOOCV", r"LOOCV RMSE (0\.10\d+)\)", ec["log"]["LOOCV"], tol=0.05)
    check_value(rep, text, "power R2 on envelope", r"attains R2 = (0\.99\d)", ec["power"]["R2"], tol=0.005)
    dAIC = ec["log"]["AIC"] - ec["power"]["AIC"]
    check_value(rep, text, "delta AIC", r"ΔAIC ≈ (\d+)", dAIC, tol=0.05)

    # ---------- allocation diagnostic ----------
    al = R["decomposition"]["allocation"]
    full = al["sensitivity"]["full_sample"]
    excl = al["sensitivity"]["excluding_5_high_loss"]
    check_value(rep, text, "allocation median", r"median ratio is ([\d.]+)", full["median"], tol=0.01)
    check_value(rep, text, "allocation p90", r"ninetieth percentile ([\d.]+)", full["p90"], tol=0.02)
    check_value(rep, text, "allocation max", r"the maximum ([\d.]+)", full["max"], tol=0.02)
    check_value(rep, text, "share above 2x", r"(\d+)% of runs above a factor of two",
                full["share_above_2x"] * 100, tol=0.15)
    check_value(rep, text, "max excluding outliers", r"maximum to ([\d.]+)", excl["max"], tol=0.05)

    # ---------- claims that must not reappear ----------
    check_absent(rep, text, "old doubling factor", r"3\.4-fold",
                 "superseded by 3.85 (came from the discarded v2 estimator)")
    check_absent(rep, text, "three-decimal claim", r"to three decimal",
                 "replication is close, not exact to three decimals")
    check_absent(rep, text, "bracket language", r"bracketing of extrapolation|extrapolation bracket",
                 "the bracket was withdrawn; title and text must not reintroduce it")
    check_absent(rep, text, "stale LOOCV", r"0\.173|0\.155|elevenfold|twelvefold",
                 "superseded by the pipeline value")
    check_absent(rep, text, "unsupported cost claim", r"cheaper per unit of performance",
                 "alternatives' costs and performance were not compared")
    check_absent(rep, text, "precision hierarchy", r"(set|governed) by the economics",
                 "varying the hurdle shows sensitivity, not that economic uncertainty dominates")
    check_absent(rep, text, "truncated c_eff sentence",
                 r"the effective price of compute is\s*\n",
                 "sentence left incomplete by an earlier edit")
    check_absent(rep, text, "bare V where V1 is meant", r"(?<![₁\w])V is organization-specific",
                 "the adopted rule uses V₁")
    check_absent(rep, text, "overbroad scope", r"at every hurdle",
                 "only three hurdles were examined")


    # ---------- hurdle <-> value correspondence (Table 6) ----------
    corr = R.get("hurdle_value_correspondence", {})
    for h, lbl in [("4%", "12.76"), ("3%", "176"), ("2%", "5,065")]:
        if h in corr:
            check_value(rep, text, f"V1* at {h}",
                        r"≈?\$(" + lbl.replace(",", ",?") + r")",
                        corr[h]["V1_star"], tol=0.03)
    if "economic_parameters" in R:
        check_value(rep, text, "c_eff printed",
                    r"c_eff = ([\d.]+)×10", R["economic_parameters"]["c_eff"] * 1e18, tol=0.001)

    # ---------- band sensitivity must be reported ----------
    bs = R.get("envelope_band_sensitivity", {})
    if bs:
        ok_all = all(v.get("power_preferred") for v in bs.values() if "error" not in v)
        rep.add("VALUE", "band sensitivity reported",
                bool(re.search(r"10, 12, 15, 20 and 25", text)) and ok_all,
                f"power preferred in {sum(1 for v in bs.values() if v.get('power_preferred'))}/{len(bs)} binnings")


    # ---------- scope qualifiers must travel with the headline number ----------
    # A number can be right and still overclaim. The band-sensitivity analysis
    # shows the sign of the model-envelope difference is not stable, so every
    # place that states the 18% figure must also carry a qualifier.
    para_hits = [seg for seg in text.split("\n") if "18%" in seg]
    QUAL = r"(main specification|binning|within about 20%|not stable|three hurdles)"
    unqualified = [s_[:70] for s_ in para_hits if not re.search(QUAL, s_)]
    rep.add("SCOPE", "18% qualified everywhere", not unqualified,
            f"{len(para_hits)} mention(s), {len(unqualified)} without a scope qualifier"
            + ("" if not unqualified else f"  <-- {unqualified}"))

    # sample composition must be explained where the table is discussed
    sc = R.get("sample_composition", {})
    if sc:
        rep.add("SCOPE", "sample overlap explained",
                bool(re.search(r"overlap|also among the five highest-loss", text)),
                f"rows {sc['n_row1']} and {sc['n_row2']}, overlap {sc['overlap_indices']}")


    # ---------- figures must carry the same numbers as the text ----------
    # Text can be corrected while an image keeps a superseded value; that is how a
    # stale LOOCV survived two revisions. The embedded PNGs are checked for the
    # metrics they print by extracting them from the docx and reading the legend
    # region is out of scope here, so instead we assert that the figure files on
    # disk are newer than the results file they must reflect.
    import os
    figdir = os.path.join(os.path.dirname(results_path) or ".", "..", "figures")
    figdir = os.path.normpath(figdir)
    if os.path.isdir(figdir):
        rts = os.path.getmtime(results_path)
        stale = [f for f in os.listdir(figdir) if f.endswith(".png")
                 and os.path.getmtime(os.path.join(figdir, f)) < rts - 60]
        rep.add("FIGURE", "figures newer than results", not stale,
                "all regenerated after the last results run" if not stale
                else f"older than results.json: {stale}")

    n_bad = rep.show()
    if n_bad:
        print("\nThe manuscript does not agree with the pipeline. Fix the FAIL rows above.")
    else:
        print("\nManuscript agrees with results.json on every checked quantity.")
    return 1 if n_bad else 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
