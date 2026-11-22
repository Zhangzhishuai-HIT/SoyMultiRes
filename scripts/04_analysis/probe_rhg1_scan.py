"""Copies of the sequence around the chromosome 10 lead SNP (ss715606985) in the Rhg1 region of soybean genome
assemblies, compared with the SoySNP50K calls of the same cultivars (Supplementary Note 4).

usage: python -I probe_rhg1_scan.py [--workers 4]
Assemblies: all Glycine max genome assemblies of the Legume Information System (genome_main files of
data/raw/lis_glycine_max_genomes). Chromosomes 10 and 18 of every assembly are searched with BLASTN 2.17.0
(-task blastn, e-value 1e-5, -subject) for (i) the Rhg1 repeat: matches of the Wm82.a2 coding sequence of the
alpha-SNAP gene Glyma.18G022500 with at least 98% identity over at least 140 bp, grouped into copies 10 kb or more
apart (an assembly may collapse identical repeat copies, so this is a lower bound of the copy number); (ii) the 121 bp
around the SNP on Wm82.a2 (probe_check.py, SNP at position 61). A copy of a probe site is a local alignment that covers
the 50 bp on one side of the SNP (positions 11-60, or 62-111) with at most two mismatches or gap columns; its base
is the base of the assembly next to these 50 bp in the alignment (the base that the single-base extension of an
Infinium probe on that side would read), given in the orientation of Wm82.a2. A copy is in the Rhg1 region when it
lies on chromosome 18 within 100 kb of the alpha-SNAP matches. A copy of the 121 bp (at least 85% identity, covering the SNP) lies
in the first intron of alpha-SNAP when it lies between a match of coding exon 1 (CDS 1-145) and one of coding exon 2 (CDS
146-195) of the same orientation at most 10 kb apart (column copies_in_alpha_snap_intron1, with the intron lengths; the first
intron of Glyma.18G022500 is 418 bp in Wm82.a2). Cultivars are matched to the genotyped accessions by their GRIN common name (PIconversion file of
the SoySNP50K data set) or, for assemblies named by PI number, by that number.
Writes results/loci/probe_rhg1_scan.tsv (one row per assembly).
"""
import argparse
import gzip
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

GEN = ROOT / "data/raw/lis_glycine_max_genomes"    # Glycine max genomes of the Legume Information System (symbolic link)
BIN = Path(sys.executable).parent
SNP = "ss715606985"
WORK = ROOT / "work/probe_check"
OUTFMT = "6 qseqid sseqid pident length qstart qend sstart send bitscore qseq sseq"


def queries():
    """121 bp around the SNP (from probe_check.py) and the Wm82.a2 CDS of Glyma.18G022500."""
    q = (WORK / "queries.fa").read_text().split(">")
    snp = [">" + x for x in q if x.startswith(f"{SNP}|121bp")][0]
    cds_file = GEN / "Wm82.gnm2.ann1.RVB6/glyma.Wm82.gnm2.ann1.RVB6.cds_primary.fna.gz"
    keep, buf = False, []
    with gzip.open(cds_file, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                keep = "Glyma.18G022500" in line
                if keep:
                    buf.append(">alphaSNAP\n")
            elif keep:
                buf.append(line)
    assert buf
    return snp.replace(f"{SNP}|121bp", "snp121"), "".join(buf)


def scan(tag):
    fa = GEN / tag / f"glyma.{tag}.genome_main.fna.gz"
    tmp = WORK / "dbs"
    tmp.mkdir(parents=True, exist_ok=True)
    sub = tmp / f"{tag}.chr10_18.fa"                        # chromosomes 10 and 18 only
    subprocess.run(f"zcat {fa} | awk '/^>/{{p = ($1 ~ /(Gm|Chr|chr)(10|18)$/)}} p' > {sub}", shell=True, check=True)
    qf = tmp / f"{tag}.query.fa"
    snp_q, snap_q = queries()
    qf.write_text(snp_q + snap_q)
    res = subprocess.run([str(BIN / "blastn"), "-task", "blastn", "-query", str(qf), "-subject", str(sub), "-evalue", "1e-5",
                          "-max_target_seqs", "100", "-max_hsps", "200", "-dust", "no", "-outfmt", OUTFMT],
                         capture_output=True, text=True, check=True).stdout
    sub.unlink()
    qf.unlink()
    h = pd.DataFrame([r.split("\t") for r in res.strip().split("\n") if r],
                     columns=["q", "s", "pident", "length", "qstart", "qend", "sstart", "send", "bitscore", "qseq", "sseq"])
    for c in ["pident", "length", "qstart", "qend", "sstart", "send", "bitscore"]:
        h[c] = pd.to_numeric(h[c])
    h["lo"] = h[["sstart", "send"]].min(axis=1)
    h["chrom"] = h.s.str.extract(r"(\d+)$")[0].astype(int)
    row = {"assembly": tag, "chromosomes_searched": ",".join(sorted(h.s.unique())) if len(h) else ""}
    snap = h[(h.q == "alphaSNAP") & (h.pident >= 98) & (h.length >= 140) & (h.chrom == 18)].sort_values("lo")
    if len(snap):
        pos = snap.lo.to_numpy()
        starts = [pos[0]] + [p for p0, p in zip(pos[:-1], pos[1:]) if p - p0 >= 10_000]
        row.update({"rhg1_start": int(pos.min()), "rhg1_end": int(pos.max()), "alpha_snap_copies": len(starts)})
    s = h[h.q == "snp121"]

    def probe_site(r, side):
        """(differences in the 50 bp, extension base) of one alignment for the left (11-60) or right (62-111) probe."""
        lo, hi = (11, 60) if side == "left" else (62, 111)
        if not (r.qstart <= lo and r.qend >= hi):
            return None
        qpos, diff, cols = r.qstart - 1, 0, []
        for qa, sa in zip(r.qseq, r.sseq):
            inside = lo <= (qpos + (qa != "-")) <= hi and not (qa == "-" and qpos in (lo - 1, hi))
            if qa != "-":
                qpos += 1
            cols.append((qpos, qa, sa))
            if inside and qa != sa:
                diff += 1
        if side == "left":                                   # first assembly base after the column of query position 60
            k = max(i for i, (p, qa, _) in enumerate(cols) if p == 60 and qa != "-")
            nxt = [sa for _, _, sa in cols[k + 1:] if sa != "-"]
        else:                                                # last assembly base before the column of query position 62
            k = min(i for i, (p, qa, _) in enumerate(cols) if p == 62 and qa != "-")
            nxt = [sa for _, _, sa in cols[:k] if sa != "-"][::-1]
        return diff, (nxt[0] if nxt else "")

    # copies of the 121 bp on chromosome 18 that lie in the first intron of an alpha-SNAP copy: between a match of coding
    # exon 1 (CDS 1-145) and a match of coding exon 2 (CDS 146-195) of Glyma.18G022500, at most 10 kb apart
    e1 = h[(h.q == "alphaSNAP") & (h.chrom == 18) & (h.qstart <= 3) & (h.qend >= 140) & (h.pident >= 95)]
    e2 = h[(h.q == "alphaSNAP") & (h.chrom == 18) & (h.qstart.between(144, 150)) & (h.qend >= 185) & (h.pident >= 95)]
    intr = []
    for r in s[(s.chrom == 18) & (s.qstart <= 61) & (s.qend >= 61) & (s.pident >= 85)].itertuples():
        for a1 in e1.itertuples():
            for a2 in e2.itertuples():
                plus = a1.sstart < a1.send
                lo_, hi_ = (a1.send, a2.sstart) if plus else (a2.sstart, a1.send)       # intron 1 between the two exons
                if (plus == (a2.sstart < a2.send)) and 0 < hi_ - lo_ < 10_000 and lo_ < r.lo < hi_:
                    intr.append((int(r.lo), int(hi_ - lo_ - 1)))
    intr = sorted(set(intr))
    row.update({"copies_in_alpha_snap_intron1": len(intr), "alpha_snap_intron1_lengths": ",".join(str(x[1]) for x in intr)})
    for side in ("left", "right"):
        hits = []
        for r in s.itertuples():
            ps = probe_site(r, side)
            if ps is not None and ps[0] <= 2:
                hits.append((r.chrom, r.lo, ps[0], ps[1]))
        hh = pd.DataFrame(hits, columns=["chrom", "lo", "diff", "base"]).drop_duplicates(["chrom", "lo"])
        rg = (hh.chrom == 18) & (hh.lo >= row.get("rhg1_start", -1e12) - 100_000) & (hh.lo <= row.get("rhg1_end", -1e12) + 100_000)
        on10 = hh.chrom == 10
        for name, sel in (("rhg1", rg), ("chr10", on10), ("chr18_other", (hh.chrom == 18) & ~rg)):
            d = hh[sel].sort_values("lo")
            row[f"{side}50_copies_{name}"] = len(d)
            row[f"{side}50_bases_{name}"] = ",".join(f"{b_}({k_})" for b_, k_ in zip(d.base, d["diff"]))
    return row


def cultivar(tag):
    return re.sub(r"\.gnm\d+\..*$", "", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tags = sorted(p.name for p in GEN.iterdir() if p.is_dir() and ".ann" not in p.name
                  and (p / f"glyma.{p.name}.genome_main.fna.gz").exists())
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(scan, tags))
    out = pd.DataFrame(rows)
    out.insert(1, "cultivar", out.assembly.map(cultivar))
    # SoySNP50K call of the same cultivar: GRIN common name -> accession (PIconversion of the SoySNP50K data set)
    conv = pd.read_csv(ROOT / "data/raw/soysnp50k_gnm2/glyma.Wm82.gnm2.div.Song_Hyten_2015.PIconversion.txt.gz",
                       sep="\t", dtype=str, encoding="latin-1").iloc[:, :2]
    conv.columns = ["acc", "name"]
    norm = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower().replace("no_", "").replace("no.", ""))  # noqa: E731
    by_name = conv.assign(k=conv.name.map(norm)).groupby("k").acc.apply(list).to_dict()
    alias = {"Wm82": "williams82", "Wm82_IGA1008": "williams82", "Wm82_NJAU": "williams82", "PI_398296": "PI398296",
             "PI_548362": "PI548362"}
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    j = int(np.flatnonzero(snps.id.to_numpy() == SNP)[0])
    ids = data.samples()
    dos = pd.Series(np.asarray(np.load(data.P / "geno/dosage.npy", mmap_mode="r")[:, j]), index=ids)
    accs, calls = [], []
    for c in out.cultivar:
        k = alias.get(c, c)
        cand = [k] if k.startswith("PI") else by_name.get(norm(k), [])
        cand = [x if x in dos.index else x[:-1] for x in cand]             # PI548402S in the file, PI548402 here
        cand = sorted({x for x in cand if x in dos.index})
        accs.append(",".join(cand))
        calls.append(",".join(str(int(dos[x])) for x in cand))
    out["genotyped_accessions"] = accs
    out[f"{SNP}_dosage"] = calls
    out.to_csv(ROOT / "results/loci/probe_rhg1_scan.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["assembly"]).to_string(index=False))


if __name__ == "__main__":
    main()
