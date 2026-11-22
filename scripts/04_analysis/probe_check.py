"""Uniqueness of the sequence around SNPs of the SoySNP50K array in soybean genome assemblies (Supplementary Note 4).

usage: python -I probe_check.py [--controls 500] [--threads 4]
SNPs: the lead SNPs of the two race-differential loci (ss715606985, chromosome 10; ss715628625, chromosome 18) and,
as a reference, --controls SNPs drawn at random from the 39,627 analysed SNPs (seed 20261118). Query: the 121 bp
around the SNP on Wm82.a2 (60 bp on each side; the reference base is checked against the SNP table), and the 50 bp
on either side of the SNP (the length of an Infinium probe). Search: BLASTN 2.17.0 (-task blastn, e-value 1e-5) in
the Wm82.a2, Wm82.a4 and Peking (gnm3) assemblies of the Legume Information System (genome_main files). A match
counts when it covers at least 90% of the query with at least 90% identity. The match at the SNP's own position is
the one on the same chromosome (Wm82.a2: within 200 bp of the SNP; other assemblies: the best match on the
chromosome of the same number); all other matches are secondary. For the own match, the base aligned to the SNP is
reported (reference or alternative allele of the assembly).
Writes results/loci/probe_check.tsv (one row per SNP, assembly and query) and prints a summary.
"""
import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soymultires import data  # noqa: E402

GEN = ROOT / "data/raw/lis_glycine_max_genomes"    # Glycine max genomes of the Legume Information System (symbolic link)
ASSEMBLIES = {"Wm82.a2": "Wm82.gnm2.DTC4", "Wm82.a4": "Wm82.gnm4.4PTR", "Peking": "Peking.gnm3.KKCD"}
LEAD = {"ss715606985": "chromosome 10 lead SNP", "ss715628625": "chromosome 18 lead SNP"}
BIN = Path(sys.executable).parent
FLANK, PROBE = 60, 50


def chrom_no(name):
    """Number of the chromosome in a sequence name (glyma.Wm82.gnm2.Gm10, glyma.Peking.gnm3.chr10), else -1."""
    tail = name.split(".")[-1]
    for p in ("Gm", "chr"):
        if tail.startswith(p) and tail[len(p):].isdigit():
            return int(tail[len(p):])
    return -1


def read_fasta(path, keep):
    seqs, name, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name in keep:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif name in keep:
                buf.append(line.strip().upper())
    if name in keep:
        seqs[name] = "".join(buf)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    snps = pd.read_csv(data.P / "geno/snps.tsv", sep="\t")
    rng = np.random.default_rng(20261118)
    ctrl = snps[~snps.id.isin(LEAD)].iloc[np.sort(rng.choice(len(snps) - len(LEAD), a.controls, replace=False))]
    sel = pd.concat([snps[snps.id.isin(LEAD)], ctrl])
    a2 = read_fasta(GEN / ASSEMBLIES["Wm82.a2"] / f"glyma.{ASSEMBLIES['Wm82.a2']}.genome_main.fna.gz",
                    {f"glyma.Wm82.gnm2.{c}" for c in sel.chrom.unique()})
    work = ROOT / "work/probe_check"
    work.mkdir(parents=True, exist_ok=True)
    queries, info = [], {}
    for _, s in sel.iterrows():
        chrom = a2[f"glyma.Wm82.gnm2.{s.chrom}"]
        i = int(s.pos) - 1                                    # 0-based position of the SNP
        assert chrom[i] == s.ref, (s.id, chrom[i], s.ref)     # positions of the SNP table are Wm82.a2
        q = {"121bp": chrom[i - FLANK:i + FLANK + 1], "left50": chrom[i - PROBE:i], "right50": chrom[i + 1:i + 1 + PROBE]}
        for k, seq in q.items():
            queries.append(f">{s.id}|{k}\n{seq}\n")
            info[f"{s.id}|{k}"] = {"snp": s.id, "chrom": s.chrom, "pos": int(s.pos), "ref": s.ref, "alt": s.alt,
                                   "query": k, "query_length": len(seq), "role": LEAD.get(s.id, "control")}
    qf = work / "queries.fa"
    qf.write_text("".join(queries))
    rows = []
    for asm, tag in ASSEMBLIES.items():
        db = work / tag
        if not Path(f"{db}.njs").exists() and not Path(f"{db}.nal").exists():
            fa = GEN / tag / f"glyma.{tag}.genome_main.fna.gz"
            subprocess.run(f"zcat {fa} | {BIN}/makeblastdb -in - -dbtype nucl -title {tag} -out {db}", shell=True, check=True)
        out = work / f"hits_{tag}.tsv"
        subprocess.run([str(BIN / "blastn"), "-task", "blastn", "-query", str(qf), "-db", str(db), "-evalue", "1e-5",
                        "-max_target_seqs", "100", "-max_hsps", "100", "-num_threads", str(a.threads), "-dust", "no",
                        "-outfmt", "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qseq sseq",
                        "-out", str(out)], check=True)
        h = pd.read_csv(out, sep="\t", header=None, names=["q", "s", "pident", "length", "qstart", "qend", "sstart", "send",
                                                            "evalue", "bitscore", "qseq", "sseq"])
        for qid, d in info.items():
            hh = h[h.q == qid]
            cov = (hh.qend - hh.qstart + 1) / d["query_length"]
            hh = hh[(cov >= 0.9) & (hh.pident >= 90)].copy()
            same = hh.s.map(chrom_no) == int(d["chrom"][2:])
            if asm == "Wm82.a2":
                own_mask = same & (np.minimum(hh.sstart, hh.send) <= d["pos"] + 200) & (np.maximum(hh.sstart, hh.send) >= d["pos"] - 200)
            else:
                own_mask = pd.Series(False, index=hh.index)
                if same.any():
                    own_mask.loc[hh[same].bitscore.idxmax()] = True
            own, sec = hh[own_mask], hh[~own_mask]
            base = ""
            if d["query"] == "121bp" and len(own):
                o = own.sort_values("bitscore").iloc[-1]
                if o.qstart <= FLANK + 1 <= o.qend:                # base aligned to query position 61 (the SNP); sseq
                    qpos = o.qstart - 1                            # is given in the orientation of the query
                    for qa, sa in zip(o.qseq, o.sseq):
                        if qa != "-":
                            qpos += 1
                            if qpos == FLANK + 1:
                                base = sa
                                break
            rows.append({**d, "assembly": asm, "own_match": len(own) > 0,
                         "own_position": f"{own.s.iloc[0].split('.')[-1]}:{min(own.sstart.iloc[0], own.send.iloc[0])}" if len(own) else "",
                         "own_identity": float(own.pident.max()) if len(own) else np.nan, "base_at_snp": base,
                         "allele_at_snp": "ref" if base == d["ref"] else "alt" if base == d["alt"] else ("" if base == "" else "other"),
                         "secondary_matches": len(sec),
                         "secondary": "; ".join(f"{r.s.split('.')[-1]}:{min(r.sstart, r.send)} ({r.pident:.1f}%, {r.length} bp)"
                                                for r in sec.sort_values("bitscore", ascending=False).head(10).itertuples())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/loci/probe_check.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    lead = out[out.role != "control"]
    print(lead[["snp", "role", "assembly", "query", "own_match", "own_position", "own_identity", "allele_at_snp", "secondary_matches", "secondary"]].to_string(index=False))
    c = out[out.role == "control"]
    print("controls with at least one secondary match (fraction of SNPs):")
    print(c.groupby(["assembly", "query"]).secondary_matches.apply(lambda v: f"{(v > 0).mean():.3f} of {len(v)}").to_string())
    print("controls without an own match:")
    print(c.groupby(["assembly", "query"]).own_match.apply(lambda v: f"{(~v).mean():.3f}").to_string())


if __name__ == "__main__":
    main()
