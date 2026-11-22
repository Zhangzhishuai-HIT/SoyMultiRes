# Data card (SoyMultiRes)

Built by `scripts/01_data/build_pheno.py` (step 1 of the plan). Sources: GRIN-Global descriptor downloads in
`down_data/GRIN_soybean_descriptors/` (MANIFEST.tsv; definition pages in `definitions_20261111/`) and the SoySNP50K
VCF sample list (`down_data/LIS_glycine_max_diversity/Wm82.gnm2.div.Song_Hyten_2015`).

## Rules

- Accessions are matched by the raw VCF sample id (GRIN prefix + number + suffix without spaces, e.g. PI407999-1).
- Every score is oriented so that **higher = more resistant/tolerant**. Categorical codes are ordinal levels (table below); numeric damage/severity scores are negated.
- BRNSTEMROT U = susceptible in preliminary field tests (coded with S); PMV U = field-resistant, not confirmed (between S and R); PHYTOROT/PYTHIUMROT H = heterogeneous; NSCANKER/HIGHTEMP I = intermediate; BPMV/SALTREACT T = tolerant.
- `resistant` = GRIN resistant classes for categorical traits; for numeric traits the least-damaged 10% of an evaluation batch (ties included; e.g. FROGEYE2 score 1 = 'resistant with no lesions' covers 66%).
- Records of the same accession in the same batch are averaged; batches stay separate in `pheno_long.tsv.gz`.
- RUST_TAN: numeric 1-5, higher = more severe (GRIN method SOYBEAN.RUST.RF.2004, saved as definitions_20261111/m492634.html: 1 = no visible symptoms, 5 = severe infection); negated for the models.
- Mendelian calibration traits use HapGAM's binary codings (flower purple vs white, pubescence tawny vs gray, indeterminate vs determinate, seed coat pigmented vs yellow) and maturity group 000 = -2 ... X = 10.

## 33 stress traits

| Trait | Short | Category | Accessions | Batches | Levels (low → high resistance) | Resistant share |
|---|---|---|---:|---:|---|---:|
| NEMATCYST_1 | SCN1 | SCN | 1,586 | 6 | S < MS < MR < R | 0.056 |
| NEMATCYST_3 | SCN3 | SCN | 12,774 | 10 | S < MS < MR < R | 0.022 |
| NEMATCYST_4 | SCN4 | SCN | 7,395 | 3 | S < MS < MR < R | 0.012 |
| NEMATCYST_5 | SCN5 | SCN | 11,758 | 7 | S < MS < MR < R | 0.008 |
| NEMATCYST_14 | SCN14 | SCN | 2,495 | 6 | S < MS < MR < R | 0.015 |
| PHYTOROT_1 | PR1 | Phytophthora | 9,947 | 10 | S < H < R | 0.402 |
| PHYTOROT_3 | PR3 | Phytophthora | 3,260 | 10 | S < H < R | 0.511 |
| PHYTOROT_4 | PR4 | Phytophthora | 1,469 | 7 | S < H < R | 0.485 |
| PHYTOROT_5 | PR5 | Phytophthora | 790 | 6 | S < H < R | 0.424 |
| PHYTOROT_7 | PR7 | Phytophthora | 2,968 | 9 | S < H < R | 0.457 |
| PHYTOROT_10 | PR10 | Phytophthora | 622 | 1 | S < H < R | 0.360 |
| PHYTOROT_12 | PR12 | Phytophthora | 639 | 1 | S < H < R | 0.083 |
| PHYTOROT_17 | PR17 | Phytophthora | 2,226 | 3 | S < H < R | 0.315 |
| PHYTOROT_20 | PR20 | Phytophthora | 651 | 1 | S < H < R | 0.098 |
| PHYTOROT_25 | PR25 | Phytophthora | 2,834 | 4 | S < H < R | 0.341 |
| SDS | SDS | fungi_oomycetes | 6,850 | 3 | S < MS < MR | 0.012 |
| BRNSTEMROT | BSR | fungi_oomycetes | 4,025 | 4 | U < S < R < MR | 0.037 |
| NSCANKER | NSC | fungi_oomycetes | 1,465 | 2 | S < MS < I < MR < R | 0.735 |
| FROGEYE2 | FE2 | fungi_oomycetes | 2,652 | 4 | −score (numeric) | 0.661 |
| FROGEYE3 | FE3 | fungi_oomycetes | 1,677 | 1 | −score (numeric) | 0.294 |
| RUST_TAN | RUST | fungi_oomycetes | 3,082 | 1 | −score (numeric) | 0.350 |
| PYTHIUMROT | PYT | fungi_oomycetes | 1,289 | 2 | S < H < R | 0.046 |
| BP | BP | bacteria | 3,389 | 1 | −score (numeric) | 0.208 |
| PMV | PMV | virus | 2,150 | 1 | S < U < R | 0.011 |
| BPMV | BPMV | virus | 540 | 4 | S < T | 0.043 |
| MEXBEANBTL | MBB | chewing_insects | 5,037 | 10 | −score (numeric) | 0.481 |
| LOOPER | SBL | chewing_insects | 2,653 | 2 | −score (numeric) | 0.146 |
| VBC | VBC | chewing_insects | 534 | 1 | −score (numeric) | 0.195 |
| APHID | APH | sucking_insects | 3,657 | 9 | S < MR < R | 0.019 |
| POTATOLHOP | PLH | sucking_insects | 782 | 1 | −score (numeric) | 0.643 |
| CHLOROSIS | IDC | abiotic | 1,992 | 8 | −score (numeric) | 0.298 |
| SALTREACT | SALT | abiotic | 563 | 1 | S < T | 0.266 |
| HIGHTEMP | HT | abiotic | 520 | 1 | S < I < T | 0.233 |

## Mendelian calibration traits

| Trait | Accessions | Positive share |
|---|---:|---:|
| flower_purple | 17,044 | 0.666 |
| pubescence_tawny | 16,969 | 0.536 |
| indeterminate | 15,275 | 0.453 |
| seedcoat_pigmented | 17,028 | 0.230 |
| maturity_group | 17,069 |  |

Accession table: 20,087 genotyped accessions; 17,150 with ≥1 stress trait; taxon {'Glycine max': 18554, 'Glycine soja': 1110}.
