# SoyMultiRes: MT-Boost, joint modelling of 32 soybean stress traits

Code for the article "MT-Boost: joint modelling of 32 soybean stress traits reveals shared and race-differential
resistance loci". Input data and results are not included; all
inputs are public (see Data).

## Model

MT-Boost predicts the standardized score of stress trait *t* for an accession with SNP genotypes **x** as

    F_t(x) = s_t(x) + η_s Σ_{m ≤ S_t} v_{m,t}(x) + η_p Σ_{m ≤ P_t} u_{m,t}(x)

where s_t is the starting value (zero or cross-validated GBLUP predictions), v_{m,t} are shared trees and
u_{m,t} are trait-specific trees.
- **Shared trees** are grown for every stress category (or for all traits): a split is chosen for all traits of the
  category, with the second-order split gains of the traits summed after weighting every trait equally, and every
  trait has its own leaf values.
- **Trait-specific trees** use the trait's records alone.
- **Splits.** Accessions are inbred lines, so a split separates carriers and non-carriers of the alternative allele
  of one SNP. One tree level for all traits and SNPs is a single matrix product, which runs on a GPU.
- **Per-trait choice.** For every trait, three inner folds choose the start, the number of shared rounds S_t
  (0 or the trait's best number) and the number of own rounds P_t. Fifteen Optuna trials draw the other settings,
  and every trait uses the trial with its highest inner accuracy.
- **Final prediction** of a trait: the mean of four refits, namely the chosen model and the best model without shared
  trees, each with two seeds.

Implementation: `src/soymultires/mtboost.py` (tree growth) and `scripts/03_model/train_mtboost.py` (tuning, choice,
refits).

## Names used in the code

| In the code | In the article |
|---|---|
| `mtboost_v2_final` | MT-Boost |
| `mtboost_v2_noshare4`, `mtboost_v2_noshare2` | MT-Boost without shared trees (four or two seeds) |
| `gblup`, `mtboost_v2_gblup` | GBLUP (third scenario: the GBLUP start of MT-Boost) |
| `bayesb`, `lightgbm_avg4`, `bglr_mt` | BayesB, LightGBM (mean of four seeds), multi-trait GBLUP |
| `best_single_avg4` | best single-trait model (GBLUP, BayesB or LightGBM, chosen per trait on inner folds) |
| `cv1`, `cv1_inner` | five-fold cross-validation (new accessions); inner folds |
| `cv2`, `cv3`, `all` | measured traits (second scenario); held-out experiments (third scenario); final models on all accessions |
| `NEMATCYST_<r>`, `PHYTOROT_<r>`, ... | GRIN descriptor codes (SCN race r, *Phytophthora* race r, ...; `src/soymultires/traits.py`) |

`scripts/03_model/train_mt.py`, `train_hgmt.py` and `mtgbm_probe.py`, with `src/soymultires/mtmodel.py` and
`hgmt.py`, are multi-trait models that were tried during development and are not reported in the article. The same
applies to the MegaLMM trial scripts in `scripts/02_baselines/`.

## Layout

| Path | Content |
|---|---|
| `src/soymultires/` | Python package: trait definitions, data access and training/test sets, accuracy measures, REML (from HapGAM), SNP blocks, LightGBM tuning, MT-Boost |
| `scripts/01_data/` | phenotypes from the GRIN descriptors, genotype and relationship matrices, near-identical groups, training and test sets, trait associations |
| `scripts/02_baselines/` | GBLUP, BayesB (BGLR), multi-trait GBLUP (BGLR Multitrait), LightGBM |
| `scripts/03_model/` | MT-Boost: `train_mtboost.py start` (GBLUP starts, CPU), `fit` (GPU), `reseed` (extra seeds), `collect` |
| `scripts/04_analysis/` | paired bootstrap comparisons, selection index, mixed-model association analysis (EMMAX, GPU), calibration with known genes, locus map and candidate genes, race-differential check, multi-resistance ranking |
| `scripts/05_figures/` | figures |
| `jobs/` | SLURM scripts, GPU launcher and task queue, task lists |
| `envs/` | conda environments (`environment_soymultires.yml`: Python 3.11, PyTorch 2.5.1, LightGBM 4.7, Optuna 5.0; `environment_soymultires_r.yml`: R 4.3, BGLR 1.1.4) |
| `docs/DATA_CARD.md` | trait codings, record counts and data versions |

Each script describes its inputs, outputs and usage in its first lines.

## Installation

    mamba env create -f envs/environment_soymultires.yml      # Python
    mamba env create -f envs/environment_soymultires_r.yml    # BGLR (BayesB, multi-trait GBLUP)

MT-Boost and the association analysis use a CUDA GPU when one is available; the other models run on CPU.

## Data

All inputs are public. Place them, or symbolic links to them, under `data/raw/`:

| `data/raw/...` | Content |
|---|---|
| `grin_descriptors/` | GRIN-Global descriptor tables of the soybean collection (https://npgsweb.ars-grin.gov; one CSV per descriptor) |
| `soysnp50k_gnm2/` | LIS `Wm82.gnm2.div.Song_Hyten_2015` (SoySNP50K genotypes on Wm82.a2) |
| `wm82_gnm2_annotation/` | LIS `Wm82.gnm2.ann1.RVB6` (gene models and functional annotation) |

The SoySNP50K VCF is parsed with `scripts/01_data/parse_soysnp50k.py` of the HapGAM repository
(https://github.com/Zhangzhishuai-HIT/HapGAM). `scripts/01_data/build_geno.py` and `build_pheno.py` read the
parsed files from `../HapGAM/data/interim/soysnp50k/` (variables `HG` and `SAMPLES` at the top of the scripts). The
known genes used for calibration are listed in `build_geno.py`. The job scripts in `jobs/` and `envs/create_envs.sh`
contain paths of the original machine.

## Order of the analysis

1. Data: `inventory_grin_stress.py`, `build_geno.py`, `near_identical_groups.py`, `build_pheno.py`,
   `make_splits.py`, `pheno_correlations.py`, `pair_associations.py` (`scripts/01_data/`).
2. Other models: `run_gblup.py`, `bglr_prepare.py` + `bglr_fit.R`, `bglr_mt_prepare.py` + `bglr_mt_fit.R`,
   `run_lgbm.py` (`--reseed 3` for the extra seeds, `--collect`) (`scripts/02_baselines/`).
3. MT-Boost: `train_mtboost.py start`, `fit`, `reseed`, `collect`, for the scenarios `cv1_inner`, `cv1`, `cv3` and
   `all`.
4. Analysis: `make_best_single.py`, `compare_models.py`, `cv2_selection_index.py`, `gwas_emmax.py`,
   `calibrate_readouts.py`, `locus_map.py --readout emmax --window_kb 0 --top 0.01`,
   `race_differential_check.py`, `germplasm.py` (`scripts/04_analysis/`).
5. Figures: `scripts/05_figures/make_figures.py`.

## Licence

MIT (see `LICENSE`).
