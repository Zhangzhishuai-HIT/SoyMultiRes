# SoyMultiRes: TraitBoost, joint genomic prediction of 33 soybean stress traits and mapping of their shared loci

Code for the article "TraitBoost: joint genomic prediction of 33 soybean stress traits and mapping of their shared loci". Input
data and results are not included; all inputs are public (see Data).

## Model

TraitBoost predicts the standardized score of stress trait *t* for an accession with SNP genotypes **x** as

    F_t(x) = s_t(x) + η_s Σ_{m ≤ S_t} v_{m,t}(x) + η_p Σ_{m ≤ P_t} u_{m,t}(x)

where s_t is the starting value, v_{m,t} are shared trees and u_{m,t} are trait-specific trees.
- **Shared trees** are grown for a group of traits (all 33 traits, or the traits of a stress category): a split is
  chosen for all traits of the group, with the second-order split gains of the traits summed after weighting every
  trait equally, and every trait has its own leaf values.
- **Trait-specific trees** use the records of one trait.
- **Splits.** Accessions are inbred lines, so a split separates carriers and non-carriers of the alternative allele
  of one SNP. One tree level for all traits and SNPs is a single matrix product, which runs on a GPU.
- **Options for every trait.** Three inner folds choose among trait-specific trees from zero, trait-specific trees
  after S_t rounds of shared trees, and trait-specific trees that start from cross-validated GBLUP predictions (no
  shared trees), and choose the number of rounds P_t. Fifteen Optuna trials draw the other settings, and every trait
  uses the trial with its highest inner accuracy.
- **Prediction** of a trait: the mean of four refits, namely the chosen option and the best option without shared
  trees, each with two random seeds. The number of rounds of trait-specific trees may be zero (shared trees alone,
  or the GBLUP predictions alone).

Implementation: `src/soymultires/mtboost.py` (tree growth) and `scripts/03_model/train_mtboost.py` (tuning, choice,
refits, split gains).

## Names used in the code

The code keeps the working name of the model (`mtboost`).

| In the code | In the article |
|---|---|
| `mtboost_v2_final` | TraitBoost |
| `mtboost_v2_noshare4` | TraitBoost without shared trees (mean of four refits) |
| `mtboost_v2_share4` | the chosen option alone (mean of four refits) |
| `gblup`, `mtboost_v2_gblup` | GBLUP (held-out experiments: the GBLUP predictions computed for TraitBoost) |
| `bayesb`, `lightgbm_avg4`, `bglr_mt` | BayesB, LightGBM (mean of four refits), multi-trait GBLUP |
| `best_single_avg4` | best single-trait model (GBLUP, BayesB or LightGBM; new accessions: chosen within each training set; held-out experiments: chosen without the records of the held-out experiment; `make_best_single.py --nested`) |
| `cv1`, `cv1_inner` | new accessions (five-fold cross-validation); inner cross-validation |
| `cv2`, `cv3`, `all` | measured traits; held-out experiments; models fitted to all accessions |
| `batch` | experiment (a GRIN evaluation study) |
| `trade-off` (locus class) | opposite directions between stresses |
| `NEMATCYST_<r>`, `PHYTOROT_<r>`, ... | GRIN descriptor codes (SCN race r, *Phytophthora sojae* race r, ...; `src/soymultires/traits.py`) |

`scripts/03_model/train_mt.py`, `train_hgmt.py`, `mtgbm_probe.py` and `diag_prox.py`, with
`src/soymultires/mtmodel.py` and `hgmt.py`, belong to multi-trait models that were tried during development and are
not reported in the article. The same applies to the BGLR trial script in `scripts/02_baselines/`; the MegaLMM timing
trial (`megalmm_trial_prep.py`, `megalmm_trial.R`, MegaLMM 0.9.5 installed from source, `envs/ENV_CHANGES.tsv`) is
reported in Supplementary Note 1 as the reason the factor model was not used.

## Layout

| Path | Content |
|---|---|
| `src/soymultires/` | Python package: trait definitions, data access and training/test sets, accuracy measures, REML (from HapGAM), SNP blocks, LightGBM tuning, TraitBoost |
| `scripts/01_data/` | phenotypes from the GRIN descriptors, genotype and relationship matrices, near-identical groups, training and test sets, associations between traits, known genes and loci |
| `scripts/02_baselines/` | GBLUP, BayesB (BGLR), multi-trait GBLUP (BGLR Multitrait), LightGBM |
| `scripts/03_model/` | TraitBoost: `train_mtboost.py start` (GBLUP predictions, CPU), `fit` (GPU), `reseed` (further refits; split gains of the final model), `collect` |
| `scripts/04_analysis/` | best single-trait model, accuracy tables, paired bootstrap over groups of near-identical accessions, selection index, mixed-model association analysis (EMMAX, GPU; with or without fixed effects for taxon and principal components), choice of the locus score with known genes, loci and genes in shared loci, robustness of the loci, known loci, loci reported in earlier studies, pubescence form at the chromosome 12 locus, race-differential loci, multi-resistance index and its out-of-fold check, independent checks of the evaluation |
| `scripts/05_figures/` | figures |
| `jobs/` | SLURM scripts, GPU launcher and task queue, task lists |
| `envs/` | conda environments (`environment_soymultires.yml`: Python 3.11, PyTorch 2.5.1, LightGBM 4.7, Optuna 5.0; `environment_soymultires_r.yml`: R 4.3, BGLR 1.1.4) |
| `docs/DATA_CARD.md` | trait codings, record counts and data versions |

Each script describes its inputs, outputs and usage in its first lines.

## Installation

    mamba env create -f envs/environment_soymultires.yml      # Python
    mamba env create -f envs/environment_soymultires_r.yml    # BGLR (BayesB, multi-trait GBLUP)

TraitBoost and the association analysis use a CUDA GPU when one is available; the other models run on CPU.
`train_mtboost.py` enables TF32 matrix products on the GPU by default (`--no_tf32` switches them off). All TraitBoost
fits and refits of the article (33 traits) were run with the code of this release, the default `--chunk` and TF32 on
NVIDIA RTX 4090 GPUs (`jobs/autodl/`: GBLUP starts `run_starts.sh`, task queue `run_queue.py` with
`tasks_race10.txt`); stored predictions are reproduced exactly only with the same setting, chunk, code and GPU type.

## Data

All inputs are public. Place them, or symbolic links to them, under `data/raw/`:

| `data/raw/...` | Content |
|---|---|
| `grin_descriptors/` | GRIN-Global descriptor tables of the soybean collection (https://npgsweb.ars-grin.gov; one CSV per descriptor; check that the first line names the descriptor of the file: one download of descriptor 51138, *Phytophthora* rot race 10, returned another descriptor and had to be repeated) |
| `soysnp50k_gnm2/` | LIS `Wm82.gnm2.div.Song_Hyten_2015` (SoySNP50K genotypes on Wm82.a2) |
| `wm82_gnm2_annotation/` | LIS `Wm82.gnm2.ann1.RVB6` (gene models and functional annotation) |
| `grin_pubescence/` | GRIN-Global descriptor PUBFORM (pubescence form), for `pubescence_check.py` |
| `soysnp50k_marker_positions/` | LIS SoySNP50K marker positions on Wm82.gnm1, gnm2 and gnm4 (`Wm82.gnm*.mrk.SoySNP50K`) |
| `lis_glycine_max_genomes/` | LIS *Glycine max* genome assemblies and annotations (`https://data.legumeinfo.org/Glycine/max/`; genome_main files), for `probe_check.py` and `probe_rhg1_scan.py` |

The SoySNP50K VCF is parsed with `scripts/01_data/parse_soysnp50k.py` of the HapGAM repository
(https://github.com/Zhangzhishuai-HIT/HapGAM). `scripts/01_data/build_geno.py` and `build_pheno.py` read the
parsed files from `../HapGAM/data/interim/soysnp50k/` (variables `HG` and `SAMPLES` at the top of the scripts). The
genes and loci used for calibration and annotation are listed, with their sources, in `scripts/01_data/known_loci.py`.
The job scripts in `jobs/` and `envs/create_envs.sh` contain paths of the original machine.

## Order of the analysis

1. Data (`scripts/01_data/`): `inventory_grin_stress.py`, `build_geno.py`, `near_identical_groups.py`,
   `build_pheno.py`, `make_splits.py` (*P. sojae* race 10, which was added after the folds had been fixed, is left out of
   everything that defines the folds, so that the folds are those of all fits), `pheno_correlations.py`,
   `pair_associations.py` (and with `splits/folds.tsv` as fourth argument, one accession per group of near-identical
   accessions), `known_loci.py`.
2. Other models (`scripts/02_baselines/`): `run_gblup.py`, `run_gblup_inner.py` (`--traits` for single traits),
   `bglr_prepare.py` + `bglr_fit.R`,
   `bglr_mt_prepare.py` + `bglr_mt_fit.R` (`--scenario cv1` and `cv3`), `run_lgbm.py` (`--reseed 3` for three further
   refits, `--collect`).
3. TraitBoost (`scripts/03_model/train_mtboost.py`): `start`, `fit`, `reseed`, `collect` for the scenarios
   `cv1_inner`, `cv1`, `cv3` and `all` (on the GPU server: `jobs/autodl/run_starts.sh`, then `run_queue.py`);
   `reseed --scenario all --first_seed 0 --seeds 2` writes the split gains of
   the final model, and `reseed --scenario cv1 --fold <f> --first_seed 0 --seeds 2 --dump_full` (default `--chunk`) the
   predictions of all traits for the test accessions of a fold (out-of-fold check of the multi-resistance index; it
   needs the GBLUP predictions of every trait for all test accessions, `start --scenario cv1 --fold <f> --eval_all`,
   which it computes itself when they are not cached). Repeated tuning runs for new accessions (`jobs/autodl/tasks_tuning_repeats.txt`): `fit --scenario cv1
   --fold <f> --tag v2r<k> --trials 15 --seeds 2 --optuna_seed <k> --config '{"seed": <k>}'` (k = 1, 2), then
   `collect --scenario cv1 --tag v2r<k>` (`jobs/r11_repeats_post.sh`).
4. Analysis (`scripts/04_analysis/`): `SINGLE=gblup,bayesb,lightgbm_avg4 SUFFIX=_avg4 gate_step6.py` and
   `make_best_single.py results/gate/cv1_inner_best_single_avg4.tsv best_single_pooled <cv1|cv1_inner|cv3>` (best
   single-trait model chosen once from the pooled inner predictions, for the comparison in Supplementary Note 1), then
   `inner_choice_check.py` (the choices of the inner cross-validation made within each training set and without the
   accessions of the held-out experiments, with the resulting accuracies) and `make_best_single.py --nested
   <cv1|cv1_inner|cv3>` (the best single-trait model of the article as `best_single_avg4`: chosen within each training set,
   and for held-out experiments without the records of the held-out experiment), `make_ensemble.py` (mean of the standardized GBLUP and
   LightGBM predictions; `--pair lightgbm_avg4 mtboost_v2_noshare4 --name ens_lightgbm_noshare --scenarios cv1 cv3` for the
   tree ensemble, the mean of LightGBM and TraitBoost without shared trees), `cv1_summary.py`, `cv3_summary.py`, `bootstrap_compare.py run` and
   `collect` for `cv1`, `cv2` and `cv3` (100,000 resamples of groups of near-identical accessions; `cv2` after
   `cv2_selection_index.py mtboost_v2_final best_single_avg4`), `bootstrap_mean_diff.py cv1` and `cv3` (mean
   difference over traits, joint resampling of all traits; `--models A B ...` compares A with the others, `--out` names
   the result file, e.g. for the repeated tuning runs and the ensemble of LightGBM and TraitBoost without shared trees;
   `--drop_from_sparse NEMATCYST_14` adds the sparse traits without SCN race 14, `jobs/r19_hapcond_meandiff.sh`), `seed_variation.py` (accuracy of single refits with
   different seeds), `gwas_emmax.py` (main analysis; `--add_traits` merges traits added later into the stored results;
   `--covar taxon_pcs --tag _taxonpc`, `--taxon gmax --tag _gmax`
   and `--condition major_genes --tag _cond`, the last with the lead SNPs near Rhg1, Rhg4 and Rps1 as fixed effects,
   for the checks; `--condition_snps ... --tag _condscn` with the five SNPs of Supplementary Note 4 as fixed effects for
   SCN races 1, 3, 5 and 14, `jobs/r11_cond_chr10.sh`), `gwas_mac.py`, `gwas_null_threshold.py` (genome-wide threshold of every trait from 1,000
   traits simulated without effects of single SNPs, whose liability variance ratio is matched to the observed REML ratio; the
   random numbers of a trait follow from its position in the original order of 32 traits, race 10 last; `--add_traits`
   for traits added later; with
   `--watch` the statistics of the simulated traits at the two race-differential lead SNPs, and
   `--only NEMATCYST_14 --watch "" --watch_blocks "2059:ss715628625,1173:ss715606985"` the largest statistic of
   their blocks; `--traitset mendel` for the five traits used to choose the score),
   `calibrate_readouts.py`, `locus_map.py --alpha 0.05 --min_mac 30 --perm 10000` (thresholds: the stricter of the
   Bonferroni and the simulated threshold, `results/gwas/null_threshold.tsv`; and with `--gwas`, `--mac`,
   `--block`, `--zlim`, `--drop`, `--min_snps`, `--tag` for the other analyses; the *G. max*-only map is
   `--gwas _gmax --mac _gmax --drop BPMV --tag _gmax`, the Bonferroni-only map `--null none --tag _bonf`, and the
   map with directions read only for the traits of a block `--read_races 0 --tag _locusreading`),
   `trait_regions.py` (loci of a trait merged into regions), `nbarc_shift.py` (NB-ARC genes of the shared loci
   against shifts of the shared loci), `locus_robustness.py` (summary of
   all analyses, directions at the most significant SNP of every trait, resistant carriers per locus),
   `control_loci.py`,
   `race_differential_check.py --block 2059`, `--block 1173`, `--genomewide` and `--crossval` (out-of-sample check of the genome-wide comparison), `chr10_carriers.py` (origin, GRIN source type and SCN race 3 resistance of the carriers of the chromosome 10 allele; z at its lead SNP with the five SNPs fitted, `results/loci/chr10_conditional.tsv`), `race_differential_lmm.py` (carriers and non-carriers among the race-3-resistant accessions with a mixed model; also with a wider search range and with equal residual and genetic variances), `probe_check.py` (BLASTN search of the 121 bp around the two race-differential lead SNPs and of 500 random SNPs in the Wm82.a2, Wm82.a4 and Peking assemblies; needs BLAST+ and the genome files of the Legume Information System; the alignments in the Peking assembly quoted in Supplementary Note 4 are in its output `work/probe_check/hits_Peking.gnm3.KKCD.tsv`), `probe_rhg1_scan.py` (copies of the sequence around the chromosome 10 lead SNP in the first intron of the α-SNAP gene of the *Rhg1* repeat in all *G. max* assemblies of the Legume Information System, against the SoySNP50K calls of the same cultivars; `jobs/r16_probe_check.sh`, `jobs/r16_probe_rhg1_scan.sh`) and `rhg1_haplotype_check.py` (calls of the chromosome 10 lead SNP, which detects the retrotransposon RAC of *rhg1-a* (Bayless et al. 2019), against the SNP haplotypes around *Rhg1* and the allele near *Rhg4*), `chr10_haplotype_conditional.py` (z of SCN races 1, 3, 5 and 14 at the chromosome 10 lead SNP with the Peking and PI 88788 SNP haplotypes at *Rhg1* as fixed effects), `race_differential_r_only.py` (the separation with only resistant, not moderately resistant, reactions counted as resistant), `reported_loci.py` (needs
   `results/literature/reported_associations.tsv`, the SNPs reported for the traits in earlier association
   studies; the list is part of Online Resource 3 of the article), `pubescence_check.py` (needs the GRIN
   descriptor PUBFORM in `data/raw/grin_pubescence/`), `candidate_sources.py` (GRIN source types of the candidates and of all accessions, for Table S16), `germplasm.py` (after the `--dump_full` refits of the five
   folds; also the out-of-fold index of every scored accession and bootstrap intervals over groups of near-identical
   accessions), `candidate_relatives.py` (identity of the never-scored candidates with scored accessions),
   `verify_pipeline.py`.
5. Figures: `scripts/05_figures/make_figures.py`.

## Licence

MIT (see `LICENSE`).

## Article files

`manuscript/build/build_all.sh` builds the manuscript, Online Resources 1-3 (Online Resource 3 with the association
statistics of all SNPs and traits, `build_esm3.py`) and the cover letter (`build_cover_letter.py`, which checks the
numbers it states against the manuscript).
