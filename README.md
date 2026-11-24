# TraitBoost

Code of TraitBoost, the model of the article "TraitBoost: joint genomic prediction of 33 soybean stress traits and
mapping of their shared loci". Input data are not included; all inputs are public (GRIN-Global descriptors of the USDA
Soybean Germplasm Collection and SoySNP50K genotypes).

## Model

TraitBoost predicts the standardized value of trait *t* for an accession with SNP genotypes **x** as

    F_t(x) = s_t(x) + η_s Σ_{m ≤ S_t} v_{m,t}(x) + η_p Σ_{m ≤ P_t} u_{m,t}(x)

where s_t is the starting value (zero or a GBLUP prediction), v_{m,t} are shared trees and u_{m,t} are trait-specific
trees.
- **Shared trees** are grown for a group of traits (all traits, or the traits of a stress category): a split is chosen
  once for all traits of the group, with the split gains of the traits summed after weighting every trait equally, and
  every trait has its own leaf values. Every trait uses only the accessions scored for it.
- **Trait-specific trees** use the records of one trait.
- **Splits** separate carriers and non-carriers of the alternative allele of one SNP; one tree level for all traits
  and SNPs is a single matrix product on the GPU.
- **Options.** For every trait, three inner folds choose among trait-specific trees only, S_t rounds of shared trees
  followed by trait-specific trees, and trait-specific trees added to cross-validated GBLUP predictions; Optuna tunes
  the hyperparameters.
- **Prediction** of a trait: the mean of four refits, two of the chosen option and two of the best option without
  shared trees.

## Files

| Path | Content |
|---|---|
| `src/soymultires/mtboost.py` | TraitBoost: growth of shared and trait-specific trees |
| `src/soymultires/reml.py` | GBLUP by restricted maximum likelihood (starting values) |
| `src/soymultires/data.py`, `src/soymultires/traits.py` | input data, training and test sets, trait definitions |
| `scripts/03_model/train_mtboost.py` | tuning, choice of the option for every trait, refits and predictions |
| `envs/environment.yml` | conda environment |

## Installation

    mamba env create -f envs/environment.yml
    mamba activate traitboost

A CUDA GPU is used when available.

## Input

Run from the repository root. The model reads
- `data/processed/pheno_long.tsv.gz`: one row per accession, trait and experiment (columns `acc`, `trait`, `batch`,
  `score`, `resistant`; higher scores mean higher resistance);
- `data/processed/geno/dosage.npy` (accessions × SNPs, allele dosages 0/1/2), `data/processed/geno/samples.tsv`
  (column `vcf_id`, in the row order of the dosages) and `data/processed/geno/grm.f32.npy` (genomic relationship
  matrix);
- `splits/folds.tsv` (columns `acc`, `cv1_fold`, `inner_fold`) and, for held-out experiments, `splits/cv3_batches.tsv`
  (columns `unit`, `batch`, `chosen_for`).

## Usage

    python scripts/03_model/train_mtboost.py start   --scenario cv1 --fold 0            # GBLUP starting values (CPU)
    python scripts/03_model/train_mtboost.py fit     --scenario cv1 --fold 0 --tag v2    # tuning, choice and refits
    python scripts/03_model/train_mtboost.py collect --scenario cv1 --tag v2             # predictions of all folds

Scenarios: `cv1` (five-fold cross-validation), `cv3` (held-out experiments) and `all` (model fitted to all accessions).
Predictions are written to `runs/<scenario>/mtboost_<tag>_final/`; `reseed` adds refits with further random seeds and
writes the split gains of the SNPs.

## Licence

MIT (see `LICENSE`).
