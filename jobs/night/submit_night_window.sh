#!/bin/bash
# Submit tonight's CPU work for SoyMultiRes inside the night window (user rule 2026-11-11, same as the pan project:
# 00:00-07:00 every CPU partition may be used; jobs outside q04/q05 must end before 07:00; by day nothing outside q04/q05
# may wait in the queue). Real submissions run only between <night> 00:00 and 06:55 (started at 00:00 by
# timer_night_window.sh); every job gets -p cu,fat --begin=<night>T00:00:00 --deadline=<night>T06:55:00.
# The BGLR lists are regenerated first, so only unfinished folds are submitted. Refuses to run while any smr_night job
# is queued or running.
# Usage (login node): jobs/night/submit_night_window.sh YYYY-MM-DD [--dry-run]
set -euo pipefail
NIGHT=${1:?night date YYYY-MM-DD}
DRY=${2:-}
date -d "$NIGHT" +%F > /dev/null
now=$(date +%s)
if [ "$DRY" != "--dry-run" ] && { [ "$now" -lt "$(date -d "$NIGHT 00:00" +%s)" ] || [ "$now" -ge "$(date -d "$NIGHT 06:55" +%s)" ]; }; then
  echo "outside the $NIGHT window (00:00-06:55): use --dry-run, or start timer_night_window.sh $NIGHT" >&2
  exit 4
fi
ROOT=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
cd "$ROOT"
busy=$(squeue -h -u "$USER" -o '%j' | grep -E '^smr_(bglr|lgbm|night)' | sort | uniq -c || true)
if [ -n "$busy" ]; then
  printf 'smr night jobs still queued or running:\n%s\n' "$busy" >&2
  [ "$DRY" = "--dry-run" ] || exit 3
fi
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/02_baselines/bglr_prepare.py lists --scenario cv1
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/02_baselines/bglr_mt_prepare.py lists --scenario cv1
# LightGBM: trait x fold pairs without a finished part file (largest traits first)
while read -r t f; do [ -f "runs/cv1/lightgbm/parts/$t.fold$f.tsv.gz" ] || echo "$t $f"; done < jobs/lists/lgbm_cv1.txt > jobs/lists/lgbm_cv1_todo.txt
# LightGBM refits with 3 extra seeds (fair baseline for seed-averaged TraitBoost), both scenarios, only missing ones
for sc in cv1_inner cv1; do while read -r t f; do
  sp=$([ "$sc" = cv1 ] && echo "fold$f" || echo "fold${f}_inner$f")
  [ -f "runs/$sc/lightgbm/parts/$t.$sp.seeds.tsv.gz" ] || echo "$t $f $sc"
done < jobs/lists/lgbm_cv1.txt; done > jobs/lists/lgbm_reseed_todo.txt
WIN=(-p cu,fat --begin="${NIGHT}T00:00:00" --deadline="${NIGHT}T06:55:00")
line="$(date '+%F %T') night=$NIGHT"
while IFS=$'\t' read -r name list res cmd; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  n=$(grep -c . "$list" 2>/dev/null || true)
  if [ "${n:-0}" -eq 0 ]; then line="$line $name=skip(empty)"; continue; fi
  # shellcheck disable=SC2086
  if [ "$DRY" = "--dry-run" ]; then
    echo "DRY: sbatch -J $name ${WIN[*]} --array=1-$n $res $cmd" >&2; id="DRY"
  else
    id=$(sbatch --parsable -J "$name" "${WIN[@]}" --array=1-"$n" $res $cmd)
  fi
  line="$line $name=$id[1-$n]"
done < jobs/night/night_plan.tsv
echo "$line"
[ "$DRY" = "--dry-run" ] || echo "$line" >> logs/night/night_window_submit.log
