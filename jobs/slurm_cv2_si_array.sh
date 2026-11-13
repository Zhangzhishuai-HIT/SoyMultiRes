#!/bin/bash
#SBATCH --partition=q04,q05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/home/user/zhangzhishuai/myhermes/2080/SoyMultiRes/logs/slurm/%x_%A_%a.log
# CV2 selection-index correction for one model per array task (line N of <list>).
set -euo pipefail
cd /home/user/zhangzhishuai/myhermes/2080/SoyMultiRes
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
M=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "cv2 selection index: $M on $(hostname) $(date)"
/home/user/zhangzhishuai/.local/share/mamba/envs/soymultires/bin/python -I scripts/04_analysis/cv2_selection_index.py "$M"
echo "end $(date)"
