#!/bin/bash
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -p sched_mit_hill
#SBATCH -p ou_bcs_high
#SBATCH -p ou_bcs_low,ou_bcs_normal
#SBATCH -t 4:50:00
#SBATCH --gres=gpu:1
#SSBATCH --constraint=24GB
#SBATCH --mem=128G
#SBATCH --array=0
# Reset any inherited modules to avoid auto-swaps like nvhpc
module purge
module load gcc/12.2.0
module load openmpi/4.1.4
module load cuda/12.4


export PMIX_MCA_gds=hash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export FLAGS="--bind-to socket -map-by core"
export MPI_FLAGS="-mca btl ^openib -mca pml ob1 -x PSM2_CUDA=1 -x PSM2_MULTIRAIL=0 -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1 -x NCCL_DEBUG -x LD_LIBRARY_PATH" # -x ANACONDA_DIR"

export VAR=`bc -l <<< "${SLURM_ARRAY_TASK_ID}"`
python main.py --model lifstdpreadout \
 --dataset twopattern \
 --batch_size 1 \
 --epochs 10 \
 --stdp_passes 1 \
 --reservoir_size 500 \
 --reservoir_path "/home/tcong13/949Final/models/500_stdp_reservoir.pt"
# python main.py --model lifstdp --dataset twopattern --batch_size 1 --epochs 40 --stdp_passes 3 --reservoir_path "/home/tcong13/949Final/models/500_stdp_reservoir.pt"