#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=30:00
#SBATCH --nodes=32
#SBATCH --tasks=256
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --output=%j.log
#SBATCH --account=m4293

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=32
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

srun -n 256 /global/homes/i/isjarana/DistEmbed/build/bin/distembed -input /pscratch/sd/i/isjarana/benchmarking/inputs/arabic-2005/arabic-2005.mtx \
	-alpha 0  -beta 1  -dataset arabic-2005 -output /pscratch/sd/i/isjarana/benchmarking/inputs/arabic-2005/fixed_d/d_128_sp_10 \
  -msbfs 1 -iter 1  -tile_width_fraction 1   -input_sparse_file /pscratch/sd/i/isjarana/benchmarking/inputs/arabic-2005/fixed_d/d_128_sp_10/sparse_local.txt\