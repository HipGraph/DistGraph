#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --nodes=2
#SBATCH --tasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH --output=%j.log

#export OMP_PROC_BIND=true
#export OMP_PLACES=threads
export OMP_NUM_THREADS=16
#export MKL_NUM_THREADS=1
#export MKL_DYNAMIC="FALSE"

srun -n 16 --cpu-bind=cores ./bin/distembed -input  minst.mtx  -alpha 0.5  -beta 0.5  -dataset minst -batch 4096  -iter 1200
