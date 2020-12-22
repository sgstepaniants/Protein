#!/bin/bash 
#SBATCH -n 10 #Request 10 tasks (cores)
#SBATCH -N 10 #Request 10 nodes
#SBATCH --time=12:00:00
#SBATCH -C centos7 #Request only Centos7 nodes
#SBATCH -p sched_mit_hill #Run on partition
#SBATCH --mem-per-cpu=4000 #Request 4G of memory per CPU
#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=gstepan@mit.edu #email recipient

module add engaging/anaconda/2.3.0
source activate py36
python -u UGW_protein_paramsearch.py