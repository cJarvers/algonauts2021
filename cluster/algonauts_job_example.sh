#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL              		# Mail events (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --partition=dev_gpu_4			                # Define on what queue to place
#SBATCH --gres=gpu:1						# Query one GPU on the node
#SBATCH --mem=10000mb						# Necessary RAM
#SBATCH --time=00:30:00                         		# Time limit hrs:min:sec
#SBATCH --output=bwunicluster/jobs/algonauts_job_%j.log         # Standard output log
#SBATCH --error=bwunicluster/jobs/algonauts_job_%j_errors.log   # Error log

# Print metadata at the beginning
pwd; hostname; date

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Algonauts workspace: ${ALGONAUTS_WS}"
echo "Algonauts repo: ${CODE_DIRECTORY}"
echo "Singularity container: ${SINGULARITY_CONTAINER}"
echo "Rng seed: ${RNG_SEED}"

# Setup of job directory
mkdir ${ALGONAUTS_WS}/jobs
JOB_DIR=${ALGONAUTS_WS}/jobs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}
mkdir ${JOB_DIR}

echo "Job directory: ${JOB_DIR}"

# Copy the data that was used for the job (to track what has been used during execution)
echo "Copying data from ${CODE_DIRECTORY} to ${JOB_DIR}"
cp ${CODE_DIRECTORY}/scripts/example_script.py ${JOB_DIR}/example_script.py

# Replace whatever job-dependent arguments, could as well be handled via CLI
sed -i -e "s/^rng_seed =.*/rng_seed = ${RNG_SEED}/g" ${JOB_DIR}/example_script.py

# Switch to the job directory an execute the script
cd ${JOB_DIR}
singularity exec --nv --bind $(pwd):/mnt ${SINGULARITY_CONTAINER} python3 /mnt/example_script.py

# Print metadata at the end
date


