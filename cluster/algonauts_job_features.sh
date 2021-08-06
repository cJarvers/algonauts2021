#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL              		# Mail events (NONE, BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --partition=single			                # Define on what queue to place
#SBATCH --mem=32000mb						# Necessary RAM
#SBATCH --time=02:00:00                         		# Time limit hrs:min:sec

# Print metadata at the beginning
pwd; hostname; date

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Jobs directory: ${JOBS_DIR}"
echo "Algonauts workspace: ${ALGONAUTS_WS}"
echo "Algonauts repo: ${CODE_DIRECTORY}"
echo "Singularity container: ${SINGULARITY_CONTAINER}"
echo "Rng seed: ${RNG_SEED}"

# Create job directory
JOB_DIR=${JOBS_DIR}/${SLURM_JOB_ID}_${SLURM_JOB_NAME}
mkdir ${JOB_DIR}
echo "Job directory: ${JOB_DIR}"

# Copy the data that was used for the job (to track what has been used during execution)
echo "Copying data from ${CODE_DIRECTORY} to ${JOB_DIR}"
cp -R ${CODE_DIRECTORY}/* ${JOB_DIR}

# Switch to the job directory an execute the script
cd ${JOB_DIR}
singularity exec --nv --bind $(pwd):/mnt --bind ${ALGONAUTS_WS}/checkpoints:/checkpoints --bind ${ALGONAUTS_WS}/data/AlgonautsVideos268_All_30fpsmax:/videos --bind ${ALGONAUTS_WS}/data/participants_data_v2021:/fmri --pwd /mnt ${SINGULARITY_CONTAINER} ./cluster/algonauts_submission.sh

# Print metadata at the end
date


