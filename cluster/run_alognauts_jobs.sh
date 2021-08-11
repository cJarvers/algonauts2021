#!/bin/bash
DEFAULT_E_MAIL=""

SINGULARITY_CONTAINER=${ALGONAUTS_WS}/singularity/"torchenvuc2.sif"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_example.sh"
JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_multigpu.sh"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_loss.sh"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_youtubefaces.sh"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_davis.sh"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_cityscapes.sh"
#JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_flippingtrafo.sh"
DEVELOPER=$(whoami)

# Setup of jobs directory
JOBS_DIR=${ALGONAUTS_WS}/jobs
mkdir ${JOBS_DIR}

if [[ -z "${E_MAIL}" ]]; then
   E_MAIL=${DEFAULT_E_MAIL}
   echo "E_MAIL environment variable hasn't been set, defaulting to ${E_MAIL}"
fi
echo "Sending job update e-mails to ${E_MAIL}"

if [[ -z "${CODE_DIRECTORY}" ]]; then
   echo "CODE_DIRECTORY environment variable hasn't been set, exiting."
   return -1
fi
if [[ -z "${ALGONAUTS_WS}" ]]; then
   echo "ALGONAUTS_WS environment variable hasn't been set, exiting."
   return -1
fi


for seed in {1..1}
do
   RNG_SEED=$seed
   SCRIPT_BASE_NAME="${JOB_SCRIPT}"
   SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME%.sh}
   SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME##*/}
   JOB_NAME=${SCRIPT_BASE_NAME}_${DEVELOPER}_seed_${RNG_SEED}
   OUTPUT_LOG=${JOBS_DIR}/$(date +%y%m%d_%H%M%S)_${JOB_NAME}.log
   ERROR_LOG=${JOBS_DIR}/$(date +%y%m%d_%H%M%S)_${JOB_NAME}_errors.log

   sbatch --export=ALL,RNG_SEED=${RNG_SEED},CODE_DIRECTORY=${CODE_DIRECTORY},ALGONAUTS_WS=${ALGONAUTS_WS},SINGULARITY_CONTAINER=${SINGULARITY_CONTAINER},JOBS_DIR=${JOBS_DIR} --job-name=${JOB_NAME} --mail-user=${E_MAIL} --output=${OUTPUT_LOG} --error=${ERROR_LOG} ${JOB_SCRIPT}
done

