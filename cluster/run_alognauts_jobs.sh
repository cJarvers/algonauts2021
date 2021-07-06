#!/bin/bash
DEFAULT_E_MAIL=""

CODE_DIRECTORY= "<path-to-algonauts-repo>"
ALGONAUTS_WS="<path-to-algonauts-workspace>"
SINGULARITY_CONTAINER=${ALGONAUTS_WS}/singularity/"torchenvuc2.sif"
JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_example.sh"
DEVELOPER=$(whoami)

if [[ -z "${E_MAIL}" ]]; then
   E_MAIL=${DEFAULT_E_MAIL}
   echo "E_MAIL environment variable hasn't been set, defaulting to ${E_MAIL}"
fi
echo "Sending job update e-mails to ${E_MAIL}"

for seed in {1..2}
do
   RNG_SEED=$seed
   SCRIPT_BASE_NAME="${JOB_SCRIPT}"
   SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME%.sh}
   SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME##*/}
   JOB_NAME=${SCRIPT_BASE_NAME}_${DEVELOPER}_seed_${RNG_SEED}
   sbatch --export=ALL,RNG_SEED=${RNG_SEED},CODE_DIRECTORY=${CODE_DIRECTORY},ALGONAUTS_WS=${ALGONAUTS_WS},SINGULARITY_CONTAINER=${SINGULARITY_CONTAINER} --job-name=${JOB_NAME} --mail-user=${E_MAIL} ${JOB_SCRIPT}
done

