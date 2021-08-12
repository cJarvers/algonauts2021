#!/bin/bash
####################################
## simple Slurm submitter script to setup   ##
## a chain of jobs using Slurm              ##
####################################
## ver.  : 2018-11-27, KIT, SCC

## Define maximum number of jobs via positional parameter 1, default is 5
max_nojob=${1:-5}

## Define type of dependency via positional parameter 2, default is 'after'
dep_type="${2:-after}"
## -> List of all dependencies:
## https://slurm.schedmd.com/sbatch.html

##############################
## our processing-specific code
#!/bin/bash
DEFAULT_E_MAIL=""

SINGULARITY_CONTAINER=${ALGONAUTS_WS}/singularity/"torchenvuc2.sif"
JOB_SCRIPT=${CODE_DIRECTORY}/cluster/"algonauts_job_multigpu_chained.sh"
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


RNG_SEED=1
SCRIPT_BASE_NAME="${JOB_SCRIPT}"
SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME%.sh}
SCRIPT_BASE_NAME=${SCRIPT_BASE_NAME##*/}
JOB_NAME=${SCRIPT_BASE_NAME}_${DEVELOPER}_seed_${RNG_SEED}
##############################
creation_date=$(date +%y%m%d_%H%M%S)
CHCKPT_PATH=${JOBS_DIR}/${creation_date}_${JOB_NAME}_chckpts

mkdir ${CHCKPT_PATH}

myloop_counter=1
## Submit loop
while [ ${myloop_counter} -le ${max_nojob} ] ; do
   ##
   ## Differ msub_opt depending on chain link number
   if [ ${myloop_counter} -eq 1 ] ; then
      slurm_opt=""
      RESUME=false
   else
      slurm_opt="-d ${dep_type}:${jobID}"
      RESUME=true
   fi
   ##
   ## Print current iteration number and sbatch command
   echo "Chain job iteration = ${myloop_counter}"
   echo "   sbatch --export=myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job}"
   ## Store job ID for next iteration by storing output of sbatch command with empty lines

   OUTPUT_LOG=${JOBS_DIR}/${creation_date}_${JOB_NAME}_${myloop_counter}.log
   ERROR_LOG=${JOBS_DIR}/${creation_date}_${JOB_NAME}_${myloop_counter}_errors.log

   jobID=$(sbatch --export=ALL,myloop_counter=${myloop_counter},RNG_SEED=${RNG_SEED},CODE_DIRECTORY=${CODE_DIRECTORY},ALGONAUTS_WS=${ALGONAUTS_WS},SINGULARITY_CONTAINER=${SINGULARITY_CONTAINER},JOBS_DIR=${JOBS_DIR},CHCKPT_PATH=${CHCKPT_PATH},RESUME=${RESUME} ${slurm_opt} --job-name=${JOB_NAME} --mail-user=${E_MAIL} --output=${OUTPUT_LOG} --error=${ERROR_LOG} ${JOB_SCRIPT} 2>&1 | sed 's/[S,a-z]* //g')
   ##
   ## Check if ERROR occured
   if [[ "${jobID}" =~ "ERROR" ]] ; then
      echo "   -> submission failed!" ; exit 1
   else
      echo "   -> job number = ${jobID}"
   fi
   ##
   ## Increase counter
   let myloop_counter+=1
done
