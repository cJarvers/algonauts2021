# Singularity For Algonauts

This folder contains all files related to building and executing a Singularity container that fits our needs of running Algonauts investigations on our cluster.

`Singularity.1.7.1` is a description file, which contains all steps to build a container.

## Building Singularity Containers

As a prerequisit you need to install Singularity.

To create a new container then simply run (assuming current directory is top-level directoy of the algonauts repo)

```bash
sudo singularity build torchenvuc2.sif singularity/Singularity.1.7.1
mv torchenvuc2.sif singularity/torchenvuc2.sif
```

This may take a little while dependent on your system's performance.

## Usage

To run a container standalone execute e.g.  (assuming current directory is top-level directoy of the algonauts repo)

```bash
cd singularity
singularity exec --nv --bind $(pwd):/mnt torchenvuc2.sif python3 /mnt/<some-python-script.py>
```

Alternatively, if inteded to be used on the cluster, just start a slurm job which will make use of the container under the hood  (assuming current directory is top-level directoy of the algonauts repo):

```bash
export E_MAIL="<your-e-mail-address>"
export CODE_DIRECTORY="<path-to-your-version-of-the-algonauts-repo>"
export ALGONAUTS_WS="<path-to-common-algonauts-workspace>"
./cluster/run_algonaut_jobs.sh
```
