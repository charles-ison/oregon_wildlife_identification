# Oregon Wildlife Identification

## HPC
To load Slurm:

```module load slurm```

How to check which partitions have available GPUs:

```nodestat <partition_name>```

Example srun bash command (parameters can be tuned per job):

```srun -p <partition_name> -A eecs --gres=gpu:2 --mem=100G --pty bash```

## Enviroment
To avoid version control issues, use the following commands to load the python enviroment for this project:
```bash
python3 -m venv env
source env/bin/activate
module load python/3.10 cuda/11.7
pip3 install -r requirements.txt
```
If you add pakages please run before committing:
```bash
pip3 freeze > requirements.txt
```
