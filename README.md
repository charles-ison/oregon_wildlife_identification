# Oregon Wildlife Identification

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

## HPC
To load Slurm:

```module load slurm```

How to check which partitions have available GPUs:

```nodestat <partition_name>```

How to start virtual environment and load required modules:

```source env/bin/activate ```

```module load python/3.10 cuda/11.7```

Example sbash command (parameters can be tuned in the run_on_hpc.sh file):

```sbash run_on_hpc.sh```

Example srun bash command (parameters can be tuned per job):

```srun -p <partition_name> -A eecs --gres=gpu:2 --mem=100G --pty bash```

Please store data to be used for training/testing in a local directory:

```saved_data/```

Model weights and confusion matrices will be saved in: 

```saved_models/```

and when using sbatch run logs will be recorded in:

 ```run_logs/```
