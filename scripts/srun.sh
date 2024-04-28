#!/bin/bash
set -x
# bash scripts/srun.sh llm_s --model XComposer2 --data MME
srun -n1 --ntasks-per-node=1 --partition $1 --gres=gpu:8 --quotatype=reserved --job-name vlmeval --cpus-per-task=64 torchrun --nproc_per_node=8 run.py ${@:2}