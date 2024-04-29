#!/bin/bash
set -x
# bash scripts/srun.sh llm_s --model Internlm_train XComposer2 --data MME MMBench_TEST_EN AI2D_TEST MMMU_TEST SEEDBench_IMG
srun -n1 --ntasks-per-node=1 --partition $1 --gres=gpu:8 --quotatype=reserved --job-name vlmeval --cpus-per-task=64 torchrun --nproc_per_node=8 run.py ${@:2}