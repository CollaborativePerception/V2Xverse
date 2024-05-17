#!/bin/bash
# work_dir=out_dir/Eval-planner-e2e-$(hostname)-$(date +%Y%m%d-%H%M%S)
# mkdir -p ${work_dir}
export MODEL_DIR=$2
work_dir=$MODEL_DIR

# Set environment variables
export PYTHONPATH=${PWD}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=$1  # "2,3" Configured according to gpu status
export OMP_NUM_THREADS=8

# print BASH variables/environment variables
echo "Set work_dir=${work_dir}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

py_script_and_args="
    --config-file ./codriving/hypes_yaml/codriving/end2end_${3:-codriving}.yaml
    --out-dir ${work_dir}
    --planner_resume $4
    --model_dir ${MODEL_DIR}
    --log-filename log_latency.txt
    "

# multi-gpu testing
python ./codriving/tools/inference_e2e_latency.py \
    ${py_script_and_args} \

# /GPFS/data/gjliu/Auto-driving/V2Xverse/out_dir/planner_e2e-DB30-20240304-142908/models/epoch_15.ckpt
# kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
