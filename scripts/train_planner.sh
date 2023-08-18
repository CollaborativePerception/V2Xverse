#!/bin/bash

work_dir=out_dir/planner-$(hostname)-$(date +%Y%m%d-%H%M%S)
mkdir -p ${work_dir}
log_file=${work_dir}/log.txt

# Set environment variables
export PYTHONPATH=${PWD}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Configured according to gpu status
export OMP_NUM_THREADS=8

# print BASH variables/environment variables
echo "Set work_dir=${work_dir}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

py_script_and_args="
    codriving/tools/train.py
    --config-file ./codriving/hypes_yaml/codriving/planner.yaml
    --out-dir ${work_dir}
    "

# multi-gpu training
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    ${py_script_and_args} \
    2>&1 | tee ${log_file}

kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
