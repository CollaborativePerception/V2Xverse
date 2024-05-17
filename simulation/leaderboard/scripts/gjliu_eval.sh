#!/bin/bash

CUDA_VISIBLE_DEVICES=5 bash leaderboard/scripts/eval_pnp_short_collab_lgj.sh 10 10 41010 0 0 &
CUDA_VISIBLE_DEVICES=5 bash leaderboard/scripts/eval_pnp_short_single_lgj.sh 10 10 41030 0 1 &

