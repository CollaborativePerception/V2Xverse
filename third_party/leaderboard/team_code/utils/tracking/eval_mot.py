import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--model_dir", type=str, help='evaluated model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    log_root_dir = '/remote-home/share/yhu/Co_Flow/'
    args = parse_args()
    model_dir = args.model_dir

    print(args)

    result_dict = []
    
    # run eval for MOTA and MOTP
    base_cmd = 'python /remote-home/share/yhu/coperception/tools/track/TrackEval/scripts/run_mot_challenge.py '
    base_cmd += '--BENCHMARK OPV2V --SPLIT_TO_EVAL test --DO_PREPROC False '
    base_cmd += f' --TRACKERS_TO_EVAL {log_root_dir}{model_dir}/track'
    base_cmd += f' --GT_FOLDER {log_root_dir}{model_dir}/track/gt/ '
    base_cmd += f' --TRACKERS_FOLDER {log_root_dir}{model_dir}/track '
    
    cmd = base_cmd + ' --METRICS CLEAR '
    
    os.system(cmd)

    # collect results
    eval_output_path = f'{log_root_dir}{model_dir}/track/pedestrian_summary.txt'
    eval_output_file = open(eval_output_path, 'r')
    # skip header
    eval_output_file.readline()
    perfs = eval_output_file.readline().split(' ')
    
    # MOTA and MOTP
    result_dict.append(float(perfs[0]))
    result_dict.append(float(perfs[1]))


    # run eval for other metrics
    cmd = base_cmd + ' --METRICS HOTA '
    os.system(cmd)
    
    # collect results
    eval_output_path = f'{log_root_dir}{model_dir}/track/pedestrian_summary.txt'
    eval_output_file = open(eval_output_path, 'r')
    # skip header
    eval_output_file.readline()
    perfs = eval_output_file.readline().split(' ')
    
    # HOTA DetA AssA DetRe DetPr AssRe AssPr LocA
    for ii in range(8):
        result_dict.append(float(perfs[ii]))
    
    df = pd.DataFrame([result_dict], columns=['MOTA', 'MOTP', 'HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA'])
    df.to_csv(f'{log_root_dir}{model_dir}/track/logs.csv', sep=',', index=False)

    split_model_dir = '/'.join(model_dir.split('/')[:3])
    comm_thre = model_dir.split('/')[3].split('_')[-2]
    tail = model_dir.split('/')[3].split('_')[0][-1]
    collect_dir = f'{log_root_dir}{split_model_dir}/result_tracking_{tail}.txt'
    with open(collect_dir, 'a') as f:
        f.write(comm_thre + ' ')
        f.write(' '.join([str(x) for x in result_dict]))
        f.write('\n')