import os
import argparse

parser = argparse.ArgumentParser(description='create dataset path for v2xverse dataset')
parser.add_argument('--dataset_dir', type=str, required=True, help='directory for dataset')
args = parser.parse_args()

dataset_dir = args.dataset_dir
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

for i in range(14):
    if not os.path.exists(os.path.join(dataset_dir,("weather-%d" % i))):
        os.mkdir(os.path.join(dataset_dir,("weather-%d" % i)))
    if not os.path.exists(os.path.join(dataset_dir,("weather-%d/results" % i))):
        os.mkdir(os.path.join(dataset_dir,("weather-%d/results" % i)))
