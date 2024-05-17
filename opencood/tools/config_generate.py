import argparse
import yaml
import opencood.hypes_yaml.yaml_utils as yaml_utils

def get_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                            help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--save_name", "-s", type=str, required=True,
                            help='yaml save path ')
    opt = parser.parse_args()
    return opt
if __name__ == "__main__":
    opt = get_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    with open(opt.save_name, 'w') as outfile:
        yaml.dump(hypes, outfile)