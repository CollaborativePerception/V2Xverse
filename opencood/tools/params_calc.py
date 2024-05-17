"""
Todo:

Given a yaml file (from hypes_yaml)

Calculate the following params number:

M1 Single
M2 Single
M3 Single
M4 Single
M1M2M3M4 End-to-end Training
M1M2M3 End-to-end Training
M1M2 End-to-end Training

ConvNeXt Aligner with WarpNet
ConvNeXt Aligner without WarpNet

"""
import torch 
import argparse
from opencood.tools import train_utils
import opencood.hypes_yaml.yaml_utils as yaml_utils
from efficientnet_pytorch import EfficientNet

def calc_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='')
    opt = parser.parse_args()
    return opt


def main():
    opt = calc_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print('creating model')
    model = train_utils.create_model(hypes)
    total_params = sum([param.nelement() for param in model.encoder_m2.camencode.trunk.parameters()])

    print(f'total params number : {total_params/1e6}')

    total_params = sum([param.nelement() for param in model.encoder_m2.camencode.up1.conv.parameters()])

    print(f'total params number : {total_params/1e6}')

    total_params = sum([param.nelement() for param in model.encoder_m2.camencode.up2.parameters()])

    print(f'total params number : {total_params/1e6}')

    total_params = sum([param.nelement() for param in model.encoder_m2.camencode.depth_head.parameters()])

    print(f'total params number : {total_params/1e6}')

    total_params = sum([param.nelement() for param in model.encoder_m2.camencode.image_head.parameters()])

    print(f'total params number : {total_params/1e6}')
    # b = EfficientNet.from_pretrained("efficientnet-b0")
    # total_b = sum([param.nelement() for param in b.parameters()])
    # print(total_b/1e6)

    # total_b = sum([param.nelement() for param in b._blocks.parameters()])
    # print(total_b/1e6)


    

if __name__=='__main__':
    main()
