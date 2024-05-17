# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    tiny_train_dataset = Subset(opencood_train_dataset, range(1300,1400))
    # tiny_train_dataset = opencood_train_dataset
    # opencood_validate_dataset = build_dataset(hypes,
    #                                           visualize=False,
    #                                           train=False)
    # tiny_validate_dataset = Subset(opencood_validate_dataset, range(24))

    train_loader = DataLoader(tiny_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False)
    # val_loader = DataLoader(tiny_validate_dataset,
    #                         batch_size=hypes['train_params']['batch_size'],
    #                         num_workers=4,
    #                         collate_fn=opencood_train_dataset.collate_batch_train,
    #                         shuffle=True,
    #                         pin_memory=False,
    #                         drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        # saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            batch_data['ego']['epoch'] = epoch
            batch_data['ego']['iter'] = i
            ouput_dict = model(batch_data['ego'])

            # first argument is always your output dictionary,
            # second argument is always your label dictionary.
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader))
            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                criterion.logging(epoch, i, len(train_loader), suffix="_single")
            # back-propagation
            final_loss.backward()
            optimizer.step()

        # hypes['train_params']['eval_freq'] = 2
        # if epoch % hypes['train_params']['eval_freq'] == 0:
        #     valid_ave_loss = []

        #     with torch.no_grad():
        #         for i, batch_data in enumerate(val_loader):
        #             model.eval()

        #             batch_data = train_utils.to_device(batch_data, device)
        #             ouput_dict = model(batch_data['ego'])

        #             final_loss = criterion(ouput_dict,
        #                                    batch_data['ego']['label_dict'])
        #             valid_ave_loss.append(final_loss.item())
        #     valid_ave_loss = statistics.mean(valid_ave_loss)
        #     print('At epoch %d, the validation loss is %f' % (epoch,
        #                                                       valid_ave_loss))

        scheduler.step(epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
