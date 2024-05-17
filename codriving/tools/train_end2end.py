# -*- coding: utf-8 -*-
# Author: Genjia Liu <lgj1zed@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

from typing import Union
import argparse
import os
import logging

import traceback
import datetime

import torch
from torch import nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from common.random import set_random_seeds
from common.registry import build_object_within_registry_from_config
from common.io import load_config_from_yaml
from common.torch_helper import load_checkpoint
from common.detection import warp_image

from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model
from codriving.utils.torch_helper import \
    move_dict_data_to_device, build_dataloader
from codriving.utils import initialize_root_logger

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

logger = logging.getLogger("train")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        type=str,
        metavar="FILE",
        help="Config file for training",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="#workers for dataloader (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="Random seed",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log interval",
    )
    parser.add_argument(
        "--out-dir",
        default="./output/",
        type=str,
        help="directory to output model/log/etc.",
        )
    parser.add_argument(
        "--log-filename",
        default="log",
        type=str,
        help="log filename",
        )
    parser.add_argument(
        "--resume",
        default='',
        type=str,
        help="Path of the checkpoint from which the training resumes",
        )

    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", type=str, default=False,
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parse_args = parser.parse_args()

    return parse_args


def main():
    import random
    import numpy as np
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.random.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

    args = parse_args()

    if args.resume=='None':
        args.resume = ''

    initialize_root_logger(path=f'{args.out_dir}/{args.log_filename}.txt')

    perception_hypes = yaml_utils.load_yaml(None, args)


    set_random_seeds(args.seed, LOCAL_RANK)
    config = load_config_from_yaml(args.config_file)

    DISTRIBUTED = \
        ("WORLD_SIZE" in os.environ) and \
        (int(os.environ["WORLD_SIZE"]) > 1)

    # Documentation: https://pytorch.org/docs/stable/distributed.html
    # TODO (yinda): use proto to define the data schema of the training context.
    if DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)
        timeout = datetime.timedelta(seconds=1800)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
        WOLRD_SIZE = torch.distributed.get_world_size()
        RANK = torch.distributed.get_rank()
        logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (WOLRD_SIZE, RANK)
        )
        DEVICE = f"cuda:{LOCAL_RANK}"
    else:
        logger.info("Training with a single process on 1 GPUs.")
        WOLRD_SIZE = 1
        RANK = 0
        # DEVICE = 0
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Rank {RANK} using device: {DEVICE}')

    ############################# for perception ###################################
    data_config = config['data']
    train_data_config = data_config['training']
    train_data_config['dataset']['perception_hypes'] = perception_hypes
    train_dataloader = build_dataloader(train_data_config, DISTRIBUTED)
    val_data_config = data_config['validation']
    val_data_config['dataset']['perception_hypes'] = perception_hypes
    val_dataloader = build_dataloader(val_data_config, DISTRIBUTED)
    
    if 'fusion_args' in perception_hypes['model']['args']:
        if 'communication' in perception_hypes['model']['args']['fusion_args']:
            perception_hypes['model']['args']['fusion_args']['communication']['random_thre'] = True
            logging.info('Training with random communication threshold')

    perception_model = train_utils.create_model(perception_hypes)
    if args.model_dir:
        saved_path = args.model_dir
        perce_init_epoch, perception_model = train_utils.load_saved_model(saved_path, perception_model)
        perception_model.to(DEVICE)
    ######################################################################################################

    # build NN model
    model_config = config['model']
    model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model_decoration_config = config['model_decoration']
    decorate_model(model, **model_decoration_config)
    model.to(DEVICE)

    # loss function
    # TODO (yinda): make sure whether loss function should be wrapped within DDP
    loss_config = config['loss']
    loss_func = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        loss_config,
    )

    train_config = config['training']
    EPOCHS = train_config['epochs']

    # build optimizer
    optimizer_config = train_config['optimizer']
    optimizer = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        optimizer_config,
        params=model.parameters(), # list(model.parameters())+list(perception_model.parameters())
        )
    optimizer_perception = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        optimizer_config,
        params=perception_model.parameters(), # model.parameters(),
        )
    # TODO (yinda): add `lr_scheduler`
    # scheduler
    # Documentation: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    lr_scheduler_config = train_config['lr_scheduler']
    lr_scheduler = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        lr_scheduler_config,
        optimizer=optimizer,
        )

    # resume
    if len(args.resume) > 0:
        # last_epoch_idx = load_checkpoint(args.resume, DEVICE, model, optimizer, lr_scheduler, strict=False)
        last_epoch_idx = load_checkpoint(args.resume, DEVICE, model)
        # last_epoch_idx = -1
    else:
        last_epoch_idx = -1
    start_epoch_idx = last_epoch_idx + 1

    print('start_epoch_idx:',start_epoch_idx)

    # model preparation for distributed training
    if DISTRIBUTED:
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[DEVICE],
            output_device=DEVICE,
            )
        perception_model = nn.parallel.DistributedDataParallel(
            perception_model,
            device_ids=[DEVICE],
            output_device=DEVICE,
            )

    best_mean_loss = 100000
    best_epoch = -1

    # main training loop
    for epoch_idx in range(start_epoch_idx, EPOCHS):
        # trainining
        train_one_epoch(
            args,
            epoch_idx,
            EPOCHS,
            model,
            perception_model,
            loss_func,
            train_dataloader,
            optimizer,
            optimizer_perception,
            lr_scheduler=lr_scheduler,
        )
        try:
            lr_scheduler.step()
        except Exception as e:
            print('scheduler error: ', e)        

        # serialization
        save_checkpoint(
            args,
            epoch_idx,
            model,
            optimizer,
            lr_scheduler,
        )

        # validation
        try:
            best_mean_loss, best_epoch = validate_one_epoch(
                args,
                epoch_idx,
                EPOCHS,
                model,
                perception_model,
                loss_func,
                val_dataloader,
                best_mean_loss,
                best_epoch,
            )
        except Exception as e:
            traceback.print_exc()
            # print('validation error: ', e)




def train_one_epoch(
    args : argparse.Namespace,
    epoch_idx : int,
    max_epoch : int,
    pred_model : nn.Module,
    perce_model : nn.Module,
    loss_func : nn.Module,
    dataloader : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    optimizer_perception : torch.optim.Optimizer,
    lr_scheduler : LRScheduler=None,
):
    """
    Train an epoch
    TODO (yinda): use proto to define the data schema of the training context.
        to reduce the number of context

    Args:
        args (argparse.Namespace): parsed arguments
        epoch_idx (int): epoch index
        max_epoch (int): maximum epoch index
        model (nn.Module): model to be trained
        loss_func (nn.Module): loss function used for supervision
        dataloader (torch.utils.data.DataLoader): training dataloader
        optimizer (torch.optim.Optimizer): optimizer for updating the parameters
        lr_scheduler (LRScheduler): learning rate scheduler
    """
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    DEVICE = f"cuda:{LOCAL_RANK}"

    pred_model.train()
    perce_model.eval()

    last_batch_idx = len(dataloader) - 1
    totl_iter_idx = epoch_idx * len(dataloader)
    max_iters_in_current_epoch = len(dataloader)

    for batch_idx, batch_data in enumerate(dataloader):
        try:
            pred_batch_data, perce_batch_data_dict = batch_data
            move_dict_data_to_device(pred_batch_data, DEVICE)

            pred_batch_data.update({'fused_feature':[],
                                    'features_before_fusion':[],})

            ############ before prediction ##########
            # perception model inference
            with torch.no_grad():
                frame_list = list(perce_batch_data_dict.keys())
                frame_list.sort()
                perception_results_list = []

                for frame in frame_list:
                    perce_batch_data_dict[frame] = train_utils.to_device(perce_batch_data_dict[frame], DEVICE)

                    perception_results = perce_model(perce_batch_data_dict[frame]['ego'])

                    fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2)
                    fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                    pred_batch_data['fused_feature'].append(fused_feature_3[:,:,:192,:])

                    perception_results_list.append(perception_results)
                    # for b in range(len(perception_results['features_before_fusion'])):
                    #     for feature_id in range(3):
                    #         feature_before_fusion = perception_results['features_before_fusion'][b][feature_id]
                
                # warp feature in time sequence
                pred_batch_data['feature_warpped_list'] = []

                for b in range(len(perception_results_list[0]['fused_feature'])):
                    feature_dim = perception_results_list[0]['fused_feature'].shape[1] # 128,256
                    feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(DEVICE).float()
                    det_map_pose = torch.zeros(1, 5, 3).to(DEVICE).float()

                    for t in range(5):
                        feature_to_warp[0, t, :] = pred_batch_data['fused_feature'][t][b] # occ_map_list[t]
                        det_map_pose[:, t] = torch.tensor(pred_batch_data['detmap_pose'][b,t]) # N, 3
                    
                    feature_warped = warp_image(det_map_pose, feature_to_warp)
                    pred_batch_data['feature_warpped_list'].append(feature_warped)
            ##########################################

            model_output = pred_model(pred_batch_data)
            loss, extra_info = loss_func(pred_batch_data, model_output)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # optimizer_perception.zero_grad()
            # optimizer_perception.step()

            # torch.cuda.synchronize()
            # torch.cuda.empty_cache()

            last_batch_reached = (batch_idx == (last_batch_idx - 1))

            if last_batch_reached or batch_idx % int(args.log_interval) == 0:
                if LOCAL_RANK == 0:
                    logging.info((
                        f'Epoch: {epoch_idx}/{max_epoch}, '
                        f'iter.: {batch_idx}/{max_iters_in_current_epoch}, '
                        f'loss: {loss.detach().cpu().numpy()}'
                    ))
        except:
            raise
            logging.info('error training batch, skip!')
            traceback.print_exc()


def validate_one_epoch(
    args : argparse.Namespace,
    epoch_idx : int,
    max_epoch : int,
    pred_model : nn.Module,
    perce_model : nn.Module,
    loss_func : nn.Module,
    dataloader : torch.utils.data.DataLoader,
    best_mean_loss : float,
    best_epoch : int,
):
    """
    Validate one epoch

    Args:
        args (argparse.Namespace): parsed arguments
        epoch_idx (int): epoch index
        max_epoch (int): maximum epoch index
        model (nn.Module): model to be validated
        loss_func (nn.Module): loss used as the validation metrics
            TODO (yinda): use customized & configurable metrics for validation
        dataloader (torch.utils.data.DataLoader): validation dataloader
    """
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    DEVICE = f"cuda:{LOCAL_RANK}"

    pred_model.eval()
    perce_model.eval()

    all_losses = list()
    all_extra_infos = list()

    max_iters_in_current_epoch = len(dataloader)

    if LOCAL_RANK == 0:
        logging.info('Validating...')
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):

            try:
                pred_batch_data, perce_batch_data_dict = batch_data
                move_dict_data_to_device(pred_batch_data, DEVICE)

                pred_batch_data.update({'fused_feature':[],
                                        'features_before_fusion':[],})

                frame_list = list(perce_batch_data_dict.keys())
                frame_list.sort()
                perception_results_list = []

                for frame in frame_list:
                    perce_batch_data_dict[frame] = train_utils.to_device(perce_batch_data_dict[frame], DEVICE)

                    perception_results = perce_model(perce_batch_data_dict[frame]['ego'])

                    fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2)
                    fused_feature_3 = torch.flip(fused_feature_2, dims=[2])
                    pred_batch_data['fused_feature'].append(fused_feature_3[:,:,:192,:])

                    perception_results_list.append(perception_results)
                    # for b in range(len(perception_results['features_before_fusion'])):
                    #     for feature_id in range(3):
                    #         feature_before_fusion = perception_results['features_before_fusion'][b][feature_id]
                
                # warp feature in time sequence
                pred_batch_data['feature_warpped_list'] = []

                for b in range(len(perception_results_list[0]['fused_feature'])):
                    feature_dim = perception_results_list[0]['fused_feature'].shape[1] # 128,256
                    feature_to_warp = torch.zeros(1, 5, feature_dim, 192, 96).to(DEVICE).float()
                    det_map_pose = torch.zeros(1, 5, 3).to(DEVICE).float()

                    for t in range(5):
                        feature_to_warp[0, t, :] = pred_batch_data['fused_feature'][t][b] # occ_map_list[t]
                        det_map_pose[:, t] = torch.tensor(pred_batch_data['detmap_pose'][b,t]) # N, 3
                    
                    feature_warped = warp_image(det_map_pose, feature_to_warp)
                    pred_batch_data['feature_warpped_list'].append(feature_warped)
                ##########################################


                model_output = pred_model(pred_batch_data)
                loss, extra_info = loss_func(pred_batch_data, model_output)
                all_losses.append(loss.to('cpu'))
                all_extra_infos.append(extra_info)
            except:
                traceback.print_exc()
            
    DISTRIBUTED = \
        ("WORLD_SIZE" in os.environ) and \
        (int(os.environ["WORLD_SIZE"]) > 1)
    if False: # DISTRIBUTED:
        # gather object from all machines
        objects_to_be_gathered = (all_losses, all_extra_infos)
        output_list = [None for _ in range(int(os.environ["WORLD_SIZE"]))]
        torch.distributed.all_gather_object(
            output_list,
            objects_to_be_gathered,
        )
        if LOCAL_RANK == 0:
            gathered_losses = [obj[0] for obj in output_list]
            gathered_losses = [t for sharded_result in gathered_losses for t in sharded_result]
            gathered_losses = [t.to('cpu') for t in gathered_losses]
            mean_loss = torch.stack(gathered_losses).mean()
            mean_loss.detach().cpu().numpy()
            logging.info(f'Epoch {epoch_idx}/{max_epoch}, validation mean loss: {mean_loss}')

            logging.info(f'mean_loss:{mean_loss.item()}, best_mean:{best_mean_loss}')
            if mean_loss.item() < best_mean_loss:
                best_mean_loss = mean_loss.item()
                best_epoch = epoch_idx
            logging.info(f'best epoch:{best_epoch}')
    else:
        mean_loss = torch.stack(all_losses).mean()
        mean_loss.detach().cpu().numpy()
        logging.info(f'Epoch {epoch_idx}/{max_epoch}, validation mean loss: {mean_loss}')

        logging.info(f'mean_loss:{mean_loss.item()}, best_mean:{best_mean_loss}')
        if mean_loss.item() < best_mean_loss:
            best_mean_loss = mean_loss.item()
            best_epoch = epoch_idx
        logging.info(f'best epoch:{best_epoch}')

    return best_mean_loss, best_epoch

    


def save_checkpoint(
    args : argparse.Namespace,
    epoch_idx : int,
    model : nn.Module,
    optimizer : torch.optim.Optimizer,
    lr_scheduler : LRScheduler,
):
    """
    Save checkpoint
    Reference: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    Args:
        args: parsed arguments
        epoch_idx: epoch index
        model: model to be checkpointed
        optimizer: optimizer state to be checkpointed
        lr_scheduler: learning rate scheduler to be checkpointed
    """
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    if LOCAL_RANK != 0:
        return

    checkpoint = dict(
        epoch=epoch_idx,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        lr_scheduler_state_dict=lr_scheduler.state_dict(),
    )
    save_dir = f'{args.out_dir}/models'
    save_path = f'{save_dir}/epoch_{epoch_idx}.ckpt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    logging.info(f'Save checkpoint to: {save_path}')


if __name__ == "__main__":
    main()