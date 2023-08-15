import argparse
import os
import logging

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import distributed

from common.random import set_random_seeds
from common.registry import build_object_within_registry_from_config
from common.io import load_config_from_yaml
import codriving
from codriving import CODRIVING_REGISTRY
from codriving.models.model_decoration import decorate_model

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
        "--local-rank",
        default=0,
        type=int,
        help="parse local rank from `torch.distributed` module",
        )
    parser.add_argument(
        "--out-dir",
        default="./output/",
        type=str,
        help="directory to output model/log/etc.",
        )

    parse_args = parser.parse_args()

    return parse_args


def main():
    args = parse_args()
    set_random_seeds(args.seed, args.local_rank)
    config = load_config_from_yaml(args.config_file)


    DISTRIBUTED = \
        ("WORLD_SIZE" in os.environ) and \
        (int(os.environ["WORLD_SIZE"]) > 1)

    # Documentation: https://pytorch.org/docs/stable/distributed.html
    if DISTRIBUTED:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        WOLRD_SIZE = torch.distributed.get_world_size()
        RANK = torch.distributed.get_rank()
        logger.info(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (WOLRD_SIZE, RANK)
        )
        DEVICE = args.local_rank
    else:
        logger.info("Training with a single process on 1 GPUs.")
        WOLRD_SIZE = 1
        RANK = 0
        DEVICE = 0

    # NOTE: assume that dataset is constructed within dataloader
    # dataset
    data_config = config['data']
    dataset_config = data_config['dataset']
    dataset = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataset_config,
    )
    # sampler
    if DISTRIBUTED:
        data_sampler = distributed.DistributedSampler(dataset)
    else:
        data_sampler = None
    # dataloader
    dataloader_config = data_config['dataloader']
    dataloader = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataloader_config,
        dataset=dataset,
        sampler=data_sampler,
    )

    # build NN model
    model_config = config['model']
    model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model_decoration_config = config['model_decoration']
    if model_decoration_config.get('clip_grad', None) is not None:
        decorate_model(model, **model_decoration_config)

    if DISTRIBUTED:
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[DEVICE],
            output_device=DEVICE,
            )

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
        params=model.parameters(),
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

    # trainining loop
    for epoch_idx in range(EPOCHS):

        train_one_epoch(
            args,
            epoch_idx,
            model,
            loss_func,
            dataloader,
            optimizer,
            lr_scheduler=lr_scheduler,
        )
        # TODO: add validation stage
        lr_scheduler.step()


def train_one_epoch(
    args : argparse.Namespace,
    epoch_idx : int,
    model : nn.Module,
    loss_func : nn.Module,
    dataloader : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    lr_scheduler=None,
):
    DEVICE = args.local_rank

    model.train()

    last_batch_idx = len(dataloader) - 1
    totl_iter_idx = epoch_idx * len(dataloader)
    max_iters_in_current_epoch = len(dataloader)

    for batch_idx, batch_data in enumerate(dataloader):
        model_output = model(batch_data)
        optimizer.zero_grad()
        loss, extra_info = loss_func(batch_data, model_output)
        # TODO (yinda): add methods for extra_info collection or analyzing

        loss.backward()
        optimizer.step()

        # TODO (yinda): add monitoring function
        torch.cuda.synchronize()

        last_batch_reached = (batch_idx == (last_batch_idx - 1))

        if last_batch_reached or batch_idx % int(args.log_interval) == 0:
            if DEVICE == 0:
                # TODO (yinda): add monitoring function here
                pass

if __name__ == "__main__":
    main()
