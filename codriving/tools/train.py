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
from codriving.utils.torch_helper import move_dict_data_to_device


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
        default=50,
        help="Log interval",
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
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))

    args = parse_args()
    set_random_seeds(args.seed, LOCAL_RANK)
    config = load_config_from_yaml(args.config_file)

    DISTRIBUTED = \
        ("WORLD_SIZE" in os.environ) and \
        (int(os.environ["WORLD_SIZE"]) > 1)

    # Documentation: https://pytorch.org/docs/stable/distributed.html
    # TODO (yinda): use proto to define the data schema of the training context.
    if DISTRIBUTED:
        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
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
        DEVICE = 0

    print(f'Rank {RANK} using device: {DEVICE}')

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
        data_sampler = distributed.DistributedSampler(dataset, shuffle=True)
    else:
        data_sampler = None
    # dataloader
    dataloader_config = data_config['dataloader']
    dataloader = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        dataloader_config,
        dataset=dataset,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # build NN model
    model_config = config['model']
    model = build_object_within_registry_from_config(
        CODRIVING_REGISTRY,
        model_config,
    )
    model.to(DEVICE)
    model_decoration_config = config['model_decoration']
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
            EPOCHS,
            model,
            loss_func,
            dataloader,
            optimizer,
            lr_scheduler=lr_scheduler,
        )
        # TODO (yinda): add a validation stage
        lr_scheduler.step()


def train_one_epoch(
    args : argparse.Namespace,
    epoch_idx : int,
    max_epoch : int,
    model : nn.Module,
    loss_func : nn.Module,
    dataloader : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    lr_scheduler=None,
):
    """
    Train an epoch
    TODO (yinda): use proto to define the data schema of the training context.
    """
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    DEVICE = f"cuda:{LOCAL_RANK}"

    model.train()

    last_batch_idx = len(dataloader) - 1
    totl_iter_idx = epoch_idx * len(dataloader)
    max_iters_in_current_epoch = len(dataloader)

    for batch_idx, batch_data in enumerate(dataloader):
        move_dict_data_to_device(batch_data, DEVICE)
        model_output = model(batch_data)
        loss, extra_info = loss_func(batch_data, model_output)
        # TODO (yinda): add methods for extra_info collection or analyzing

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO (yinda): add monitoring function
        torch.cuda.synchronize()

        last_batch_reached = (batch_idx == (last_batch_idx - 1))

        if last_batch_reached or batch_idx % int(args.log_interval) == 0:
            if LOCAL_RANK == 0:
                # TODO (yinda): add monitoring and logging function here
                # TODO (yinda): change to a more formal logging
                print((
                    f'Epoch: {epoch_idx}/{max_epoch}, '
                    f'iter.: {batch_idx}/{max_iters_in_current_epoch}, '
                    f'loss: {loss.detach().cpu().numpy()}'
                ))


if __name__ == "__main__":
    main()
