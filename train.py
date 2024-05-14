import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
import random
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingWarmRestarts

import utils.config as config
import wandb
from utils.my_dataset import RefDataset
from engine.engine import train, validate
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ))


def main_worker(gpu, args):
    args.exp_name = '_'.join([args.exp_name])
    
    args.output_dir = os.path.join(args.output_folder, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.backends.cudnn.enabled = False
    torch.cuda.set_device(args.gpu)
    

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    # dist.init_process_group(backend=args.dist_backend,
    #                         # init_method=args.dist_url,
    #                         init_method=f'tcp://localhost:{random.randint(6000, 7000)}',
    #                         world_size=args.world_size,
    #                         rank=args.rank,)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()
    # wandb
    if args.rank == 0:
        os.environ["WANDB_API_KEY"] = "94280b6b804c2e90f0d865715d73580707493ccc"
        wandb.init(
                    project="etris",
                    job_type="training",
                   config=args,
                   name=args.exp_name,
                   tags=[args.dataset, args.vision_encoder,args.text_encoder])


    # build model
    model, param_list = build_segmenter(args)
    tokenizer = model.tokenizer
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # logger.info(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=False)
                                                # find_unused_parameters=True)


    # build optimizer & lr scheduler
    # optimizer = torch.optim.AdamW(param_list, lr=args.base_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_list, lr=args.base_lr)
    # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay) #original
    scaler = amp.GradScaler()

    # build dataset
    args.batch_size = int(args.batch_size / args.ngpus_per_node)
    args.batch_size_val = int(args.batch_size_val / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                            mask_dir=args.mask_root,
                            dataset=args.dataset,
                            split=args.train_split,
                            mode='train',
                            input_size=args.input_size,
                            word_length=args.word_len)

    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                          mask_dir=args.mask_root,
                          dataset=args.dataset,
                          split=args.val_split,
                          mode='val',
                          input_size=args.input_size,
                          word_length=args.word_len)

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data, shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)
    max_steps = len(train_loader)*args.epochs
    if isinstance(args.warmup_steps, float):
        warmup_steps = int(max_steps * args.warmup_steps)
    else:
        warmup_steps =args.warmup_steps
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )

    best_loss = 100000.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args,tokenizer)

        # evaluation
        val_loss = validate(val_loader, model, epoch_log, args,tokenizer)

        # save model
        if dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'val_loss': val_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if val_loss <= best_loss:
                best_loss = val_loss
                bestname = os.path.join(args.output_dir, "best_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        # scheduler.step()
        torch.cuda.empty_cache()

    time.sleep(2)
    if dist.get_rank() == 0:
        wandb.finish()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)