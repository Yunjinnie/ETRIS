import os
import time
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
                        trainMetricGPU)
import torch.nn as nn
from utils.visualize_attn import draw_mean,rollout
import pdb

def train(train_loader, model, optimizer, scheduler, scaler, epoch, args,tokenizer):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    progress = ProgressMeter(
        len(train_loader),
        [data_time, lr, loss_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (image, text, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        # text = text.cuda(non_blocking=True)
        text_input = tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True,
            max_length=args.word_len,
        )
        if "token_type_ids" in text_input:
            del  text_input["token_type_ids"]
        device = image.device
        text_input = text_input.to(device)

        target = target.cuda(non_blocking=True)

        def loss_fn(z, h):
            loss = F.smooth_l1_loss(z, h)
            return loss
        # forward
        with amp.autocast():
            image_embeds,cropped_embeds = model(image, text_input, target)

            loss = loss_fn(image_embeds,cropped_embeds)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        model.module.update_moving_average()

        # metric
        dist.all_reduce(loss.detach())
        loss = loss / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        scheduler.step()
        lr.update( scheduler.get_last_lr()[-1])
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            if dist.get_rank() in [-1, 0]:
                wandb.log(
                    {
                        #"time/batch": batch_time.val,
                        "time/data": data_time.val,
                        "training/lr": lr.val,
                        "training/loss": loss_meter.val,
                    },
                    step=epoch * len(train_loader) + (i + 1))


@torch.no_grad()
def validate(val_loader, model, epoch, args,tokenizer):
    model.eval()
    time.sleep(2)
    for imgs, texts, target  in val_loader:
        # data
        image = imgs.cuda(non_blocking=True)
        # text = text.cuda(non_blocking=True)
        text_input = tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True,
            max_length=args.word_len,
        )
        if "token_type_ids" in text_input:
            del  text_input["token_type_ids"]
        device = image.device
        text_input = text_input.to(device)

        target = target.cuda(non_blocking=True)

        def loss_fn(z, h):
            loss = F.smooth_l1_loss(z, h)
            return loss
        # forward
        with amp.autocast():
            image_embeds,cropped_embeds = model(image, text_input, target)
            loss = loss_fn(image_embeds,cropped_embeds)
    logger.info(loss)
    wandb.log(
        {
            "val/loss": loss,
        })
    return loss


@torch.no_grad()
def inference(test_loader, model, args,tokenizer):
    model.eval()
    time.sleep(2)
    idx=0
    for imgs, texts, target,params in test_loader:
        # data
        image = imgs.cuda(non_blocking=True)
        # text = text.cuda(non_blocking=True)
        text_input = tokenizer(
            texts, return_tensors="pt", padding="max_length",
            truncation=True,
            max_length=args.word_len,
        )
        if "token_type_ids" in text_input:
            del  text_input["token_type_ids"]
        device = image.device
        text_input = text_input.to(device)

        target = target.cuda(non_blocking=True)

        def loss_fn(z, h):
            loss = F.smooth_l1_loss(z, h)
            return loss
        # forward
        with amp.autocast():
            assert args.attention_map == True
            # attentions = model(image, text_input, target)
            image_embeds,cropped_embeds,attentions = model(image, text_input, target)
            loss = loss_fn(image_embeds,cropped_embeds)
            print(texts)

        mask = rollout(attentions)
        def save_image_with_mask(img, mask, file_path):
            pdb.set_trace()
            # np_img =  cv2.cvtColor(np.transpose(img.squeeze().detach().cpu().numpy(), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            np_img=cv2.resize(np.array(img[0]),(480,480))
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask_image = show_mask_on_image(np_img, mask)
            cv2.imwrite(file_path, mask_image)

        def show_mask_on_image(img, mask):
            # img = np.float32(img)
            # import cv2
            # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("./visualize_output/temp.png", np.uint8(255 * img))
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            # heatmap = np.float32(heatmap) / 255
            cam = heatmap + img #+ (img*0.2+0.5)
            # import pdb;pdb.set_trace()
            cam = cam / np.max(cam)
            return np.uint8(cam) #np.uint8(255 * cam)

        save_image_with_mask(params['ori_img'], mask, f"./visualize_output/{texts}_depth4.png")
        idx+=1
        if idx==10:
            break
        # draw_all_heads('new_all_heads', image_list, attentions)
        # draw_mean(f'{texts}', imgs, attentions)

    logger.info(loss)
    wandb.log(
        {
            "val/loss": loss,
        })
    return loss


