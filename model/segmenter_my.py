import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build
import random
from model.tokenization_bert import BertTokenizer
from model.dino_vit import Block
from utils import utils
import copy
from functools import wraps

import pdb
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


class My(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Vision & Text Encoder

        self.tokenizer  = BertTokenizer.from_pretrained(config.text_encoder)
        embed_dim = config["word_dim"]
        # pdb.set_trace()
        self.attetion_map = config["attention_map"]
        self.visual_encoder = build.vision_encoder(
            config, config.vision_encoder, config.adapter_append
        )
        vision_width = self.visual_encoder.embed_dim

        self.text_encoder = build.text_encoder(
            config, config.text_encoder, config.adapter_append
        )

        text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_width,vision_width)
        self.predictor_depth = config.predictor_depth
        self.predictor = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=config.predictor_heads,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(self.predictor_depth)
            ]
        )
        self.predictor_norm_layer = nn.LayerNorm(vision_width)
        self.predictor_proj = nn.Linear(vision_width, vision_width, bias=True)

        self.use_momentum = config.use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(config.moving_average_decay) #0.99

        if config.freeze_vision_encoder:
            utils.freeze_model(self.visual_encoder)

        if config.freeze_text_encoder:
            utils.freeze_model(self.text_encoder)
        #
        # if config.freeze_proj:
        #     utils.freeze_model(self.vision_proj)
        #     utils.freeze_model(self.text_proj)

        if config.unlock_layernorm:
            if config.unlock_layernorm in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "norm" in name.lower():
                        param.requires_grad = True
            if config.unlock_layernorm in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "LayerNorm" in name:
                        param.requires_grad = True

        if config.unlock_dense:
            for name, param in self.visual_encoder.named_parameters():
                if "mlp" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "dense" in name:
                    param.requires_grad = True

        if config.unlock_attn:
            for name, param in self.visual_encoder.named_parameters():
                if "attn" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "attention" in name:
                    param.requires_grad = True

        if config.unlock_random:
            bert_choices = (
                "query",
                "key",
                "value",
                "attention.output.dense",
                "intermediate.dense",
            )
            for block in self.text_encoder.encoder.layer:
                parameter_to_unlock = random.choice(bert_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

            vit_choices = (
                "proj",
                "fc1",
                "fc2",
            )
            for block in self.visual_encoder.blocks:
                parameter_to_unlock = random.choice(vit_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

        if config.add_adapter:
            last_lm_layer = self.text_encoder.encoder.layer[-1]
            for param in last_lm_layer.parameters():
                param.requires_grad = True

            last_vit_layer = self.visual_encoder.blocks[-1]
            for param in last_vit_layer.parameters():
                param.requires_grad = True

            for param in self.visual_encoder.norm.parameters():
                param.requires_grad = True

        if config.conventional_adapter.insert:
            if config.conventional_adapter.insert in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

            if config.conventional_adapter.insert in ("language_only", True):
                for name, param in self.text_encoder.encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

        if config.bitfit:
            if config.bitfit in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            if config.bitfit in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True

        if config.always_freeze:
            for idx_always_locked in config.always_freeze.visual_encoder:
                for block_idx, block in enumerate(self.visual_encoder.blocks):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

            for idx_always_locked in config.always_freeze.text_encoder:
                for block_idx, block in enumerate(self.text_encoder.encoder.layer):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.parameters())
        print(
            "percentage_trainable={}".format(
                round(trainable_params / total_params * 100, 2)
            )
        )
        print("num trainable={}".format(trainable_params))
        print("total params={}".format(total_params))

    def forward(self, image, text, cropped_img):

        image_embeds = self.visual_encoder(image)
        # if self.attetion_map:
        #     attn = self.visual_encoder.get_last_selfattention(image)
        #     return attn
        # pdb.set_trace()
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.visual_encoder
            target_projections = target_encoder(cropped_img)[:, 0, :].detach()

        text_output = self.text_encoder(**text,mode="text")
        text_embeds = text_output.last_hidden_state
        text_proj = self.text_proj(text_embeds)

        concat_embeds = torch.cat([image_embeds, text_proj],dim=1)

        attention_mask_for_vl = self.get_extended_attention_mask(text.attention_mask, image_patches=image_embeds.shape[1],device=image_embeds.device)

        attn=None
        for idx,blk in enumerate(self.predictor):
            if self.attetion_map and idx== self.predictor_depth-1:
                attn = blk(concat_embeds, return_attention=True, attention_mask=attention_mask_for_vl)
            concat_embeds = blk(concat_embeds, attention_mask=attention_mask_for_vl)

        concat_embeds = self.predictor_norm_layer(concat_embeds) #F.layer_norm(h, (h.size(-1),))

        crop_prediction = concat_embeds[:, 0, :]
        crop_prediction = self.predictor_proj(crop_prediction)

        if self.attetion_map:
            return crop_prediction, target_projections,attn

        return crop_prediction,target_projections

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                    model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.visual_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.visual_encoder)


    def get_extended_attention_mask(self, attention_mask, image_patches,  device):
        batch=attention_mask.shape[0]
        image_attention_mask = torch.ones((batch,image_patches))
        image_attention_mask = image_attention_mask.to(device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=-1)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print(extended_attention_mask.size())
        # print(extended_attention_mask)

        return extended_attention_mask


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
