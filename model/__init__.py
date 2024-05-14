from .segmenter import ETRIS
from .segmenter_my import My
from loguru import logger


def build_segmenter(args):
    model = My(args)
    backbone = []
    head = []
    fix = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            backbone.append(v)
        else:
            fix.append(v)

    logger.info('Backbone with decay={}, fix={}'.format(len(backbone), len(fix)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr # lr_multi 1
    }]
    
    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    return model, param_list
