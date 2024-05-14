
def calculate_gradient_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def _calculate_gradient_norm_from_parameter_list(parameter_list):
    if not parameter_list:
        return 0
    total_norm = 0
    for p in parameter_list:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm / len(parameter_list)


def calculate_gradient_norms_for_ln(model):
    vit_parameters = [
        param
        for name, param in model.visual_encoder.named_parameters()
        if ("norm" in name.lower() and param.requires_grad)
    ]
    bert_parameters = [
        param
        for name, param in model.text_encoder.named_parameters()
        if ("LayerNorm" in name and param.requires_grad)
    ]

    return (
        _calculate_gradient_norm_from_parameter_list(vit_parameters),
        _calculate_gradient_norm_from_parameter_list(bert_parameters),
    )


def calculate_gradient_norms_for_non_ln(model):
    vit_parameters = [
        param
        for name, param in model.visual_encoder.named_parameters()
        if ("norm" not in name.lower() and param.requires_grad)
    ]
    bert_parameters = [
        param
        for name, param in model.text_encoder.named_parameters()
        if ("LayerNorm" not in name and param.requires_grad)
    ]

    return (
        _calculate_gradient_norm_from_parameter_list(vit_parameters),
        _calculate_gradient_norm_from_parameter_list(bert_parameters),
    )


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_embedding(model):
    for name, param in model.named_parameters():
        if "embed" in name:
            param.requires_grad = False