import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import pdb
def rollout(attentions, discard_ratio=0.9, head_fusion='max'):
    result = torch.eye(901)
    attentions = attentions.detach().cpu()
    attentions = attentions[:,:,:901,:901]
    with torch.no_grad():
        if head_fusion == "mean":
            attention_heads_fused = attentions.mean(axis=1)
        elif head_fusion == "max":
            attention_heads_fused = attentions.max(axis=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attentions.min(axis=1)[0]
        else:
            raise "Attention head fusion type Not supported"
        # Drop the lowest attentions, but
        # don't drop the class token
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size()[0] * discard_ratio), dim=-1,largest=False)
        indices = indices[indices != 0]
        flat[indices] = 0

        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0 * I) / 2
        a = a / a.sum(dim=-1)

        result = torch.matmul(a, result)
    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask


def draw_mean(name, images, attentions):
    output_dir = './visualize_output2/'

    num_image, num_head = attentions.shape[0], attentions.shape[1]
    attentions = attentions[0, 1:].reshape(64, 64).unsqueeze(dim=0).unsqueeze(dim=0)
    attentions = nn.functional.interpolate(attentions, scale_factor=(16, 16), mode="nearest").detach().to('cpu').numpy()
    attentions = attentions.squeeze(0)


    attentions = attentions[:,:,:901,:901].mean(axis=1)
    fig, ax = plt.subplots(2, 4, figsize=(6, 4))
    for i in range(4): #0 01 1 23 2 01 3 23
        j = 0 if i%2==0 else 2
        ax[i//2, j].imshow(images[0])
        ax[i//2, j].set_axis_off()
        ax[i//2, j+1].imshow(attentions[0], aspect=1)
        ax[i//2, j+1].set_axis_off()
    file_name = os.path.join(output_dir, name+".png")
    plt.subplots_adjust(wspace=.03, hspace=.01)
    plt.savefig(fname=file_name)
    print(f"{file_name} saved.")