import os
from copy import copy

import yaml
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from torchvision import utils
import matplotlib.pyplot as plt
from tqdm import trange

from main import config_parser


FRAMESTACK_NUMBER = 5

torch.autograd.set_grad_enabled(True)


def make_layer_grid(model, out_size, layer_idx, channel=-1):
    n_rows, n_cols = get_grid_shape(out_size)

    plt.figure(figsize=(32, 18))
    for i in trange(out_size):
        img_arr = gradient_ascent(model, i, loop_range=100)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(
            img_arr[..., channel],
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="none",
        )
        plt.title(f"Filter #{i}")
        plt.axis("off")
    plt.suptitle(f"Layer {layer_idx}", fontsize=20)
    h_pad = 128 / (n_rows * n_cols)
    plt.tight_layout(h_pad=h_pad, rect=[0, 0, 1, 0.97])

    global model_name
    if channel >= 0:
        idx = copy(channel)
    else:
        idx = FRAMESTACK_NUMBER + channel
    layers_dir = f"layers_activations/{model_name}_channel{idx}"
    os.makedirs(layers_dir, exist_ok=True)
    path = f"{layers_dir}/layer_{layer_idx}.png"
    plt.savefig(path)


def get_grid_shape(n_imgs):
    # system of equations:
    # width * height = n_imgs
    # 9*width = 16*height
    ratio = 16 / 9
    height = np.sqrt(n_imgs / ratio)
    width = n_imgs / height

    n_rows = np.ceil(height).astype(int)
    n_cols = np.ceil(width).astype(int)

    if (n_rows - 1) * n_cols >= n_imgs:
        n_rows -= 1
    if n_rows * (n_cols - 1) >= n_imgs:
        n_cols -= 1

    return n_rows, n_cols


def get_output_size(layer, num_input_features):
    padding = layer.padding[0]
    kernel_size = layer.kernel_size[0]
    stride = layer.stride[0]
    output_size = (
        np.floor(
            (num_input_features + 2 * padding - kernel_size) / stride
        ).astype(int)
        + 1
    )

    return output_size


def plot_weights(layer):
    weights = layer.weight.cpu()
    grid = utils.make_grid(weights, nrow=4)
    plt.imshow(grid.permute((1, 2, 0)))
    plt.show()


def gradient_ascent(model, filter_index, loop_range=40, step=1):
    input_data = torch.randn(
        size=(1, FRAMESTACK_NUMBER, 84, 84),
        requires_grad=True,
        device="cuda",
    )

    for _ in range(loop_range):
        loss_value, grads_value = iterate(model, filter_index, input_data)
        input_data = input_data + grads_value * step

    img_tensor = deprocess_image(input_data[0])
    return img_tensor.transpose(0, 1).transpose(1, 2).cpu().numpy()


def iterate(model, filter_index, input_data):
    output_data = model(input_data)
    loss = output_data[:, filter_index, :, :].mean()
    grads = torch.autograd.grad(loss, input_data)[0][0]
    grads /= torch.sqrt(torch.mean(torch.square(grads))) + 1e-5

    return loss, grads


def deprocess_image(x):
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1
    x += 0.5
    x *= 255
    x = torch.clamp(x, min=0, max=255).type(torch.uint8)
    return x


if __name__ == "__main__":
    with open("models_configs/sevs.yml") as fp:
        config = yaml.safe_load(fp)
    config = config_parser(config)

    model_name = config["model_name"]

    Model: PPO = config["model"]
    model = Model.load(f"models/{model_name}")

    cnn = model.policy.features_extractor.cnn

    for layer_idx, layer in enumerate(cnn, start=1):
        if isinstance(layer, nn.ReLU):
            model = cnn[:layer_idx]
            conv2d_idx = layer_idx - 2
            # make_layer_grid(
            #     model=model,
            #     out_size=model[conv2d_idx].out_channels,
            #     layer_idx=conv2d_idx,
            #     channel=-1,
            # )
            for channel in range(FRAMESTACK_NUMBER):
                make_layer_grid(
                    model=model,
                    out_size=model[conv2d_idx].out_channels,
                    layer_idx=conv2d_idx,
                    channel=channel,
                )
