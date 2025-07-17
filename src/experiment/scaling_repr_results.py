import os
import argparse
import numpy as np
import torch as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.metric.model_utility import load_model
from src.ds import CTFTerm, CTF
from src.datagen.color_mnist_2 import ColorMNISTBDDataGenerator


def get_image_grid(ax, batch, n_rows, n_cols):
    ax.figure(figsize=(n_rows, n_cols))
    ax.axis("off")
    grid = vutils.make_grid(batch[: n_rows * n_cols], padding=2, normalize=True).cpu()
    ax.imshow(np.transpose(grid, (1, 2, 0)))


def grid_plot(data, inner_rows, inner_cols, fig_name):
    num_rows = len(data)
    num_cols = len(data[0])

    left_space = 0.0
    right_space = 1.0
    wspace = 0.02
    hspace = 0.02

    pv_plots = GridSpec(num_rows, num_cols, left=left_space, right=right_space,
                        top=0.95, bottom=0.1, wspace=wspace, hspace=hspace)

    fig = plt.figure(figsize=(9, 4))
    axes = []
    for row in range(num_rows):
        for col in range(num_cols):
            axes.append(fig.add_subplot(pv_plots[row * num_cols + col]))
            ax = axes[-1]

            batch = data[row][col]

            ax.axis("off")
            ax.set_aspect('equal')
            grid = vutils.make_grid(batch[: inner_rows * inner_cols], padding=2, normalize=True, nrow=inner_cols).cpu()
            ax.imshow(np.transpose(grid, (1, 2, 0)))

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    fig.clf()


def get_exp_name(d):
    exp_name = None
    for name in os.listdir(d):
        if name[0] != '.':
            exp_name = name
    return exp_name


parser = argparse.ArgumentParser(description="Scaling Dimensionality Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
args = parser.parse_args()

d = args.dir
model_types = ["MNIST_Basic_Sampler_Scale", "MNIST_Projected_Sampler_Scale"]
sampler_base_name = "MNIST_Sampler_Base_Scale"
sampler_required = {"MNIST_Basic_Sampler_Scale", "MNIST_Projected_Sampler_Scale"}
repr_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

n_rows = 4
n_cols = 2
n_images = n_rows * n_cols


# Create query
y1 = CTFTerm({'image'}, {}, {'image': 1})
py1 = CTF({y1}, set())
y1_sampler = CTFTerm({'image', 'color', 'digit'}, {}, {'image': 1, 'color': 0, 'digit': 0})
py1_sampler = CTF({y1_sampler}, set())

query = py1
query_sampler = py1_sampler

# Make directory
os.makedirs("{}/figs".format(d), exist_ok=True)

# Load models and generate samples
samples = [[] for _ in range(len(model_types))]
for r_size in repr_sizes:
    T.manual_seed(0)
    np.random.seed(0)
    print("Representation Size: {}".format(r_size))
    sampler_base_dir = "{}/{}_{:03d}".format(d, sampler_base_name, r_size)
    sampler_base_exp_name = get_exp_name(sampler_base_dir)
    sampler_base_full_dir = "{}/{}".format(sampler_base_dir, sampler_base_exp_name)
    sampler_base = load_model(sampler_base_full_dir, repr_dir=sampler_base_full_dir, evaluating=False, verbose=False)[0]

    for i in range(len(model_types)):
        model_dir = "{}/{}_{:03d}".format(d, model_types[i], r_size)
        exp_name = get_exp_name(model_dir)
        model = load_model("{}/{}".format(model_dir, exp_name), repr_dir=sampler_base_full_dir, verbose=False)[0]

        if model_types[i] in sampler_required:
            # Using sampler
            #base_results = sampler_base.sample_ctf(py1_sampler, n=n_images)
            #cond_dict = {model.var_mapping[k]: v for (k, v) in base_results.items() if k in model.var_mapping}
            #print(cond_dict)

            # Using data
            real_data = sampler_base.datagen[:n_images]
            cond_dict = {model.var_mapping[k]: v for (k, v) in real_data.items() if k in model.var_mapping}
            print(cond_dict)

            samples[i].append(model.gen_forward(n=n_images, cond_dict=cond_dict))
        else:
            samples[i].append(model.sample_ctf(query, n=n_images)["image"])

# Create plot
grid_plot(samples, n_rows, n_cols, "{}/figs/scale_grid.png".format(d))
