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


def get_exp_name(d):
    exp_name = None
    for name in os.listdir(d):
        if name[0] != '.':
            exp_name = name
    return exp_name


def grid_plot(data, inner_rows, inner_cols, fig_name):
    num_rows = len(data)
    num_cols = len(data[0])

    left_space = 0.0
    right_space = 1.0
    wspace = 0.0

    pv_plots = GridSpec(num_rows, num_cols, left=left_space, right=right_space,
                        top=0.95, bottom=0.1, wspace=wspace)

    fig = plt.figure(figsize=(16, 4))
    axes = []
    for row in range(num_rows):
        for col in range(num_cols):
            axes.append(fig.add_subplot(pv_plots[row * num_cols + col]))
            ax = axes[-1]

            batch = data[row][col]

            ax.axis("off")
            grid = vutils.make_grid(batch[: inner_rows * inner_cols], padding=2, normalize=True, nrow=inner_cols).cpu()
            ax.imshow(np.transpose(grid, (1, 2, 0)))

    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    fig.clf()


parser = argparse.ArgumentParser(description="Colored MNIST Backdoor Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
args = parser.parse_args()

d = args.dir
model_types = ["rncm", "basic", "projected"]
sampler_name = "sampler_base"
sampler_required = {"basic", "projected"}

n_rows = 2
n_cols = 10
n_images = n_rows * n_cols

test_var = "digit"
test_val_1_raw = 0
test_val_2_raw = 5

test_val_1 = np.zeros((1, 10))
test_val_1[0, test_val_1_raw] = 1
test_val_2 = np.zeros((1, 10))
test_val_2[0, test_val_2_raw] = 1

test_val_1 = T.from_numpy(test_val_1).float()
test_val_2 = T.from_numpy(test_val_2)

y1 = CTFTerm({'image'}, {}, {'image': 1})
x1 = CTFTerm({test_var}, {}, {test_var: test_val_1})
x0 = CTFTerm({test_var}, {}, {test_var: test_val_2})
y1dox1 = CTFTerm({'image'}, {test_var: test_val_1}, {'image': 1})
y1dox1_raw = CTFTerm({'image'}, {test_var: test_val_1_raw}, {'image': 1})

py1 = CTF({y1}, set())
py1givenx1 = CTF({y1}, {x1})
py1dox1 = CTF({y1dox1}, set())
py1dox1givenx0 = CTF({y1dox1}, {x0})
py1dox1_raw = CTF({y1dox1_raw}, set())
py1dox1givenx0_raw = CTF({y1dox1_raw}, {x0})

queries = [py1, py1givenx1, py1dox1, py1dox1givenx0]
queries_raw = [py1, py1givenx1, py1dox1_raw, py1dox1givenx0_raw]

y1_sampler = CTFTerm({'image', 'color', 'digit'}, {}, {'image': 1, 'color': 0, 'digit': 0})
y1dox1_sampler = CTFTerm({'image', 'color', 'digit'}, {test_var: test_val_1}, {'image': 1, 'color': 0, 'digit': 0})
py1_sampler = CTF({y1_sampler}, set())
py1givenx1_sampler = CTF({y1_sampler}, {x1})
py1dox1_sampler = CTF({y1dox1_sampler}, set())
py1dox1givenx0_sampler = CTF({y1dox1_sampler}, {x0})

queries_sampler = [py1_sampler, py1givenx1_sampler, py1dox1_sampler, py1dox1givenx0_sampler]


exp_name = get_exp_name("{}/{}".format(d, sampler_name))
d_model = "{}/{}/{}".format(d, sampler_name, exp_name)
sampler_base, _, _, _, _ = load_model(d_model, verbose=False)
sampler_base_results = []
for q in queries_sampler:
    sampler_base_results.append(sampler_base.sample_ctf(q, n=n_images))

os.makedirs("{}/figs".format(d), exist_ok=True)

datagen = ColorMNISTBDDataGenerator(64, "sampling")

samples = []

index = 0
for model in model_types:
    print("RUNNING {}".format(model))
    exp_name = get_exp_name("{}/{}".format(d, model))
    d_model = "{}/{}/{}".format(d, model, exp_name)
    m, _, _, _, _ = load_model(d_model, verbose=False)

    samples.append([])
    for j in range(len(queries)):
        q = queries[j]
        samp_data = sampler_base_results[j]
        if model in sampler_required:
            cond_dict = {m.var_mapping[k]: v for (k, v) in samp_data.items() if k in m.var_mapping}
            samples[index].append(m.gen_forward(n=n_images, cond_dict=cond_dict))
        else:
            samples[index].append(m.sample_ctf(q, n=n_images)["image"])

    index += 1

samples.append([])
for q in queries_raw:
    samples[-1].append(datagen.sample_ctf(q, n=n_images)["image"])

grid_plot(samples, n_rows, n_cols, "{}/figs/digit_results_grid.png".format(d))