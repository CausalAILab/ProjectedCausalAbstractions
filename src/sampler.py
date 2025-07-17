import os
import argparse

import numpy as np

from src.pipeline import SamplerPipeline
from src.scm.ncm import GAN_NCM
from src.run import SamplerRunner
from src.datagen import ColorMNISTDataGenerator, CelebADataGenerator, ColorMNISTBDDataGenerator
from src.datagen.scm_datagen import SCMDataTypes as sdt

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

valid_pipelines = {
    "gan": SamplerPipeline
}
valid_generators = {
    "mnist": ColorMNISTDataGenerator,
    "celeba": CelebADataGenerator,
    "mnistbd": ColorMNISTBDDataGenerator
}

mode_choices = {'basic', 'projected'}
gan_choices = {"vanilla", "bgan", "wgan", "wgangp"}
gan_arch_choices = {"dcgan", "biggan"}
gan_disc_choices = {"standard", "biggan", "resnet"}

# Basic setup settings
parser = argparse.ArgumentParser(description="Sampler Runner")
parser.add_argument('name', help="name of the experiment")
parser.add_argument('mode', help="type of sampling procedure")
parser.add_argument('gen', help="data generating model")
parser.add_argument('pipeline', help="pipeline to use")
parser.add_argument('dir', help="directory of trained NCM")

# Hyper-parameters for optimization
parser.add_argument('--lr', type=float, default=1e-4, help="generator optimizer learning rate (default: 1e-4)")
parser.add_argument('--alpha', type=float, default=0.99, help="optimizer alpha (default: 0.99)")
parser.add_argument('--bs', type=int, default=128, help="batch size (default: 128)")
parser.add_argument('--grad-acc', type=int, default=1, help="number of accumulated batches per backprop (default: 1)")
parser.add_argument('--max-epochs', type=int, default=1000, help="maximum number of training epochs (default: 1000)")
parser.add_argument('--patience', type=int, default=-1, help="patience for early stopping (default: -1)")

# Hyper-parameters for function NNs
parser.add_argument('--h-layers', type=int, default=2, help="number of hidden layers (default: 2)")
parser.add_argument('--h-size', type=int, default=128, help="neural network hidden layer size (default: 128)")
parser.add_argument('--feature-maps', type=int, default=64, help="CNN feature maps (default: 64)")
parser.add_argument('--u-size', type=int, default=1, help="dimensionality of U variables (default: 1)")
parser.add_argument('--batch-norm', action="store_true", help="set flag to use batch norm")

# Hyper-parameters for GAN
parser.add_argument('--gan-mode', default="vanilla", help="GAN loss function (default: vanilla)")
parser.add_argument('--gan-arch', default="dcgan", help="NN Architecture for GANs (default: dcgan)")
parser.add_argument('--disc-type', default="standard", help="discriminator type (default: standard)")
parser.add_argument('--disc-lr', type=float, default=2e-4, help="discriminator optimizer learning rate (default: 2e-4)")
parser.add_argument('--disc-h-layers', type=int, default=2,
                    help="number of hidden layers in discriminator (default: 2)")
parser.add_argument('--disc-h-size', type=int, default=-1,
                    help="width of hidden layers in discriminator (default: computed from size of inputs)")
parser.add_argument('--disc-resnet-groups', type=int, default=2,
                    help="number of resnet blocks in discriminator (default: 2)")
parser.add_argument('--d-iters', type=int, default=1,
                    help="number of discriminator iterations per generator iteration (default: 1)")
parser.add_argument('--grad-clamp', type=float, default=0.01,
                    help="value for clamping gradients in WGAN (default: 0.01)")
parser.add_argument('--gp-weight', type=float, default=10.0,
                    help="regularization constant for gradient penalty in WGAN-GP (default: 10.0)")
parser.add_argument('--gp-one-side', action="store_true",
                    help="use one-sided version of gradient penalty in WGAN-GP")

# Image settings
parser.add_argument('--img-size', type=int, default=16, help="resize images to this size (use powers of 2)")
parser.add_argument('--data-repr-size', type=int, default=-1, help="fixed size of image representation")

# Experiment parameters
parser.add_argument('--no-normalize', action="store_true", help="turn off dataset normalizing")
parser.add_argument('--n-trials', '-t', type=int, default=1, help="number of trials")
parser.add_argument('--n-samples', '-n', type=int, default=10000, help="number of samples (default: 10000)")
parser.add_argument('--gpu', help="GPU to use")

# Developer settings
parser.add_argument('--verbose', action="store_true", help="print more information")

args = parser.parse_args()

# Basic setup
mode_choice = args.mode.lower()
pipeline_choice = args.pipeline.lower()
gan_choice = args.gan_mode.lower()
gan_arch_choice = args.gan_arch.lower()
gen_choice = args.gen.lower()

assert mode_choice in mode_choices
assert pipeline_choice in valid_pipelines
assert gan_choice in gan_choices
assert gan_arch_choice in gan_arch_choices
assert gen_choice in valid_generators

dat_model = valid_generators[gen_choice]

pipeline = valid_pipelines[pipeline_choice]
ncm_dir = args.dir

gpu_used = 0 if args.gpu is None else [int(args.gpu)]

# Hyperparams to be passed to all downstream objects
hyperparams = {
    'mode': mode_choice,
    'pipeline': pipeline_choice,
    'ncm-dir': ncm_dir,
    'lr': args.lr,
    'alpha': args.alpha,
    'bs': args.bs,
    'grad-acc': args.grad_acc,
    'max-epochs': args.max_epochs,
    'patience': args.patience if args.patience > 0 else args.max_epochs,
    'h-layers': args.h_layers,
    'h-size': args.h_size,
    'feature-maps': args.feature_maps,
    'u-size': args.u_size,
    'batch-norm': args.batch_norm,
    'gan-mode': gan_choice,
    'gan-arch': gan_arch_choice,
    'disc-type': args.disc_type,
    'disc-lr': args.disc_lr,
    'disc-h-layers': args.disc_h_layers,
    'disc-h-size': args.disc_h_size,
    'disc-resnet-groups': args.disc_resnet_groups,
    'd-iters': args.d_iters,
    'grad-clamp': args.grad_clamp,
    'gp-weight': args.gp_weight,
    'gp-one-side': args.gp_one_side,
    'img-size': args.img_size,
    'data-repr-size': args.data_repr_size if args.data_repr_size > 0 else None,
    'normalize': not args.no_normalize,
    'verbose': args.verbose
}

print(hyperparams)

if pipeline_choice == "gan":
    # Adjust data batch size accordingly when training more discriminator iterations
    hyperparams['bs'] = hyperparams['bs'] * hyperparams['d-iters']

if args.n_samples == -1:
    # Run experiment on several sample sizes if not specified
    n_list = 10.0 ** np.linspace(3, 5, 5)
else:
    n_list = [args.n_samples]

# Run for each sample size in n_list
for n in n_list:
    # Avoid using more data than available if possible
    n = int(n)
    hyperparams["bs"] = min(args.bs, n)

    # Run for n_trials amount of trials
    for i in range(args.n_trials):
        while True:
            try:
                # Create a runner for the NCM and pass all hyperparams
                runner = SamplerRunner(pipeline, dat_model, None)
                if not runner.run(args.name, n, i,
                                  hyperparams=hyperparams, gpu=gpu_used, verbose=hyperparams["verbose"]):
                    break
            except Exception as e:
                # Raise any errors
                print(e)
                print('[failed]', i, args.name)
                raise
