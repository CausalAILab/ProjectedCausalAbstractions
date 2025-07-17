import os
import glob
import shutil
import hashlib
import json

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.ds.causal_graph import CausalGraph
from src.datagen import SCMDataset, get_transform
from src.datagen import SCMDataTypes as sdt
import src.metric.visualization as vis
from src.metric.model_utility import load_model
from .base_runner import BaseRunner


class SamplerRunner(BaseRunner):
    """
    Runner for training the sampler post-ncm training.
    """
    def __init__(self, pipeline, dat_model, ncm_model):
        self.pipeline = pipeline
        self.pipeline_name = pipeline.__name__
        self.dat_model = dat_model
        self.dat_model_name = dat_model.__name__
        self.ncm_model = None
        self.ncm_model_name = None

    def create_trainer(self, directory, model_name, max_epochs, patience, gpu=None):
        """
        Creates a PyTorch Lightning trainer.
        """
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/{model_name}/',
                                                  monitor="train_loss",
                                                  save_on_train_epoch_end=True)

        accelerator = "cpu"
        devices = "auto"
        if gpu is not None:
            accelerator = "gpu"
            devices = gpu

        return pl.Trainer(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(monitor='train_loss',
                                           patience=patience,
                                           min_delta=self.pipeline.min_delta,
                                           check_on_train_epoch_end=True)
            ],
            max_epochs=max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(save_dir=f'{directory}/logs/{model_name}/'),
            log_every_n_steps=1,
            devices=devices,
            accelerator=accelerator
        ), checkpoint

    def get_key(self, n, trial_index):
        """
        Creates an identifier for the specific trial.
        """
        return ('gen=%s-pipeline=%s-n_samples=%s-trial_index=%s'
                % (self.dat_model_name, self.pipeline_name, n, trial_index))

    def run(self, exp_name, n, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        """
        Runs the pipeline. Returns the resulting model.
        """
        key = self.get_key(n, trial_index)
        d = 'out/%s/%s' % (exp_name, key)  # name of the output directory

        if hyperparams is None:
            hyperparams = dict()

        with self.lock(f'{d}/lock', lockinfo) as acquired_lock:
            # Attempts to grab the lock for a particular trial. Only attempts the trial if the lock is obtained.
            if not acquired_lock:
                print('[locked]', d)
                return

            try:
                # Return if best.th is generated (i.e. training is already complete)
                if os.path.isfile(f'{d}/best.th'):
                    print('[done]', d)
                    return

                # Do not replace everything if representational model exists
                if not os.path.isfile(f'{d}/best_rep.th'):
                    # Since training is not complete, delete all directory files except for the lock
                    print('[running]', d)
                    for file in glob.glob(f'{d}/*'):
                        if os.path.basename(file) != 'lock':
                            if os.path.isdir(file):
                                shutil.rmtree(file)
                            else:
                                try:
                                    os.remove(file)
                                except FileNotFoundError:
                                    pass

                # Set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                T.manual_seed(seed)
                np.random.seed(seed)
                if verbose:
                    print('Key:', key)
                    print('Seed:', seed)

                if gpu is None:
                    gpu = int(T.cuda.is_available())

                # Load main model
                if verbose:
                    print("Loading NCM model...")
                ncm_dir = hyperparams["ncm-dir"]
                ncm_pipeline, ncm_dir_params, ncm_hyperparams, v_size, v_type = load_model(ncm_dir, verbose=verbose)


                # Create data-generating model and generate data
                if verbose:
                    print("Generating data...")
                sampler_mode = hyperparams["mode"]
                dat_m = self.dat_model(image_size=hyperparams["img-size"], mode="sampling",
                               repr_dim=hyperparams["data-repr-size"],
                                       normalize=hyperparams["normalize"])  # Data generating model
                if hyperparams["data-repr-size"] is not None:
                    dat_m.repr_map.load_state_dict(T.load("{}/repr_map.th".format(ncm_dir)))
                dat_set = SCMDataset(  # Convert data to a Torch Dataset object
                    dat_m, n, augment_transform=None, use_tau=True, soft_sampling=True, soft_sampling_mode=sampler_mode)

                # Create pipeline
                i_size, i_type, o_name, o_size, o_type, var_mapping = dat_m.get_sampler_metadata(mode=sampler_mode)
                u_size = hyperparams["u-size"]
                m = self.pipeline(dat_set, i_size, i_type, u_size, o_name, o_size, o_type, var_mapping,
                                  hyperparams=hyperparams)

                # Initial visualization
                img_sample = dat_set.get_image_batch(64)
                img_sample_fake = m.sample(64, ncm_pipeline.ncm)
                for img_var in img_sample:
                    if verbose:
                        vis.show_image_grid(img_sample[img_var])
                        vis.show_image_grid(img_sample_fake)
                    else:
                        vis.show_image_grid(img_sample[img_var], dir=f'{d}/before_train_real_{img_var}.png')
                        vis.show_image_grid(img_sample_fake, dir=f'{d}/before_train_fake_{img_var}.png')

                # Train model
                trainer, checkpoint = self.create_trainer(d, "sampler", hyperparams['max-epochs'],
                                                          hyperparams['patience'], gpu)
                trainer.fit(m)  # Fit the pipeline on the data
                #ckpt = T.load(checkpoint.best_model_path)  # Find best model
                #m.load_state_dict(ckpt['state_dict'])  # Save best model

                # Save results
                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(m.state_dict(), f'{d}/best.th')

                # Final visualization
                img_lists = m.img_lists
                for v in img_lists:
                    if len(img_lists[v]) > 0 and o_type == sdt.IMAGE:
                        if verbose:
                            print("Image count: {}".format(len(img_lists[v])))
                            vis.show_image_timeline(img_lists[v])
                        else:
                            vis.show_image_timeline(img_lists[v], dir=f'{d}/training_fake_{img_var}.gif')

                img_sample = dat_set.get_image_batch(64)
                img_sample_fake = m.sample(64, ncm_pipeline.ncm)
                for img_var in img_sample:
                    if verbose:
                        vis.show_image_grid(img_sample[img_var])
                        vis.show_image_grid(img_sample_fake)
                    else:
                        vis.show_image_grid(img_sample[img_var], dir=f'{d}/after_train_real_{img_var}.png')
                        vis.show_image_grid(img_sample_fake, dir=f'{d}/after_train_fake_{img_var}.png')
                return m
            except Exception:
                # Move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
