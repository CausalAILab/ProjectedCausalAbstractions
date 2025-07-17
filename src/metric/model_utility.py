import json
import os

import numpy as np
import torch as T

from src.pipeline import GANPipeline, SamplerPipeline, GANReprPipeline
from src.scm.ncm import GAN_NCM
from src.scm.repr_nn.representation_nn import RepresentationalNN
from src.ds import CausalGraph
from src.datagen import ColorMNISTDataGenerator, CelebADataGenerator, ColorMNISTBDDataGenerator,\
    SCMDataset, get_transform
from src.pipeline.repr_pipeline import RepresentationalPipeline
from src.datagen import SCMDataTypes as sdt


valid_pipelines = {
    "GANPipeline": GANPipeline,
    "SamplerPipeline": SamplerPipeline,
    "GANReprPipeline": GANReprPipeline
}
valid_generators = {
    "ColorMNISTDataGenerator": ColorMNISTDataGenerator,
    "ColorMNISTBDDataGenerator": ColorMNISTBDDataGenerator,
    "CelebADataGenerator": CelebADataGenerator
}
architectures = {
    "GAN_NCM": GAN_NCM
}


def str_to_val(s):
    """
    Converts string s to its corresponding Python data type.
    """

    def is_int(x):
        if len(x) < 1:
            return False
        if x[0] == '+' or x[0] == '-':
            return x[1:].isdecimal()
        return x.isdecimal()

    def is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    if is_int(s):
        return int(s)
    elif is_float(s):
        return float(s)
    elif s == 'False':
        return False
    elif s == 'True':
        return True
    elif s == 'None':
        return None

    return s


def parse_directory(d):
    """
    Returns dictionary of parameters in the directory name.
    """
    base = os.path.basename(d)
    if len(base) == 0:
        base = os.path.basename(d[:-1])
    dir_params = {}
    for param in base.split('-'):
        p_split = param.split('=')
        dir_params[p_split[0]] = str_to_val(p_split[1])
    return dir_params


def load_hyperparams(d):
    """
    Loads hyperparameters file into dictionary.
    """
    with open(d, 'r') as f:
        hp = json.load(f)
        for key in hp:
            hp[key] = str_to_val(hp[key])
        return hp


def load_model(d, repr_dir=None, evaluating=True, verbose=False):
    hyperparams = load_hyperparams("{}/hyperparams.json".format(d))
    dir_params = parse_directory(d)

    gen = valid_generators[dir_params["gen"]]
    pipeline = valid_pipelines[dir_params["pipeline"]]
    ncm_model = None
    if dir_params["pipeline"] != "SamplerPipeline":
        ncm_model = architectures[dir_params["model"]]

    # Reconstruct data generating model
    dat_m = gen(hyperparams['img-size'], mode=hyperparams['mode'], repr_dim=hyperparams.get("data-repr-size", None),
                normalize=hyperparams["normalize"], evaluating=evaluating)  # Data generating model

    if repr_dir is not None:
        dat_m.repr_map.load_state_dict(T.load("{}/repr_map.th".format(repr_dir)))

    if dir_params["pipeline"] != "SamplerPipeline":
        use_tau = hyperparams["use-tau"]
        use_projected_cdag = hyperparams["use-projected-cdag"]
        if use_tau:
            if use_projected_cdag:
                cg_name = dat_m.cg_projected
            else:
                cg_name = dat_m.cg_high_level
            v_size = dat_m.v_size_high_level
            v_type = dat_m.v_type_high_level
        else:
            cg_name = dat_m.cg
            v_size = dat_m.v_size
            v_type = dat_m.v_type
        cg = CausalGraph.read("dat/cg/{}.cg".format(cg_name))  # Causal diagram object

        dat_set = SCMDataset(  # Convert data to a Torch Dataset object
            dat_m, n=dir_params["n_samples"],
            augment_transform=get_transform(hyperparams["transform"], hyperparams["img-size"]),
            use_tau=use_tau)

        # Reconstruct representational model
        rep_m = None
        rep_v_size = {k: v for (k, v) in v_size.items()}
        rep_v_type = {k: v for (k, v) in v_type.items()}
        if hyperparams['repr'] != "none":
            if dir_params["pipeline"] != "GANReprPipeline":
                rep_m = RepresentationalPipeline(dat_set, cg, v_size, v_type, hyperparams=hyperparams)
                rep_m.load_state_dict(T.load('{}/best_rep.th'.format(d)))
                if verbose:
                    print("Printing representation model...")
                    for v in rep_m.model.encode_v:
                        print("FUNCTION {}".format(v))
                        print(rep_m.model.encoders[v])
                        print(rep_m.model.decoders[v])
                rep_m = rep_m.model

            if hyperparams['rep-image-only']:
                for v in v_type:
                    if v_type[v] == sdt.IMAGE:
                        rep_v_size[v] = hyperparams['rep-size']
                        rep_v_type[v] = hyperparams['rep-type']
            else:
                rep_v_size = {v: hyperparams['rep-size'] for v in v_type}
                rep_v_type = {v: hyperparams['rep-type'] for v in v_type}
    else:
        dat_set = SCMDataset(dat_m, n=dir_params["n_samples"], augment_transform=None, use_tau=True,
                             soft_sampling=True, soft_sampling_mode=hyperparams["mode"])
        v_size = dat_m.v_size_high_level
        v_type = dat_m.v_type_high_level

    # Reconstruct NCM
    if dir_params["pipeline"] == "GANReprPipeline":
        rep_m = RepresentationalNN(cg, v_size, v_type, hyperparams=hyperparams)
        m = pipeline(dat_set, cg, v_size, v_type, rep_v_size, rep_v_type, repr_model=rep_m, hyperparams=hyperparams,
                     ncm_model=ncm_model)
    elif dir_params["pipeline"] == "SamplerPipeline":
        i_size, i_type, o_name, o_size, o_type, var_mapping = dat_m.get_sampler_metadata(mode=hyperparams["mode"])
        u_size = hyperparams["u-size"]
        m = pipeline(dat_set, i_size, i_type, u_size, o_name, o_size, o_type, var_mapping, hyperparams=hyperparams)
    else:
        m = pipeline(dat_set, cg, rep_v_size, rep_v_type, repr_model=rep_m, hyperparams=hyperparams,
                     ncm_model=ncm_model)
    if verbose:
        print("Printing NCM...")
        for v in m.ncm.v_size:
            print("FUNCTION {}".format(v))
            print(m.ncm.f[v])

        if dir_params["model"] == "GAN_NCM":
            print("Discriminator:")
            print(m.disc)

    m.load_state_dict(T.load("{}/best.th".format(d)))
    return m, dir_params, hyperparams, v_size, v_type