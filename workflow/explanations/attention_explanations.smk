import numpy as np
from pathlib import Path

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os.path import join
from lifelines import CoxPHFitter
import pickle
from snakemake.utils import Paramspace

from matplotlib import pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.utils.misc import *

from tte.explanations.data_setup import get_loader
from tte.explanations.attention_explanations import (
    occlusion_multibatch,
    occlusion_multibatch_detail,
    occlusion_multibatch_detail_all,
)

SPLITS = load_splits()
ENDPOINTS = load_endpoints()

# ENDPOINTS = pd.read_csv(PATHS.MANUAL_ENDPOINTS, header=None).squeeze().tolist()
# PRETRAIN_NAME = "attention_pretraining_endpointsel_from_omop_avail"
SETTING_NAMES = [
    "attention_fulltable_endpointsel_from_omop_multi",
    "attention_fulltable_endpointsel_from_omop_multi_ontoemb",
]


GEN_PARAMS = dict(
    # n_token_fill_or_sample=8,
    n_token_fill_or_sample=None,
    # n_token_fill_or_sample=128,
    num_workers=0,
    split=0,
    plotby="avg_val",
)
DEBUG_PARAMS = dict(
    num_samples=32,
    bs=8,
    binarize=True,
)
FULL_PARAMS = dict(
    num_samples=None,
    bs=1,
    binarize=True,
)
PARAMS = {
    **GEN_PARAMS,
    **FULL_PARAMS,
    # **DEBUG_PARAMS,
}
params = Box(PARAMS)


def model_select(models):
    keys = sorted(models.keys())
    keys = [
        k
        for k in keys
        if "depth~4" in k and "embed_dim~256" in k and str(GEN_PARAMS["split"]) in k
    ]
    print(keys)
    # print(keys)
    # print(len(keys))
    assert len(keys) == 1
    return keys[0]


rule run_all_occlusion_exp:
    input:
        expand(
            "results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/{endpoint}_unfiltered.csv.gz",
            endpoint=sorted(ENDPOINTS),
            setting_name=SETTING_NAMES,
            tsplit=["test"],
            # tsplit=["train"],
            zero_or_remove=["zero"],
        ),
        expand(
            "results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/{endpoint}_filtered.csv.gz",
            endpoint=sorted(ENDPOINTS),
            setting_name=SETTING_NAMES,
            tsplit=["test"],
            # tsplit=["train"],
            zero_or_remove=["zero"],
        ),


# rule filter_all:
#     input:
#         expand(
#             "results/{setting_name}/occ_exp_{tsplit}/{endpoint}_filtered.csv.gz",
#             tsplit=["train"],
#             endpoint=sorted(ENDPOINTS),
#         ),


rule unfilter_exp:
    input:
        exp="results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/attrs.pkl",
    output:
        "results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/{endpoint}_unfiltered.csv.gz",
    run:
        I, C, Ms, target_names = pickle.load(open(input.exp, "rb"))
        endpoint = wildcards.endpoint
        k = np.where(np.array(target_names) == endpoint)[0][0]
        M = Ms[k]

        # exclusions = pickle.load(open(input.exclusions, "rb"))[wildcards.endpoint]
        # sub_ind = np.isin(I, exclusions, invert=True, assume_unique=True)
        # I = I[sub_ind]
        # M = M[sub_ind]


        sub_cols = np.array(M.sum(0) != 0).flatten()
        C = C[sub_cols]
        M = M[:, sub_cols]

        counts = np.array((M != 0).sum(0)).flatten()
        means = np.array(M.sum(0)).flatten() / counts
        df = pd.DataFrame(
            {
                "count": counts,
                "avg_val": means,
            },
            index=C,
        )
        df.to_csv(output[0])


rule filter_exp:
    input:
        exp="results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/attrs.pkl",
        exclusions="data/endpoints/exclusions/endpoint_specific_exclusions_from_omop.pkl",
    output:
        "results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/{endpoint}_filtered.csv.gz",
    run:
        I, C, Ms, target_names = pickle.load(open(input.exp, "rb"))
        endpoint = wildcards.endpoint
        k = np.where(np.array(target_names) == endpoint)[0][0]
        M = Ms[k]

        exclusions = pickle.load(open(input.exclusions, "rb"))[wildcards.endpoint]
        sub_ind = np.isin(I, exclusions, invert=True, assume_unique=True)
        I = I[sub_ind]
        M = M[sub_ind]


        sub_cols = np.array(M.sum(0) != 0).flatten()
        C = C[sub_cols]
        M = M[:, sub_cols]

        counts = np.array((M != 0).sum(0)).flatten()
        means = np.array(M.sum(0)).flatten() / counts
        df = pd.DataFrame(
            {
                "count": counts,
                "avg_val": means,
            },
            index=C,
        )
        df.to_csv(output[0])


rule run_occlusion_exp_upd_joint:
    input:
        models="export/{setting_name}/{setting_name}.pkl",
    output:
        output="results/{setting_name}/multi{zero_or_remove}_occ_exp_{tsplit}/attrs.pkl",
    params:
        **PARAMS,
        # output_dir=f"results/{SETTING_NAME}/multi{{zero_or_remove}}_occ_exp_{{tsplit}}",
    run:
        exp = pickle.load(open(input.models, "rb"))
        models = exp["models"]
        binarize = params.binarize
        model = model_select(models)
        net = models[model]["net"]
        print(model)
        split = int(model.split("/")[2])
        # TODO
        # sources = ['condition_occurrence']
        sources = model.split("/")[11].split("~")[1].split("?")
        provider = model.split("/")[10].split("~")[1]
        endpoints = model.split("/")[6].split("~")[1].split("?")
        assert endpoints == ["all"]
        endpoints = "all"
        target_names = models[model]["target_names"]
        dl = get_loader(
            split=split,
            sources=sources,
            endpoints=endpoints,
            bs=params.bs,
            n_token_fill_or_sample=params.n_token_fill_or_sample,
            provider=provider,
            num_workers=params.num_workers,
            tsplit=wildcards.tsplit,
            binarize=binarize,
            indiv_exclusions=False,
        )

        # dir = Path(params.output_dir)
        I, C, attrs = occlusion_multibatch_detail_all(
            net,
            dl,
            n_batches=(
                params.num_samples // params.bs
                if isinstance(params.num_samples, int)
                else params.num_samples
            ),
            ndim=len(target_names),
            zero_or_remove=wildcards.zero_or_remove,
        )
        pickle.dump([I, C, attrs, target_names], open(output.output, "wb"))
        # for class_index, target in enumerate(target_names):
        #     fn = dir / f"{target}.csv.gz"
        #     attrs[class_index].to_csv(fn)



rule run_occlusion_exp_upd:
    input:
        models=f"export/{SETTING_NAMES[0]}/{SETTING_NAMES[0]}.pkl",
    output:
        touch(f"results/{SETTING_NAMES[0]}/occ_exp_{{tsplit}}/done.txt"),
        # f"results/{SETTING_NAME}/occ_exp_{{binarize}}.pkl",
    params:
        **PARAMS,
        # template=f"export/{SETTING_NAME}/{target}.pkl",
    run:
        raise NotImplementedError()
        exp = pickle.load(open(input.models, "rb"))
        models = exp["models"]
        binarize = params.binarize
        model = model_select(models)
        net = models[model]["net"]
        print(model)
        split = int(model.split("/")[2])
        sources = model.split("/")[11].split("~")[1].split("?")
        provider = model.split("/")[10].split("~")[1]
        endpoints = model.split("/")[6].split("~")[1].split("?")
        assert endpoints == ["all"]
        endpoints = "all"
        target_names = models[model]["target_names"]
        dl = get_loader(
            split=split,
            sources=sources,
            endpoints=endpoints,
            bs=params.bs,
            n_token_fill_or_sample=params.n_token_fill_or_sample,
            provider=provider,
            num_workers=params.num_workers,
            tsplit=wildcards.tsplit,
            binarize=binarize,
            indiv_exclusions=False,
        )

        dir = Path(f"results/{SETTING_NAMES[0]}/occ_exp_{wildcards.tsplit}")
        for class_index, target in enumerate(target_names):
            attr = occlusion_multibatch_detail(
                net,
                dl,
                target=class_index,
                n_batches=(
                    params.num_samples // params.bs
                    if isinstance(params.num_samples, int)
                    else params.num_samples
                ),
            )
            fn = dir / f"{target}.csv.gz"
            attr.to_csv(fn)

