# template for all experiments
# need to fill in pretraining; setting_name; spec_params & wildcard constraints; and sweep function, nothing else
import numpy as np
from pathlib import Path
from lifelines.utils import concordance_index
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os.path import join
from lifelines import CoxPHFitter
import pickle
from snakemake.utils import Paramspace

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.survival.lifelines_models import aggregate_lifeline_models
from tte.utils.misc import load_vars, load_paths, load_iids, load_endpoints, load_splits
from tte.survival.downstream_step import downstream_step
from tte.utils.model_selection import (
    pick_best_multi_model,
    aggregate_multi_models,
    pick_best_model,
)
from tte.plotting.survival_plotting import plot_c_index_absolute

PRETRAIN_NAME = (
    "attention_pretraining_endpointsel_from_omop_avail"  # or "baseline_pretraining"
)
SETTING_NAME = "baseline_randemb"

paths = load_paths()
# CENTERS = load_vars().CENTERS[:2]
# SPLITS = list(range(len(load_vars().CENTER_SPLITS)))[:2]
SPLITS = load_splits()
ENDPOINTS = load_endpoints()

SPEC_PARAMS = {
    "endpoints": ["all"],
    "indiv_exclusion": ["False"],
    "avail": [ "condition_occurrence=1" ],
    "depth": [4],
    "embed_dim": [256],
    "sources": [ "condition_occurrence?procedure_occurrence?drug_exposure?extra" ],
    "provider": [ "rand" ],
    "input_dim": [None],
    "loss_type": ["cox"],
}
P = join(*[f"{param}~{{{param}}}" for param in sorted(SPEC_PARAMS.keys())])


### TODO: set constraints on specs
wildcard_constraints:
    sources="|".join([s.replace("?", "\?") for s in SPEC_PARAMS["sources"]]),
    endpoint="|".join(ENDPOINTS + ["all"]),


### TODO: define sweep; set params of interest to {param}
### sweep params need to match those in baseline_pretraining/attention_pretraining
### make sure to sort the columns (last two lines of this function)
def sweep():
    A = pd.DataFrame(
        [
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": "{depth}",
                "embed_dim": "{embed_dim}",
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": False,
                "sources": "{sources}",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": True,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": "{input_dim}",
                "loss_type": "{loss_type}",
                "optimizer": "adamw",
                "low_bit": "False",
                "num_workers": "4",
            },
        ]
    )
    # LRS = [3e-4, 1e-3, 3e-3]
    LRS = [1e-3]
    # LRS = [1e-5, 1e-4, 1e-3]
    WDS = [0.001]
    # BSS = [256, 512, 1024, 2048]
    BSS = [512]
    OTHER_SETTINGS = [
        {
            # "warmup_epochs": 1,
            # "min_epochs": 1,
            # "epochs": 1,
            "warmup_epochs": 2,
            "min_epochs": 8,
            "epochs": 8,
            "early_stopping_patience": -1,
        },
    ]
    O = pd.DataFrame(
        [
            {
                **other,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
            }
            for other in OTHER_SETTINGS
            for lr in LRS
            for wd in WDS
            for bs in BSS
        ]
    )
    sweep_params = A.merge(O, how="cross")

    embed_params = pd.DataFrame(
        [
            {
                "provider": "{provider}",
                "model": "none",
                "input_type": "none",
                "pca_dim": 0,
            },
        ]
    )
    sweep_params = sweep_params[sorted(sweep_params.columns)]
    embed_params = embed_params[sorted(embed_params.columns)]
    return sweep_params, embed_params


sweep_params, embed_params = sweep()
sweep_paramspace = Paramspace(sweep_params)
embed_paramspace = Paramspace(embed_params)

from config.sweeps.demo import sweep_params as downstream_params

downstream_paramspace = Paramspace(downstream_params)


def filter_downstream(fn):
    endpoint = fn.split("/")[-1].split(".")[0]
    dic = dict(x.split("~") for x in fn.split("/")[2:-1])

    return (endpoint in dic["endpoints"]) or (dic["endpoints"] == "all")


rule export_with_downstream:
    input:
        pretrained_models=expand(
            f"models/{{setting_name}}/{{split}}/{P}/best_model.pkl",
            split=SPLITS,
            P=P,
            setting_name=SETTING_NAME,
            **SPEC_PARAMS,
        ),
        downstream=[
            fn
            for fn in expand(
                f"results/{{setting_name}}/{P}/{{endpoint}}.pkl",
                endpoint=ENDPOINTS,
                setting_name=SETTING_NAME,
                **SPEC_PARAMS,
            )
            if filter_downstream(fn)
        ],
        # demo_baseline_downstream=expand(
        #     "results/{setting_name}/avail~{avail}/demo_{endpoint}.pkl",
        #     endpoint=ENDPOINTS,
        #     setting_name=SETTING_NAME,
        #     avail=SPEC_PARAMS["avail"],
        # ),
    output:
        models=f"export/{SETTING_NAME}/{SETTING_NAME}.pkl",
        c_plot=f"export/{SETTING_NAME}/{SETTING_NAME}.png",
    default_target: True
    run:
        recs = []
        recs_CI_lower = []
        recs_CI_upper = []
        print(input.downstream)
        for fn in input.downstream:
            print(fn)

            res = pickle.load(open(fn, "rb"))
            global_c = res['global_c_index']
            global_c_CI = res["global_c_index_CI"]

            endpoint = Path(fn).stem
            spec = dict(tuple(x.split("~")) for x in fn.split("/")[2:-1])
            specname = "_".join(f"{k}={v}" for k, v in spec.items())
            # recs.append({"global_c": global_c, "endpoint": endpoint, "spec": specname})
            recs.append({
                "global_c": global_c,
                "endpoint": endpoint,
                "spec": specname,
                })
            recs_CI_lower.append({
                "global_c_CI_lower": global_c_CI[0],
                "endpoint": endpoint,
                "spec": specname,
                })
            recs_CI_upper.append({
                "global_c_CI_upper": global_c_CI[1],
                "endpoint": endpoint,
                "spec": specname,
                })


        if "demo_baseline_downstream" in input.keys():
            for fn in input.demo_baseline_downstream:
                # global_c = pickle.load(open(fn, "rb"))["global_c_index"]
                res = pickle.load(open(fn, "rb"))
                global_c = res['global_c_index']
                global_c_CI = res["global_c_index_CI"]
                endpoint = Path(fn).stem.split('demo_')[-1]
                specname = "demo_baseline"
                recs.append(
                    {"global_c": global_c, "endpoint": endpoint, "spec": specname}
                )
                recs_CI_lower.append({
                    "global_c_CI_lower": global_c_CI[0],
                    "endpoint": endpoint,
                    "spec": specname,
                    })
                recs_CI_upper.append({
                    "global_c_CI_upper": global_c_CI[1],
                    "endpoint": endpoint,
                    "spec": specname,
                    })
        print(recs)
        recs = pd.DataFrame(recs)
        print(recs)
        specs = np.unique(recs["spec"])
        endpoints = np.unique(recs["endpoint"])
        recs.set_index("endpoint", inplace=True)
        cols_c = {spec: recs[recs["spec"] == spec]["global_c"] for spec in specs}
        recs_c = pd.DataFrame(cols_c)

        recs_CI_lower = pd.DataFrame(recs_CI_lower)
        recs_CI_upper = pd.DataFrame(recs_CI_upper)
        recs_CI_lower.set_index("endpoint", inplace=True)
        recs_CI_upper.set_index("endpoint", inplace=True)
        cols_CI_lower = {
            spec: recs_CI_lower[recs_CI_lower["spec"] == spec]["global_c_CI_lower"]
            for spec in specs
        }
        cols_CI_upper = {
            spec: recs_CI_upper[recs_CI_upper["spec"] == spec]["global_c_CI_upper"]
            for spec in specs
        }
        recs_CI_lower = pd.DataFrame(cols_CI_lower)
        recs_CI_upper = pd.DataFrame(cols_CI_upper)

        models = dict()
        for inp in input.pretrained_models:
            with open(inp, "rb") as f:
                model = pickle.load(f)

            models[inp] = {
                "net": model["net"],
                "c_indices": model["test_cindices"],
                "target_names": model["target_names"],
                # "wildcards": dict(wildcards),
                # "inp": inp,
            }
        # dump = {"c_index": recs_c, "models": models}
        dump = {
            "c_index": recs_c,
            'c_index_CI_lower': recs_CI_lower,
            'c_index_CI_upper': recs_CI_upper,
            "models": models,
            }
        with open(output.models, "wb") as f:
            pickle.dump(dump, f)
        plot_c_index_absolute(recs_c, output.c_plot)


rule aggregate_demo_endpoint:
    input:
        expand("models/demo/{{endpoint}}/{split}/avail~{{avail}}/best_val_model.pkl", split=SPLITS),
    output:
        f"results/{SETTING_NAME}/avail~{{avail}}/demo_{{endpoint}}.pkl",
    run:
        ...
        res = aggregate_lifeline_models(input, verbose=True)
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule aggregate_downstream_endpoint:
    input:
        expand(
            "models/{setting_name}/{P}/{{endpoint}}/{split}/best_val_model.pkl",
            split=SPLITS,
            setting_name=SETTING_NAME,
            P=P,
        ),
    output:
        f"results/{SETTING_NAME}/{P}/{{endpoint}}.pkl",
    run:
        ...
        res = aggregate_lifeline_models(input, verbose=True)
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule gather_downstream_center:
    input:
        expand(
            "models/{setting_name}/{P}/{endpoint}/{split}/{sweep}/model.pkl",
            split="{split}",
            endpoint="{endpoint}",
            sweep=downstream_paramspace.instance_patterns,
            setting_name=SETTING_NAME,
            P=P,
        ),
    output:
        f"models/{SETTING_NAME}/{P}/{{endpoint}}/{{split}}/best_val_model.pkl",
    run:
        best_model = pick_best_model(input)
        with open(output[0], "wb") as f:
            pickle.dump(best_model, f)


rule downstream_center_sweepstep:
    input:
        f"results/{SETTING_NAME}/{P}/results.pkl",
        meta=paths.META_PREPROCESSED,
        train_iids="data/splits/{split}_train.csv",
        val_iids="data/splits/{split}_val.csv",
        test_iids="data/splits/{split}_test.csv",
        exclusions=paths.FULL_EXCLUSIONS_FROM_OMOP,
    output:
        f"models/{SETTING_NAME}/{P}/{{endpoint}}/{{split}}/{downstream_paramspace.wildcard_pattern}/model.pkl",
    params:
        rescale_age=True,
    run:
        ...
        if wildcards.endpoint == "T2D":
            exclusions = [
                input.exclusions,
                "data/endpoints/exclusions/t2d_extra_exclusions.csv",
            ]
        else:
            exclusions = input.exclusions
        res = downstream_step(
            res_fn=input[0],
            split=wildcards.split,
            exclusions=exclusions,
            endpoint=wildcards.endpoint,
            meta_fn=input.meta,
            rescale_age=params.rescale_age,
            penalizer=wildcards.penalizer,
            l1_ratio=wildcards.l1_ratio,
        )
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule aggregate_pretraining_models:
    input:
        expand(
            "models/{setting_name}/{split}/{P}/best_model.pkl",
            split=SPLITS,
            setting_name=SETTING_NAME,
            P=P,
        ),
    output:
        f"results/{SETTING_NAME}/{P}/results.pkl",
    run:
        res = aggregate_multi_models(input, verbose=True)
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule gather_pretraining_models:
    input:
        expand(
            "models/{pretrain_name}/{split}/endpoints~{endpoints}/indiv_exclusion~{indiv_exclusion}/avail~{avail}/{embed}/{sweep}/model.pkl",
            split="{split}",
            endpoints="{endpoints}",
            indiv_exclusion="{indiv_exclusion}",
            avail="{avail}",
            embed=embed_paramspace.instance_patterns,
            sweep=sweep_paramspace.instance_patterns,
            pretrain_name=PRETRAIN_NAME,
        ),
    output:
        f"models/{SETTING_NAME}/{{split}}/{P}/best_model.pkl",
    params:
        score_to_rank="loss",
    run:
        best_model = pick_best_multi_model(input, params.score_to_rank)
        with open(output[0], "wb") as f:
            pickle.dump(best_model, f)


module attention_pretraining:
    snakefile:
        "../training/attention_pretraining_from_omop.smk"

use rule * from attention_pretraining



module demo:
    snakefile:
        "../training/demo_baseline_model.smk"


use rule * from demo
