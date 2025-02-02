import numpy as np
from pathlib import Path
from lifelines.utils import concordance_index
import sys
import pandas as pd
from lifelines import CoxPHFitter
import pickle
from snakemake.utils import Paramspace

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.basic_survival_data import load_survival_endpoints
from tte.survival.lifelines_models import run_lifelines_coxph, aggregate_lifeline_models
from tte.utils.model_selection import pick_best_model
from tte.utils.misc import (
    load_vars,
    load_paths,
    load_iids,
    filter_for_exclusions,
    load_splits,
    load_endpoints,
)
from config.sweeps.demo import sweep_params

paths = load_paths()
# CENTERS = load_vars().CENTERS
SPLITS = load_splits()
ENDPOINTS = load_endpoints()
# SPLITS = list(range(len(load_vars().CENTER_SPLITS)))
# ENDPOINTS = load_vars().ENDPOINTS
# ENDPOINTS = pd.read_csv(paths.MANUAL_ENDPOINTS, header=None).squeeze().tolist()

# CENTERS = CENTERS[2:5]
# ENDPOINTS = ENDPOINTS[1:2]

sweep_paramspace = Paramspace(sweep_params)


rule demo_models:
    input:
        expand(
            "results/demo/{endpoint}.pkl",
            endpoint=ENDPOINTS,
        ),


rule aggregate_demo_models_endpoint:
    input:
        expand("models/demo/{{endpoint}}/{split}/best_val_model.pkl", split=SPLITS),
    output:
        "results/demo/{endpoint}.pkl",
    run:
        res = aggregate_lifeline_models(input, verbose=True)
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule demo_model_center:
    input:
        expand(
            "models/demo/{endpoint}/{split}/avail~{avail}/{sweep}/model.pkl",
            split="{split}",
            endpoint="{endpoint}",
            sweep=sweep_paramspace.instance_patterns,
            avail="{avail}",
        ),
    output:
        "models/demo/{endpoint}/{split}/avail~{avail}/best_val_model.pkl",
    run:
        best_model = pick_best_model(input)
        with open(output[0], "wb") as f:
            pickle.dump(best_model, f)


rule demo_model_center_sweepstep:
    input:
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        meta=paths.META_PREPROCESSED,
        train_iids="data/splits/{split}_train.csv",
        val_iids="data/splits/{split}_val.csv",
        test_iids="data/splits/{split}_test.csv",
        exclusions=paths.FULL_EXCLUSIONS_FROM_OMOP,
        avails="data/misc/available_data.csv",
    output:
        f"models/demo/{{endpoint}}/{{split}}/avail~{{avail}}/{sweep_paramspace.wildcard_pattern}/model.pkl",
    params:
        rescale_age=True,
    run:
        if wildcards.endpoint == "T2D":
            exclusions = [
                input.exclusions,
                "data/endpoints/exclusions/t2d_extra_exclusions.csv",
            ]
        else:
            exclusions = input.exclusions
        tiids, viids, ttiids = load_iids(wildcards.split, exclusions)

        avails = pd.read_csv(input.avails, index_col=0)
        avail_req = {
            v.split("=")[0]: int(v.split("=")[1]) for v in wildcards.avail.split("?")
        }
        inds = pd.Series(True, index=avails.index)
        for key in avail_req:
            inds = inds & (avails[f"{key}_processed"] >= avail_req[key])
        inds = inds.index[inds].tolist()

        tiids = np.intersect1d(tiids, inds)
        viids = np.intersect1d(viids, inds)
        ttiids = np.intersect1d(ttiids, inds)

        meta = pd.read_csv(input.meta, index_col=0)
        if wildcards.endpoint in ["C3_BREAST", "C3_PROSTATE"]:
            cols = ["age_at_recruitment"]
        else:
            cols = ["sex_male", "age_at_recruitment"]

        survival = pd.read_csv(input.survival, index_col=0)
        survival = survival[
            [f"{wildcards.endpoint}_duration", f"{wildcards.endpoint}_event"]
        ]
        survival[cols] = meta[cols]

        res = run_lifelines_coxph(
            survival,
            tiids,
            viids,
            ttiids,
            penalizer=float(wildcards.penalizer),
            l1_ratio=float(wildcards.l1_ratio),
            duration_col=f"{wildcards.endpoint}_duration",
            event_col=f"{wildcards.endpoint}_event",
            normalize_columns=["age_at_recruitment"] if params.rescale_age else [],
        )

        with open(output[0], "wb") as f:
            pickle.dump(res, f)
