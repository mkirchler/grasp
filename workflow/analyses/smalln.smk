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
from tte.utils.smalln_selection import subselect_all
from tte.survival.lifelines_models import run_lifelines_coxph
from tte.utils.model_selection import pick_best_model

paths = load_paths()
SPLITS = load_splits()
ENDPOINTS = load_endpoints()
NS = [10_000, 25_000, 50_000, 100_000, 200_000]


rule aggregate_results:
    input:
        attn="export/attn_smalln/attn_smalln.pkl",
        xgb="export/prop_xgb_smalln/results.pkl",
    output:
        "extra_analyses/smalln_results.csv",
    run:
        attn = pickle.load(open(input.attn, "rb"))
        dfa = attn["c_index"]
        dfa_L = attn["c_index_CI_lower"]
        dfa_U = attn["c_index_CI_upper"]
        dfa.columns = pd.MultiIndex.from_tuples(
            [
                (
                    (
                        c.split("_provider=")[1].split("_sources")[0],
                        c.split("cox_n=")[1].split("_provider")[0],
                    )
                    if "cox" in c
                    else ("demo", c.split("baseline_")[1])
                )
                for c in dfa.columns
            ]
        )
        dfa_L.columns = dfa.columns
        dfa_U.columns = dfa.columns


        dfx, dfx_L, dfx_U = pd.read_pickle(input.xgb)
        dfx.columns = pd.MultiIndex.from_tuples(
            [
                (
                    "xgb",
                    c.split("n=")[1].split("_avail")[0],
                )
                for c in dfx.columns
            ]
        )
        dfx_L.columns = dfx.columns
        dfx_U.columns = dfx.columns
        full = pd.concat([dfa, dfx], axis=1)
        full_L = pd.concat([dfa_L, dfx_L], axis=1)
        full_U = pd.concat([dfa_U, dfx_U], axis=1)
        full.to_csv(output[0], header=True)


def count_full_n(endpoint="J10_ASTHMA", split=0, avail="condition_occurrence=1"):
    from tte.utils.misc import load_paths, load_iids

    paths = load_paths()
    input_exclusions = paths.FULL_EXCLUSIONS_FROM_OMOP.format(endpoint=endpoint)
    if endpoint == "T2D":
        exclusions = [
            input_exclusions,
            "data/endpoints/exclusions/t2d_extra_exclusions.csv",
        ]
    else:
        exclusions = input_exclusions
    tiids, viids, ttiids = load_iids(split, exclusions)
    input_survival = paths.SURVIVAL_PREPROCESSED_FROM_OMOP
    survival = pd.read_csv(input_survival, index_col=0)

    tiids = np.intersect1d(tiids, survival.index)
    viids = np.intersect1d(viids, survival.index)
    ttiids = np.intersect1d(ttiids, survival.index)
    input_avails = "data/misc/available_data.csv"
    avails = pd.read_csv(input_avails, index_col=0)
    avail_req = {v.split("=")[0]: int(v.split("=")[1]) for v in avail.split("?")}
    inds = pd.Series(True, index=avails.index)
    for key in avail_req:
        inds = inds & (avails[f"{key}_processed"] >= avail_req[key])
    inds = inds.index[inds].tolist()
    tiids = np.intersect1d(tiids, inds)
    return len(tiids)


def plot(df, df_L, df_U, out_fn, errors="errorbar"):
    from matplotlib import pyplot as plt

    means = df.mean()
    means_L = df_L.mean()
    means_U = df_U.mean()

    plt.figure(figsize=(10, 6))
    methods = means.index.levels[2]
    for method in methods:
        sub = means.loc[[x for x in means.index if x[2] == method]]
        sub.index = [int(x[1]) for x in sub.index]
        sub_L = means_L.loc[[x for x in means_L.index if x[2] == method]]
        sub_L.index = [int(x[1]) for x in sub_L.index]
        sub_U = means_U.loc[[x for x in means_U.index if x[2] == method]]
        sub_U.index = [int(x[1]) for x in sub_U.index]
        # sub.index = pd.to_numeric(sub.index, errors="coerce").fillna(max_val)
        sub = sub.sort_index()
        sub_L = sub_L.sort_index()
        sub_U = sub_U.sort_index()
        if errors == "shade":
            plt.plot(sub.index, sub.values, label=method)
            plt.fill_between(sub.index, sub_L.values, sub_U.values, alpha=0.2)
        elif errors == "errorbar":
            plt.errorbar(
                sub.index,
                sub.values,
                yerr=[sub.values - sub_L.values, sub_U.values - sub.values],
                fmt="-o",
                label=method,
            )
        elif errors == "none":
            plt.plot(sub.index, sub.values, label=method)
        else:
            raise ValueError(f"Unknown error type: {errors}")
        # plt.fill_between(sub.index, sub_L.values, sub_U.values, alpha=0.2, color="gray")
    plt.xlabel("n")
    plt.ylabel("Average Value")
    plt.title("Average Values per Method")
    plt.legend()
    plt.savefig(out_fn, bbox_inches="tight")


rule prep_splits:
    input:
        expand("data/misc/smalln_splits/{split}_{n}.csv", split=SPLITS, n=NS),


rule prep_split:
    output:
        "data/misc/smalln_splits/{split}_{n}.csv",
    params:
        seed=111,
    run:
        if wildcards.n == "None":
            n = None
        else:
            n = int(wildcards.n)
        train_iids = subselect_all(n, int(wildcards.split), seed=int(params.seed))
        pd.Series(train_iids).to_csv(output[0], index=False, header=False)


from config.sweeps.demo import sweep_params

sweep_paramspace = Paramspace(sweep_params)


rule demo_model_center:
    input:
        expand(
            "models/demo_smalln/{endpoint}/{split}/{n}/avail~{avail}/{sweep}/model.pkl",
            n="{n}",
            split="{split}",
            endpoint="{endpoint}",
            avail="{avail}",
            sweep=sweep_paramspace.instance_patterns,
        ),
    output:
        "models/demo_smalln/{endpoint}/{split}/{n}/{avail}/best_val_model.pkl",
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
        subset_train_iids="data/misc/smalln_splits/{split}_{n}.csv",
    output:
        f"models/demo_smalln/{{endpoint}}/{{split}}/{{n}}/avail~{{avail}}/{sweep_paramspace.wildcard_pattern}/model.pkl",
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

        tiids_subset = pd.read_csv(input.subset_train_iids, header=None)[0].values
        tiids = np.intersect1d(tiids, tiids_subset)

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
