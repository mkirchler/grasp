import numpy as np
from os.path import join
from pathlib import Path
from lifelines.utils import concordance_index
import sys
import pandas as pd
import pickle
from snakemake.utils import Paramspace


sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.omop_loading import load_sparse_omop, propagate_ontology
from tte.utils.misc import (
    load_vars,
    load_paths,
    load_iids,
    to_bool,
    splitp,
    PATHS,
    load_splits,
    load_endpoints,
)
from tte.survival.xgboost_models import run_sparse_xgb_aft, aggregate_xgb_models
from tte.plotting.survival_plotting import plot_c_index_absolute

from tte.utils.model_selection import pick_best_model

paths = load_paths()
consts = load_vars()
SPLITS = load_splits()
ENDPOINTS = load_endpoints()
AVAIL = [
    "condition_occurrence=1",
]

SOURCES = [
    "extra?condition_occurrence?procedure_occurrence?drug_exposure",
]

# run on 0-split only!
# SPLITS = SPLITS[:1]
pre_sweep_exploratory = pd.DataFrame(
    [
        {
            "min_count": min_count,
            "binarize_counts": binarize_counts,
            "sources": sources,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_boost_round": num_boost_round,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "ancestors": ancestors,
        }
        for sources in SOURCES
        for min_count in [1]
        for binarize_counts in [True]
        for learning_rate in [0.1]
        # 2 always
        for max_depth in [2, 5, 25]
        # 500 always
        for num_boost_round in [500, 1500]
        # 1-3
        for aft_loss_distribution_scale in [0.1, 0.3, 1, 3, 10]
        #  --> 4-6 always
        for ancestors in [1, 2, 4, 6]
    ]

)

fine_sweep_params = pd.DataFrame(
    [
        {
            "min_count": min_count,
            "binarize_counts": binarize_counts,
            "sources": sources,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_boost_round": num_boost_round,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "ancestors": ancestors,
        }
        for sources in SOURCES
        for min_count in [1]
        for binarize_counts in [True]
        for learning_rate in [0.1]
        for max_depth in [2]
        for num_boost_round in [500]
        # from:
        # for aft_loss_distribution_scale in [3.]
        for aft_loss_distribution_scale in [3.]
        # for aft_loss_distribution_scale in [0.1, 0.3, 1, 3, 10]
        # from:
        for ancestors in [3, 4, 5, 6, 7]
        # for ancestors in [1, 2, 3, 4, 5, 6, 7, 8]
    ]
)
# sweep_params = pd.DataFrame(
#     [
#         {
#             "min_count": min_count,
#             "binarize_counts": binarize_counts,
#             "sources": sources,
#             "learning_rate": learning_rate,
#             "max_depth": max_depth,
#             "num_boost_round": num_boost_round,
#             "aft_loss_distribution_scale": aft_loss_distribution_scale,
#             "ancestors": ancestors,
#         }
#         for sources in SOURCES
#         for min_count in [100]
#         for binarize_counts in [True]
#         for learning_rate in [0.1]
#         for max_depth in [2]
#         for num_boost_round in [500]
#         for aft_loss_distribution_scale in [2.0]
#         # for ancestors in [0, 1, 2, 3]
#         for ancestors in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     ]
# )
# sweep_paramspace = Paramspace(pre_sweep_exploratory)
sweep_paramspace = Paramspace(fine_sweep_params)


rule full_export:
    input:
        results=expand(
            "results/prop_xgb/{endpoint}/avail~{avail}/{param}/model.pkl",
            param=sweep_paramspace.instance_patterns,
            endpoint=ENDPOINTS,
            avail=AVAIL,
        ),
        models=expand(
            "models/prop_xgb/{endpoint}/{split}/avail~{avail}/{param}/model.pkl",
            split=SPLITS,
            param=sweep_paramspace.instance_patterns,
            endpoint=ENDPOINTS,
            avail=AVAIL,
        ),
    output:
        results=f"export/prop_xgb/results.pkl",
        c_plot=f"export/prop_xgb/c_plot.png",
        best_model=f"export/prop_xgb/best_model.pkl",
    default_target: True
    run:
        recs = []
        recs_CI_lower = []
        recs_CI_upper = []
        for fn in input.results:
            R = pickle.load(open(fn, "rb"))
            global_c = R["global_c_index"]
            global_c_CI = R["global_c_index_CI"]
            endpoint = fn.split("/")[2]
            spec = dict(tuple(x.split("~")) for x in fn.split("/")[3:-1])
            specname = "_".join(f"{k}={v}" for k, v in spec.items())
            recs.append(
                {
                    "global_c": global_c,
                    "endpoint": endpoint,
                    "spec": specname,
                }
            )
            recs_CI_upper.append(
                {
                    "global_c_CI_upper": global_c_CI[1],
                    "endpoint": endpoint,
                    "spec": specname,
                }
            )
            recs_CI_lower.append(
                {
                    "global_c_CI_lower": global_c_CI[0],
                    "endpoint": endpoint,
                    "spec": specname,
                }
            )

        recs = pd.DataFrame(recs)
        specs = np.unique(recs["spec"])
        endpoints = np.unique(recs["endpoint"])
        recs.set_index("endpoint", inplace=True)
        cols = {spec: recs[recs["spec"] == spec]["global_c"] for spec in specs}
        recs = pd.DataFrame(cols)

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

        with open(output.results, "wb") as f:
            pickle.dump([recs, recs_CI_lower, recs_CI_upper], f)
        plot_c_index_absolute(recs, output.c_plot)

        best_idx = recs.mean().idxmax()

        exp_models = []
        for fn in input.models:
            endpoint = fn.split("/")[2]
            split = fn.split("/")[3]
            spec = dict(tuple(x.split("~")) for x in fn.split("/")[4:-1])
            specname = "_".join(f"{k}={v}" for k, v in spec.items())
            # if True:
            if specname == best_idx:
                model = pickle.load(open(fn, "rb"))
                exp_models.append(
                    {
                        "split": split,
                        "endpoint": endpoint,
                        "concepts": model["concepts"],
                        "concept_ids": model["concept_ids"],
                        "bst": model["bst"],
                        "c_index": model["test_cindex"],
                        **spec,
                    }
                )
        with open(output.best_model, "wb") as f:
            pickle.dump(exp_models, f)


rule aggregate_endpoint:
    input:
        expand(
            "models/prop_xgb/{endpoint}/{split}/avail~{{avail}}/{param}/model.pkl",
            endpoint="{endpoint}",
            split=SPLITS,
            param=sweep_paramspace.wildcard_pattern,
        ),
    output:
        f"results/prop_xgb/{{endpoint}}/avail~{{avail}}/{sweep_paramspace.wildcard_pattern}/model.pkl",
    run:
        res = aggregate_xgb_models(input, verbose=True)
        with open(output[0], "wb") as f:
            pickle.dump(res, f)



rule xgb_model_center_sweepstep:
    input:
        sparse_omop=paths.PROPAGATED_SPARSE_OMOP,
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        meta=paths.META_PREPROCESSED,
        train_iids="data/splits/{split}_train.csv",
        val_iids="data/splits/{split}_val.csv",
        test_iids="data/splits/{split}_test.csv",
        exclusions=paths.FULL_EXCLUSIONS_FROM_OMOP,
        avails="data/misc/available_data.csv",
    output:
        f"models/prop_xgb/{{endpoint}}/{{split}}/avail~{{avail}}/{sweep_paramspace.wildcard_pattern}/model.pkl",
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

        survival = pd.read_csv(input.survival, index_col=0)

        tiids = np.intersect1d(tiids, survival.index)
        viids = np.intersect1d(viids, survival.index)
        ttiids = np.intersect1d(ttiids, survival.index)

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

        sp_arr, iids, concepts, concept_ids = pickle.load(open(input.sparse_omop, "rb"))
        survival = survival.loc[iids]


        xgb_params = dict(
            max_depth=int(wildcards.max_depth),
            learning_rate=float(wildcards.learning_rate),
            aft_loss_distribution_scale=float(wildcards.aft_loss_distribution_scale),
        )

        res = run_sparse_xgb_aft(
            arr=sp_arr,
            survival=survival,
            train_iids=tiids,
            val_iids=viids,
            test_iids=ttiids,
            duration_col=f"{wildcards.endpoint}_duration",
            event_col=f"{wildcards.endpoint}_event",
            xgb_params=xgb_params,
            num_boost_round=int(wildcards.num_boost_round),
        )

        res["concepts"] = concepts
        res["concept_ids"] = concept_ids
        res.update(dict(wildcards))

        with open(output[0], "wb") as f:
            pickle.dump(res, f)


rule prop:
    input:
        [
            paths.PROPAGATED_SPARSE_OMOP.format(
                split=0,
                sources=s,
                min_count=100,
                binarize_counts=True,
                ancestors=2,
            )
            for s in SOURCES
        ],


rule propagate_onto_labels:
    input:
        paths.PROCESSED_SPARSE_OMOP.format(
            split="{split}", sources="{sources}", min_count=0, binarize_counts=True
        ),
    output:
        paths.PROPAGATED_SPARSE_OMOP,
    run:
        sp_arr, iids, concepts, concept_ids = pickle.load(open(input[0], "rb"))

        if "extra" in wildcards.sources:
            extra_cols = np.where(np.isin(concepts, ["age_at_baseline", "sex_male"]))[0]
        else:
            extra_cols = []
        filled_arr = propagate_ontology(
            sp_arr, concept_ids, max_sep=int(wildcards.ancestors), extra_cols=extra_cols
        )

        tiids, viids, ttiids = load_iids(wildcards.split)
        strain = set(tiids)
        train_bind = np.array([iid in strain for iid in iids])
        include_columns = np.array(
            ((filled_arr[train_bind] != 0).sum(0) >= int(wildcards.min_count))
        ).flatten()

        filled_arr = filled_arr[:, include_columns]
        concepts = concepts[include_columns]
        concept_ids = concept_ids[include_columns]

        with open(output[0], "wb") as f:
            pickle.dump((filled_arr, iids, concepts, concept_ids), f)


def input_fns(wildcards):
    sources = wildcards.sources.split("?")
    return [
        (
            PATHS.PROCESSED_OMOP.format(source=source)
            if source != "extra"
            else PATHS.PROCESSED_EXTRA
        )
        for source in sources
    ]


rule cache_sparse_omop_data:
    input:
        input_fns,
    output:
        paths.PROCESSED_SPARSE_OMOP,
    run:
        tiids, viids, ttiids = load_iids(wildcards.split)
        sources = sorted(splitp(wildcards.sources))
        sources = wildcards.sources.split("?")
        input_fns = {
            source: (
                PATHS.PROCESSED_OMOP.format(source=source)
                if source != "extra"
                else PATHS.PROCESSED_EXTRA
            )
            for source in sources
        }


        sp_arr, iids, concepts, concept_ids = load_sparse_omop(
            # sources,
            input_fns=input_fns,
            train_iids=tiids,
            min_count_per_concept=int(wildcards.min_count),
            binarize_counts=to_bool(wildcards.binarize_counts),
            remove_invalid_concepts=True,
        )
        pickle.dump(
            (sp_arr, iids, concepts, concept_ids),
            open(output[0], "wb"),
        )


module omop_processing:
    snakefile:
        "../data/omop_processing.smk"


use rule * from omop_processing
