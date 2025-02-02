import numpy as np
from time import sleep
import subprocess
from os.path import join
from pathlib import Path
import sys
import pandas as pd
import pickle
from snakemake.utils import Paramspace

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.training.setup_pretraining_models import run_attention_survival_model
from tte.utils.misc import (
    load_vars,
    load_paths,
    load_iids,
    to_bool,
    filter_survival_endpoints,
    splitp,
    load_endpoints,
    load_splits,
)


from config.sweeps.attention_pretraining import sweep_params, embed_params

paths = load_paths()
VARS = load_vars()
SPLITS = load_splits()
ENDPOINTS = load_endpoints()

WANDB_NAME = "pretraining_smalln"
WANDB_PROJECT = "omop_tte_v0.1"

sweep_params["n"] = 10_000
sweep_params = sweep_params[sorted(sweep_params.columns)]
sweep_paramspace = Paramspace(sweep_params)
embed_paramspace = Paramspace(embed_params)


def cleanup():
    print("Running cleanup operations...")
    subprocess.run(["sync"])
    # sleep(60)

    try:
        # subprocess.run(['nvidia-smi'])
        pass
    except Exception as e:
        print("not running nvidia-smi:", e)
    # sleep(60)
    print("Cleanup operations complete.")


wildcard_constraints:
    center="[0-9]+",


def prefetched_embeddings(wildcards):
    sources = wildcards.sources.split("?")
    return [
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern,
            f"{source}_active_embeddings.h5",
        )
        for source in sources
    ]


def prefetched_records(wildcards):
    sources = wildcards.sources.split("?")
    return [
        join(paths.PROCESSED_SET_DATA_DIR, source, "records.h5") for source in sources
    ]


def prefetched_records_pkl(wildcards):
    sources = wildcards.sources.split("?")
    return [
        join(paths.PROCESSED_SET_DATA_DIR, source, "records.pkl") for source in sources
    ]


rule records_to_pkl:
    input:
        join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.pkl"),
    run:
        import h5py
        import pickle
        from tqdm import tqdm

        codec = dict()
        with h5py.File(input[0], "r") as f:
            data = dict()
            n = len(f)
            for k, v in tqdm(f.items(), total=n):
                item = []
                arr = v[()]
                for concept, age, value in arr:
                    if concept not in codec:
                        codec[concept] = len(codec)
                    item.append([codec[concept], age, value])
                data[k] = item

        with open(output[0], "wb") as f:
            pickle.dump([data, codec], f)


def inp_exclusions(wildcards):
    if wildcards.endpoints == "all":
        endpoints = ENDPOINTS
    else:
        endpoints = splitp(wildcards.endpoints)
    return [
        paths.FULL_EXCLUSIONS_FROM_OMOP.format(endpoint=endpoint)
        for endpoint in endpoints
    ]


# e.g. avail = "condition_occurrence=10?drug_exposure=0"
# or avail = "condition_occurrence=0" (ie, not restriction)
rule attention_pretraining_model_center_sweepstep_endpointsel_avail:
    input:
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        prefetched_embeddings=prefetched_embeddings,
        prefetched_records=prefetched_records_pkl,
        # prefetched_records=prefetched_records,
        avails="data/misc/available_data.csv",
        # exclusions=paths.FULL_EXCLUSIONS_FROM_OMOP
        exclusions=inp_exclusions,
        train_iids="data/misc/smalln_splits/{split}_{n}.csv",
    output:
        f"models/pretraining_smalln/{{split}}/endpoints~{{endpoints}}/indiv_exclusion~{{indiv_exclusion}}/avail~{{avail}}/{embed_paramspace.wildcard_pattern}/{sweep_paramspace.wildcard_pattern}/model.pkl",
    params:
        use_gpu=None,  # None = auto detect if available
        # sources=SOURCES,
        empty_concepts=VARS.EMPTY_CONCEPTS,
        dummy=True,
    run:
        if wildcards.endpoints == "all":
            endpoints = ENDPOINTS
            if to_bool(wildcards.indiv_exclusion):
                print(
                    "\n\n\nWARNING: using all endpoints but also excluding any prior endpoints --> will result in biased & small train set!\n\n\n"
                )

        else:
            endpoints = splitp(wildcards.endpoints)
        sources = wildcards.sources.split("?")

        survival = pd.read_csv(input.survival, index_col=0)
        survival = filter_survival_endpoints(survival, endpoints)

        if to_bool(wildcards.indiv_exclusion):
            exclusions = input.exclusions
        else:
            exclusions = None

        tiids, viids, ttiids = load_iids(int(wildcards.split), exclusion_fn=exclusions)
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

        print(
            f"Before filtering visits: {len(tiids)} training, {len(viids)} validation, and {len(ttiids)} test patients"
        )

        tiids = np.intersect1d(tiids, inds)
        viids = np.intersect1d(viids, inds)
        ttiids = np.intersect1d(ttiids, inds)

        tiids_subset = pd.read_csv(input.train_iids, header=None)[0].values
        tiids = np.intersect1d(tiids, tiids_subset)


        print(
            f"After filtering visits: {len(tiids)} training, {len(viids)} validation, and {len(ttiids)} test patients"
        )

        train = survival.loc[tiids]
        val = survival.loc[viids]
        test = survival.loc[ttiids]

        extra = dict()
        (
            net,
            val_loss,
            test_loss,
            val_cindices,
            test_cindices,
            (test_durations, test_events, test_preds),
            training_time,
        ) = run_attention_survival_model(
            train=train,
            val=val,
            test=test,
            records_h5fns={
                source: record
                for source, record in zip(sources, input.prefetched_records)
            },
            embeddings_h5fns={
                source: embed
                for source, embed in zip(sources, input.prefetched_embeddings)
            },
            # h5fn=input.prefetched_embeddings[0],
            warmup_epochs=int(wildcards.warmup_epochs),
            min_epochs=int(wildcards.min_epochs),
            epochs=int(wildcards.epochs),
            early_stopping_patience=int(wildcards.early_stopping_patience),
            batch_size=int(wildcards.batch_size),
            lr=float(wildcards.lr),
            weight_decay=float(wildcards.weight_decay),
            optimizer=wildcards.optimizer,
            low_bit=to_bool(wildcards.low_bit),
            n_token_fill_or_sample=int(wildcards.n_token_fill_or_sample),
            fill_value=float(wildcards.fill_value),
            num_heads=int(wildcards.num_heads),
            depth=int(wildcards.depth),
            embed_dim=int(wildcards.embed_dim),
            binarize=to_bool(wildcards.binarize),
            trunc_pad_eval=to_bool(wildcards.trunc_pad_eval),
            positional_embeddings_normalized=to_bool(
                wildcards.positional_embeddings_normalized
            ),
            learnable_positional_factor=to_bool(wildcards.learnable_positional_factor),
            input_dim=None
            if wildcards.input_dim == "None"
            else int(wildcards.input_dim),
            loss_type=wildcards.loss_type,
            num_workers=int(wildcards.num_workers),
            empty_concepts=params.empty_concepts,
            wandb_project=WANDB_PROJECT,
            wandb_name=WANDB_NAME.format(**extra),
            use_gpu=params.use_gpu,
            log_params=dict(
                **wildcards,
                **params,
            ),
            age_embed_config={
                "use_age_embeddings": to_bool(wildcards.use_age_embeddings),
                "min_val": 30,
                "max_val": 80,
                "bins": 101,
            },
            val_embed_config={
                "use_val_embeddings": to_bool(wildcards.use_val_embeddings),
                "min_val": 0,
                "max_val": 100,
                "bins": 101,
            },
            positional_constant=to_bool(wildcards.positional_constant),
        )
        res = {
            "test_preds": test_preds,
            "test_cindices": test_cindices,
            "val_cindices": val_cindices,
            "test_loss": test_loss,
            "val_loss": val_loss,
            "net": net,
            "test_durations": test_durations,
            "test_events": test_events,
            "target_names": list(test_preds.columns),
            "training_time": training_time,
        }
        with open(output[0], "wb") as f:
            pickle.dump(res, f)


module visit_filtering:
    snakefile:
        "../data/visit_filtering.smk"


module set_data_processing:
    snakefile:
        "../data/set_data_processing.smk"


module smalln:
    snakefile:
        "../analyses/smalln.smk"


# use rule {filter_for_active_embeddings, process_set_data} from set_data_processing
use rule * from set_data_processing


use rule extract_min_available from visit_filtering


use rule prep_split from smalln
