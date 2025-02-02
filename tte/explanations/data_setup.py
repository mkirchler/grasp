from os.path import join
import pandas as pd

from tte.utils.misc import *
from tte.data.set_data import setup_set_dataloader

AVAILS = "data/misc/available_data.csv"


def get_def(bs=32, nt=16, nw=0):
    return get_loader(
        0,
        ["condition_occurrence"],
        "all",
        bs=bs,
        n_token_fill_or_sample=nt,
        provider="openai_new",
        num_workers=nw,
        tsplit="train",
        binarize=True,
    )


def get_def_net():
    R = pickle.load(
        open(
            "export/attention_fulltable_endpointsel_from_omop_multi/attention_fulltable_endpointsel_from_omop_multi.pkl",
            "rb",
        )
    )
    models = R["models"]
    keys = list(models.keys())
    keys = [k for k in keys if "depth~4" in k and "embed_dim~256" in k and "0" in k]
    assert len(keys) == 1
    model = keys[0]
    net = models[model]['net']
    return net


def get_loader(
    split,
    sources,
    endpoints,
    avail="condition_occurrence=1",
    indiv_exclusions=False,
    bs=512,
    n_token_fill_or_sample=64,
    provider="openai",
    num_workers=12,
    tsplit="test",
    binarize=False,
):
    if endpoints == "all":
        paths = load_paths()
        ENDPOINTS = pd.read_csv(paths.MANUAL_ENDPOINTS, header=None).squeeze().tolist()
        endpoints = ENDPOINTS
        if indiv_exclusions:
            print(
                "\n\n\nWARNING: using all endpoints but also excluding any prior endpoints --> will result in biased & small train set!\n\n\n"
            )
    survival = pd.read_csv(PATHS.SURVIVAL_PREPROCESSED, index_col=0)
    survival = filter_survival_endpoints(survival, endpoints)
    if indiv_exclusions:
        exclusions = [
            PATHS.FULL_EXCLUSIONS_FROM_OMOP.format(endpoint=endpoint)
            for endpoint in endpoints
        ]
    else:
        exclusions = None

    tiids, viids, ttiids = load_iids(int(split), exclusion_fn=exclusions)
    tiids = np.intersect1d(tiids, survival.index)
    viids = np.intersect1d(viids, survival.index)
    ttiids = np.intersect1d(ttiids, survival.index)

    print(
        f"Using {len(tiids)} training, {len(viids)} validation, and {len(ttiids)} test patients"
    )
    avails = pd.read_csv(AVAILS, index_col=0)
    avail_req = {v.split("=")[0]: int(v.split("=")[1]) for v in avail.split("?")}
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
    print(
        f"After filtering visits: {len(tiids)} training, {len(viids)} validation, and {len(ttiids)} test patients"
    )

    train = survival.loc[tiids]
    val = survival.loc[viids]
    test = survival.loc[ttiids]

    survival = {"train": train, "val": val, "test": test}[tsplit]

    prefetched_records = [
        join(PATHS.PROCESSED_SET_DATA_DIR, source, "records.h5") for source in sources
    ]
    records_h5fns = {
        source: record for source, record in zip(sources, prefetched_records)
    }

    prefetched_embeddings = [
        join(
            PATHS.PROCESSED_SET_DATA_DIR,
            f"input_type~none/model~none/pca_dim~0/provider~{provider}",
            f"{source}_active_embeddings.h5",
        )
        for source in sources
    ]
    embeddings_h5fns = {
        source: embed for source, embed in zip(sources, prefetched_embeddings)
    }

    loader = setup_set_dataloader(
        records_h5fns=records_h5fns,
        embeddings_h5fns=embeddings_h5fns,
        survival=survival,
        n_token_fill_or_sample=n_token_fill_or_sample,
        fill_value=0,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        binarize=binarize,
        empty_concepts=VARS.EMPTY_CONCEPTS,
    )
    return loader
