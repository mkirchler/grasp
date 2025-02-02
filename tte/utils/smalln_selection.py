import pandas as pd
import numpy as np


from tte.utils.misc import *


def subselect_all(train_n=10_000, split=0, seed=123):
    avail = "condition_occurrence=1"
    paths = load_paths()
    endpoints = load_endpoints()
    survival = pd.read_csv(paths.SURVIVAL_PREPROCESSED_FROM_OMOP, index_col=0)
    survival = filter_survival_endpoints(survival, endpoints)
    tiids, viids, ttiids = load_iids(split, None)
    avails = "data/misc/available_data.csv"
    avails = pd.read_csv(avails, index_col=0)
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

    if train_n is None:
        train_n = len(tiids)
    subset_tiids = np.random.RandomState(seed).choice(tiids, train_n, replace=False)

    return subset_tiids
