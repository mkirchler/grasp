import pickle

from lifelines.utils import concordance_index
import numpy as np


def pick_best_model(input_files, pick_by="val_cindex"):
    all_res = []
    all_val_scores = []
    for path in input_files:
        res = pickle.load(open(path, "rb"))
        all_res.append(res)
        all_val_scores.append(res[pick_by])
    best_model = all_res[np.argmax(all_val_scores)]
    return best_model


def pick_best_multi_model(input_files, score_to_rank="cindex"):
    assert score_to_rank in ["loss", "cindex"]
    all_res = []
    all_val_scores = []
    for path in input_files:
        res = pickle.load(open(path, "rb"))
        all_res.append(res)
        if score_to_rank == "cindex":
            scores = np.array(list(res["val_cindices"].values()))
            scores = scores[~np.isnan(scores)]
            score = np.mean(scores)
        elif score_to_rank == "loss":
            score = -res["val_loss"]
            if np.isnan(score):
                score = - np.inf
        all_val_scores.append(score)
    best_model = all_res[np.argmax(all_val_scores)]
    return best_model


import pandas as pd


def aggregate_multi_models(input_files, verbose=True):
    durations = []
    preds = []
    events = []
    local_c_indices = []
    ns = []
    for inp in input_files:
        res = pickle.load(open(inp, "rb"))
        durations.append(res["test_durations"])
        preds.append(res["test_preds"])
        events.append(res["test_events"])
        local_c_indices.append(res["test_cindices"])
        ns.append(res["test_preds"].shape[0])
    local_c_indices = pd.DataFrame(local_c_indices)
    weighted_mean_cindices = (local_c_indices.T * ns).sum(1) / sum(ns)
    durations = pd.concat(durations)
    preds = pd.concat(preds)
    events = pd.concat(events)
    global_c_indices = dict()
    if verbose:
        # print("local c-indices")
        # print(local_c_indices.T)
        # print("weighted mean c-indices")
        # print(weighted_mean_cindices)

        print("global c-indices")
        print(
            "{:<20} {:<15} {:<15}".format(
                "Target", "W-avg C-Index", "Global C-Index"
            )
        )
    for target in durations.columns:
        try:
            global_c_index = concordance_index(
                durations[target], -preds[target], events[target]
            )
        except ZeroDivisionError:
            global_c_index = np.nan
        if verbose:
            print(
                "{:<20} {:<15.4f} {:<15.4f}".format(
                    target,
                    weighted_mean_cindices.loc[target],
                    global_c_index,
                )
            )
        global_c_indices[target] = global_c_index
    return {
        "durations": durations,
        "preds": preds,
        "events": events,
        "global_c_indices": global_c_indices,
    }
