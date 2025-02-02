import numpy as np
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import xgboost as xgb
import pandas as pd
import pickle

from tte.utils.metrics import concordance_index_CI


def tmp_setup_data():
    from tte.utils.misc import load_paths, load_vars, load_iids

    paths = load_paths()
    split = 0
    endpoint = "AUD_SWEDISH"
    exclusions = paths.FULL_EXCLUSIONS_FROM_OMOP.format(endpoint=endpoint)
    tiids, viids, ttiids = load_iids(split, exclusions)

    survival = pd.read_csv(paths.SURVIVAL_PREPROCESSED_FROM_OMOP, index_col=0)

    tiids = np.intersect1d(tiids, survival.index)
    viids = np.intersect1d(viids, survival.index)
    ttiids = np.intersect1d(ttiids, survival.index)

    avails = "data/misc/available_data.csv"
    avails = pd.read_csv(avails, index_col=0)
    avail = "condition_occurrence=1"
    avail_req = {v.split("=")[0]: int(v.split("=")[1]) for v in avail.split("?")}
    inds = pd.Series(True, index=avails.index)
    for key in avail_req:
        inds = inds & (avails[f"{key}_processed"] >= avail_req[key])
    inds = inds.index[inds].tolist()
    tiids = np.intersect1d(tiids, inds)
    viids = np.intersect1d(viids, inds)
    ttiids = np.intersect1d(ttiids, inds)

    sparse_omop = paths.PROCESSED_SPARSE_OMOP.format(
        split=0,
        sources="extra?condition_occurrence?procedure_occurrence?drug_exposure",
        min_count=100,
        binarize_counts=True,
    )
    sp_arr, iids, concepts, concept_ids = pickle.load(open(sparse_omop, "rb"))
    survival = survival.loc[iids]

    xgb_params = dict(max_depth=2, learning_rate=0.1, aft_loss_distribution_scale=2.0)
    duration_col = f"{endpoint}_duration"
    event_col = f"{endpoint}_event"
    num_boost_rounds = 500
    return (
        sp_arr,
        survival,
        tiids,
        viids,
        ttiids,
        duration_col,
        event_col,
        xgb_params,
        num_boost_rounds,
    )


def all_setup(
    arr,
    survival,
    train_iids,
    val_iids,
    test_iids,
    duration_col="duration",
    event_col="event",
    xgb_params={},
    num_boost_round=500,
):
    survival = survival.copy()
    print(survival.head(), survival.shape)
    print("train before: ", train_iids.shape)
    print(train_iids)
    train_iids = np.intersect1d(train_iids, survival.index)
    print("train after: ", train_iids.shape)
    val_iids = np.intersect1d(val_iids, survival.index)
    test_iids = np.intersect1d(test_iids, survival.index)

    train_ = survival.loc[train_iids]
    val_ = survival.loc[val_iids]
    test_ = survival.loc[test_iids]

    train_ind = np.where(survival.index.isin(train_iids))[0]
    val_ind = np.where(survival.index.isin(val_iids))[0]
    test_ind = np.where(survival.index.isin(test_iids))[0]

    Xtr = arr[train_ind]
    Xv = arr[val_ind]
    Xte = arr[test_ind]

    train = get_sparse_xgb_aft_label_format(Xtr, train_, duration_col, event_col)
    val = get_sparse_xgb_aft_label_format(Xv, val_, duration_col, event_col)
    test = get_sparse_xgb_aft_label_format(Xte, test_, duration_col, event_col)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 2,
    }
    params.update(xgb_params)
    return params, train, val, test


def run_sparse_xgb_aft(
    arr,
    survival,
    train_iids,
    val_iids,
    test_iids,
    duration_col="duration",
    event_col="event",
    xgb_params={},
    num_boost_round=500,
    # normalize_columns=[],
):
    """
    arr & survival need to already be aligned in the rows
    """
    survival = survival.copy()
    print(survival.head(), survival.shape)
    print("train before: ", train_iids.shape)
    print(train_iids)
    train_iids = np.intersect1d(train_iids, survival.index)
    print("train after: ", train_iids.shape)
    val_iids = np.intersect1d(val_iids, survival.index)
    test_iids = np.intersect1d(test_iids, survival.index)

    train_ = survival.loc[train_iids]
    val_ = survival.loc[val_iids]
    test_ = survival.loc[test_iids]

    train_ind = np.where(survival.index.isin(train_iids))[0]
    val_ind = np.where(survival.index.isin(val_iids))[0]
    test_ind = np.where(survival.index.isin(test_iids))[0]

    Xtr = arr[train_ind]
    Xv = arr[val_ind]
    Xte = arr[test_ind]

    # scalers = dict()
    # print(normalize_columns, train_.head())
    # for col in normalize_columns:
    #     print(col)
    #     scaler = StandardScaler()
    #     scaler.fit(train_[col].values.reshape(-1, 1))
    #     train_[col] = scaler.transform(train_[col].values.reshape(-1, 1))
    #     val_[col] = scaler.transform(val_[col].values.reshape(-1, 1))
    #     test_[col] = scaler.transform(test_[col].values.reshape(-1, 1))

    #     scalers[col] = scaler
    train = get_sparse_xgb_aft_label_format(Xtr, train_, duration_col, event_col)
    val = get_sparse_xgb_aft_label_format(Xv, val_, duration_col, event_col)
    test = get_sparse_xgb_aft_label_format(Xte, test_, duration_col, event_col)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 2,
    }
    params.update(xgb_params)
    bst = xgb.train(
        params,
        train,
        num_boost_round=num_boost_round,
        evals=[(train, "train"), (val, "val")],
    )

    train_preds = bst.predict(train)
    val_preds = bst.predict(val)
    test_preds = bst.predict(test)

    try:
        train_cindex = concordance_index(
            train_[duration_col], train_preds, train_[event_col]
        )
    except ZeroDivisionError:
        train_cindex = np.nan
    if len(val_) == 0:
        val_cindex = np.nan
    else:
        try:
            val_cindex = concordance_index(
                val_[duration_col], val_preds, val_[event_col]
            )
        except ZeroDivisionError:
            val_cindex = np.nan
    try:
        test_cindex = concordance_index(
            test_[duration_col], test_preds, test_[event_col]
        )
    except ZeroDivisionError:
        test_cindex = np.nan

    print(
        f"concordance index - train: {train_cindex:.4f} -- val_cindex: {val_cindex:.4f} -- test_cindex: {test_cindex:.4f}"
    )

    res = {
        "test_preds": pd.DataFrame(test_preds, index=test_.index),
        "test_durations": test_[duration_col],
        "test_events": test_[event_col],
        "val_cindex": val_cindex,
        "test_cindex": test_cindex,
        "bst": bst,
        # "scalers": scalers,
        "test_data": {"data": test.get_data(), "label": test.get_label()},
    }
    return res


def run_xgb_aft(
    df,
    train_iids,
    val_iids,
    test_iids,
    duration_col="duration",
    event_col="event",
    normalize_columns=[],
):
    raise NotImplementedError()
    df = df.copy()
    print(df.head(), df.shape)
    print("train before: ", train_iids.shape)
    print(train_iids)
    train_iids = np.intersect1d(train_iids, df.index)
    print("train after: ", train_iids.shape)
    val_iids = np.intersect1d(val_iids, df.index)
    test_iids = np.intersect1d(test_iids, df.index)

    train_ = df.loc[train_iids]
    val_ = df.loc[val_iids]
    test_ = df.loc[test_iids]

    scalers = dict()
    print(normalize_columns, train_.head())
    for col in normalize_columns:
        print(col)
        scaler = StandardScaler()
        scaler.fit(train_[col].values.reshape(-1, 1))
        train_[col] = scaler.transform(train_[col].values.reshape(-1, 1))
        val_[col] = scaler.transform(val_[col].values.reshape(-1, 1))
        test_[col] = scaler.transform(test_[col].values.reshape(-1, 1))

        scalers[col] = scaler
    train = get_xgb_aft_label_format(train_, duration_col, event_col)
    val = get_xgb_aft_label_format(val_, duration_col, event_col)
    test = get_xgb_aft_label_format(test_, duration_col, event_col)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 2,
    }
    bst = xgb.train(
        params, train, num_boost_round=500, evals=[(train, "train"), (val, "val")]
    )

    train_preds = bst.predict(train)
    val_preds = bst.predict(val)
    test_preds = bst.predict(test)

    try:
        train_cindex = concordance_index(
            train_[duration_col], train_preds, train_[event_col]
        )
    except ZeroDivisionError:
        train_cindex = np.nan
    if len(val_) == 0:
        val_cindex = np.nan
    else:
        try:
            val_cindex = concordance_index(
                val_[duration_col], val_preds, val_[event_col]
            )
        except ZeroDivisionError:
            val_cindex = np.nan
    try:
        test_cindex = concordance_index(
            test_[duration_col], test_preds, test_[event_col]
        )
    except ZeroDivisionError:
        test_cindex = np.nan

    print(
        f"concordance index - train: {train_cindex:.4f} -- val_cindex: {val_cindex:.4f} -- test_cindex: {test_cindex:.4f}"
    )

    res = {
        "test_preds": pd.DataFrame(test_preds, index=test_.index),
        "test_durations": test_[duration_col],
        "test_events": test_[event_col],
        "val_cindex": val_cindex,
        "test_cindex": test_cindex,
        "bst": bst,
        # "cph": cph,
        "scalers": scalers,
    }
    return res


def get_xgb_cox_label_format(df, duration_col, event_col):
    df = df.copy()
    target = np.where(df[event_col] == 1, df[duration_col], -df[duration_col])
    df.drop(columns=[duration_col, event_col], inplace=True)
    data = xgb.DMatrix(df, label=target)
    return data


def get_sparse_xgb_aft_label_format(arr, df, duration_col, event_col):
    df = df.copy()
    y_lower_bound = df[duration_col].values.copy()
    y_upper_bound = df[duration_col].values.copy()
    y_upper_bound[df[event_col] == 0] = np.inf
    data = xgb.DMatrix(arr)
    data.set_float_info("label_lower_bound", y_lower_bound)
    data.set_float_info("label_upper_bound", y_upper_bound)
    return data


def get_xgb_aft_label_format(df, duration_col, event_col):
    df = df.copy()

    y_lower_bound = df[duration_col].values.copy()
    y_upper_bound = df[duration_col].values.copy()
    y_upper_bound[df[event_col] == 0] = np.inf
    df.drop(columns=[duration_col, event_col], inplace=True)
    df.columns = [
        col.replace("[", "(").replace("]", ")").replace("<", "lt") for col in df.columns
    ]
    data = xgb.DMatrix(df)
    data.set_float_info("label_lower_bound", y_lower_bound)
    data.set_float_info("label_upper_bound", y_upper_bound)
    return data


def aggregate_xgb_models(
    input_files,
    verbose=True,
    bootstrap_confidence_level=0.95,
    n_bootstraps=1000,
    bootstrap_n_jobs=4,
    bootstrap_seed=None,
):
    durations = []
    preds = []
    events = []
    c_indices = []
    ns = []
    # params = []
    for inp in input_files:
        res = pickle.load(open(inp, "rb"))
        print(res["bst"])
        durations.append(res["test_durations"])
        preds.append(res["test_preds"])
        events.append(res["test_events"])
        c_indices.append(res["test_cindex"])
        ns.append(res["test_preds"].shape[0])
        # params.append(res["bst"])
    # avg_params = (pd.DataFrame(params).T * ns).sum(1) / sum(ns)
    # avg_params = pd.DataFrame({"coef": avg_params, "exp(coef)": np.exp(avg_params)})
    # if verbose:
    #     print(f"avg parameters")
    #     print(avg_params)

    durations = pd.concat(durations)
    preds = pd.concat(preds)
    events = pd.concat(events)
    c_indices = pd.Series(c_indices)
    weighted_mean_cindex = (c_indices * ns).sum() / sum(ns)
    weighted_std_cindex = np.sqrt(
        ((c_indices - weighted_mean_cindex) ** 2 * ns).sum() / sum(ns)
    )
    if verbose:
        print(f"mean cindex: {c_indices.mean():.4f} -- std: {c_indices.std():.4f}")
        print(
            f"size-weighted mean cindex: {weighted_mean_cindex:.4f} -- std: {weighted_std_cindex:.4f}"
        )
    # global_c_index = concordance_index(durations, preds, events)
    global_c_index, c_index_CI = concordance_index_CI(
        durations,
        preds,
        events,
        confidence_level=bootstrap_confidence_level,
        n_bootstraps=n_bootstraps,
        n_jobs=bootstrap_n_jobs,
        seed=bootstrap_seed,
    )

    if verbose:
        print(f"global cindex: {global_c_index:.4f}")
    return {
        "durations": durations,
        "preds": preds,
        "events": events,
        "c_indices": c_indices,
        "weighted_mean_cindex": weighted_mean_cindex,
        "weighted_std_cindex": weighted_std_cindex,
        "global_c_index": global_c_index,
        "global_c_index_CI": c_index_CI,
        # 'avg_params': avg_params,
    }
