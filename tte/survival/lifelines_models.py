import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from lifelines.utils import concordance_index
from tte.utils.metrics import concordance_index_CI


def run_lifelines_coxph(
    df,
    train_iids,
    val_iids,
    test_iids,
    penalizer,
    l1_ratio,
    duration_col="duration",
    event_col="event",
    normalize_columns=[],
):
    df = df.copy()
    print(df.head(), df.shape)
    print("train before: ", train_iids.shape)
    print(train_iids)
    train_iids = np.intersect1d(train_iids, df.index)
    print("train after: ", train_iids.shape)
    val_iids = np.intersect1d(val_iids, df.index)
    test_iids = np.intersect1d(test_iids, df.index)

    train = df.loc[train_iids]
    val = df.loc[val_iids]
    test = df.loc[test_iids]

    scalers = dict()
    print(normalize_columns, train.head())
    for col in normalize_columns:
        print(col)
        scaler = StandardScaler()
        scaler.fit(train[col].values.reshape(-1, 1))
        train[col] = scaler.transform(train[col].values.reshape(-1, 1))
        val[col] = scaler.transform(val[col].values.reshape(-1, 1))
        test[col] = scaler.transform(test[col].values.reshape(-1, 1))

        scalers[col] = scaler

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        cph.fit(
            train,
            duration_col=duration_col,
            event_col=event_col,
        )
    except ConvergenceError as e:
        print(f"Convergence error: {e}")
        n_events = train[event_col].sum()
        train_stats = train.drop([event_col, duration_col], axis=1).std()
        print(f"Number of events: {n_events}")
        print(f"Train stats: {train_stats}")
        if n_events > 0 and all(train_stats > 0.01):
            print("retrying with smaller step size")
            success = False
            for step_size in [0.5, 0.1, 0.01]:
                try:
                    cph.fit(
                        train,
                        duration_col=duration_col,
                        event_col=event_col,
                        fit_options=dict(step_size=step_size),
                    )
                    success = True
                    break
                except:
                    continue
            if not success:
                print(f'Failed to converge even with step size {step_size}')
                raise e
                    

    cph.print_summary()

    if len(val) == 0:
        val_cindex = np.nan
    else:
        try:
            val_cindex = cph.score(val, scoring_method="concordance_index")
        except ZeroDivisionError:
            val_cindex = np.nan
    try:
        test_cindex = cph.score(test, scoring_method="concordance_index")
    except ZeroDivisionError:
        test_cindex = np.nan
    test_preds = cph.predict_partial_hazard(test)

    print(
        f"concordance index - train: {cph.concordance_index_:.4f} -- val_cindex: {val_cindex:.4f} -- test_cindex: {test_cindex:.4f}"
    )

    res = {
        "test_preds": test_preds,
        "test_durations": test[duration_col],
        "test_events": test[event_col],
        "val_cindex": val_cindex,
        "test_cindex": test_cindex,
        "cph": cph,
        "scalers": scalers,
    }
    return res


def aggregate_lifeline_models(
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
    params = []
    for inp in input_files:
        res = pickle.load(open(inp, "rb"))
        res["cph"].print_summary()
        durations.append(res["test_durations"])
        preds.append(res["test_preds"])
        events.append(res["test_events"])
        c_indices.append(res["test_cindex"])
        ns.append(res["test_preds"].shape[0])
        params.append(res["cph"].params_)
    avg_params = (pd.DataFrame(params).T * ns).sum(1) / sum(ns)
    avg_params = pd.DataFrame({"coef": avg_params, "exp(coef)": np.exp(avg_params)})
    if verbose:
        print(f"avg parameters")
        print(avg_params)

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
    # global_c_index = concordance_index(durations, -preds, events)
    global_c_index, c_index_CI = concordance_index_CI(
        durations,
        -preds,
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
        "avg_params": avg_params,
    }
