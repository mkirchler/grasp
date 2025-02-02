import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from joblib import Parallel, delayed

from sksurv.metrics import integrated_brier_score

from tte.utils.misc import load_vars, load_paths

# from scipy.stats import bootstrap


def concordance_index_CI(
    event_times, predicted_scores, event_observed=None, confidence_level=0.95, n_bootstraps=1000, seed=None, n_jobs=1
):
    if n_jobs > 1 and seed is not None:
        raise ValueError("Cannot use n_jobs > 1 with fixed seed")
    assert confidence_level >= 0 and confidence_level <= 1
    alpha = (1 - confidence_level) / 2

    if isinstance(predicted_scores, (pd.Series, pd.DataFrame)):
        assert (event_times.index == predicted_scores.index).all()
        assert (event_times.index == event_observed.index).all()
        event_times = event_times.values.flatten()
        predicted_scores = predicted_scores.values.flatten()
        event_observed = event_observed.values.flatten()

    if n_jobs > 1:
        def compc(event_times, predicted_scores, event_observed):
            ind = np.random.choice(len(event_times), len(event_times), replace=True)
            sub_event_times = event_times[ind]
            sub_predicted_scores = predicted_scores[ind]
            sub_event_observed = event_observed[ind]
            c = concordance_index(
                sub_event_times, sub_predicted_scores, sub_event_observed
            )
            return c

        cs = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(compc)(
                event_times, predicted_scores, event_observed
            )
            for _ in tqdm(range(n_bootstraps))
        )
    else:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        rng = np.random.RandomState(seed)
        cs = []
        for _ in tqdm(range(n_bootstraps)):
            # c = compc(event_times, predicted_scores, event_observed)
            ind = rng.choice(len(event_times), len(event_times), replace=True)
            sub_event_times = event_times[ind]
            sub_predicted_scores = predicted_scores[ind]
            sub_event_observed = event_observed[ind]
            c = concordance_index(
                sub_event_times, sub_predicted_scores, sub_event_observed
            )
            cs.append(c)
    
    
    full_c = concordance_index(event_times, predicted_scores, event_observed)
    cs = np.array(cs)
    CI = np.quantile(cs, [alpha, 1 - alpha])
    return full_c, CI


def load_tmp():
    p = "models/attention_fulltable_endpointsel_from_omop_multi/avail~condition_occurrence=1/depth~4/embed_dim~256/endpoints~all/indiv_exclusion~False/input_dim~None/loss_type~cox/provider~openai_new/sources~condition_occurrence?procedure_occurrence?drug_exposure?extra/AUD_SWEDISH/0/best_val_model.pkl"
    A = pickle.load(open(p, "rb"))
    cph = A["cph"]
    test = A["test_data"]
    return cph, test


def standard_times(T=100):
    # DAYS_IN_YEAR = 365.25
    # consts = load_vars()
    # mn = (consts.WASHOUT_DAYS + 182) / DAYS_IN_YEAR
    mn = 2.5
    mx = 7.5
    return np.linspace(mn, mx, T)


def multi_ibs(cphs, test_datas, T=100):
    times = standard_times(T=T)
    survs = []
    valss = []
    for cph, test_data in zip(cphs, test_datas):
        surv = cph.predict_survival_function(test_data, times=times).T
        event_col = [col for col in test_data.columns if col.endswith("event")]
        assert len(event_col) == 1
        event_col = event_col[0]
        duration_col = [col for col in test_data.columns if col.endswith("duration")]
        assert len(duration_col) == 1
        duration_col = duration_col[0]
        vals = np.array(
            # list(zip(test_data.event, test_data.duration)),
            list(zip(test_data[event_col], test_data[duration_col])),
            dtype=[("event", bool), ("duration", float)],
        )
        survs.append(surv)
        valss.append(vals)
    survs = np.vstack(survs)
    valss = np.concatenate(valss)
    return integrated_brier_score(valss, valss, survs, times)


def compute_survival_times_aft_standard(preds, aft_scale, T=100):
    times = standard_times(T=T)
    # TODO


def compute_ibs_standard(cph, test_data, T=100):
    times = standard_times(T=T)
    surv = cph.predict_survival_function(test_data, times=times).T

    event_col = [col for col in test_data.columns if col.endswith("event")]
    assert len(event_col) == 1
    event_col = event_col[0]
    duration_col = [col for col in test_data.columns if col.endswith("duration")]
    assert len(duration_col) == 1
    duration_col = duration_col[0]
    vals = np.array(
        list(zip(test_data[event_col], test_data[duration_col])),
        dtype=[("event", bool), ("duration", float)],
    )
    return integrated_brier_score(vals, vals, surv, times)


def compute_ibs_times(cph, test_data, times):
    surv = cph.predict_survival_function(test_data, times=times).T
    vals = np.array(
        list(zip(test_data.event, test_data.duration)),
        dtype=[("event", bool), ("duration", float)],
    )
    return integrated_brier_score(vals, vals, surv, times)


def compute_ibs_minmax(cph, test_data, T=100):
    times = np.linspace(test_data.duration.min(), test_data.duration.max() - 0.01, T)
    surv = cph.predict_survival_function(test_data, times=times).T

    vals = np.array(
        list(zip(test_data.event, test_data.duration)),
        dtype=[("event", bool), ("duration", float)],
    )
    return integrated_brier_score(vals, vals, surv, times)


# def load_tmp_():
#     # p = "results/attention_fulltable_endpointsel_from_omop_multi/avail~condition_occurrence=1/depth~4/embed_dim~256/endpoints~all/indiv_exclusion~False/input_dim~None/loss_type~cox/provider~openai_new/sources~condition_occurrence?procedure_occurrence?drug_exposure?extra/AUD_SWEDISH.pkl"
#     p = "results/attention_fulltable_endpointsel_from_omop_multi/avail~condition_occurrence=1/depth~4/embed_dim~256/endpoints~all/indiv_exclusion~False/input_dim~None/loss_type~cox/provider~openai_new/sources~condition_occurrence?procedure_occurrence?drug_exposure?extra/results.pkl"
#     R = pickle.load(open(p, "rb"))
#     P = R['preds']
#     E = R['events']
#     D = R['durations']
#     c = R['global_c_index']
#     return P, E, D, c


#         # res = downstream_step(
#         #     res_fn=input[0],
#         #     split=wildcards.split,
#         #     exclusions=exclusions,
#         #     endpoint=wildcards.endpoint,
#         #     meta_fn=input.meta,
#         #     rescale_age=params.rescale_age,
#         #     penalizer=wildcards.penalizer,
#         #     l1_ratio=wildcards.l1_ratio,
#         # )
