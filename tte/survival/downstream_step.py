import numpy as np
from pathlib import Path
from lifelines.utils import concordance_index
import sys
import pandas as pd
from lifelines import CoxPHFitter
import pickle

from tte.utils.misc import load_vars, load_paths, load_iids
from tte.survival.lifelines_models import run_lifelines_coxph, aggregate_lifeline_models


def downstream_step(
    res_fn, split, exclusions, endpoint, meta_fn, rescale_age, penalizer, l1_ratio, dont_include_demo=False
):
    tiids, viids, ttiids = load_iids(int(split), exclusion_fn=exclusions)

    meta = pd.read_csv(meta_fn, index_col=0)
    if endpoint in ["C3_BREAST", "C3_PROSTATE"]:
        demo_cols = ["age_at_recruitment"]
    else:
        demo_cols = ["sex_male", "age_at_recruitment"]

    res = pickle.load(open(res_fn, "rb"))
    # print(res)
    durations = res["durations"][endpoint]
    events = res["events"][endpoint]
    risk_score = res["preds"][[endpoint]]

    if not dont_include_demo:
        risk_score[demo_cols] = meta.loc[risk_score.index, demo_cols]

    risk_score["duration"] = durations
    risk_score["event"] = events

    normalize_columns = [endpoint]
    if rescale_age and not dont_include_demo:
        normalize_columns.append("age_at_recruitment")

    res = run_lifelines_coxph(
        risk_score,
        tiids,
        viids,
        ttiids,
        penalizer=float(penalizer),
        l1_ratio=float(l1_ratio),
        normalize_columns=normalize_columns,
    )
    return res
    with open(output[0], "wb") as f:
        pickle.dump(res, f)
