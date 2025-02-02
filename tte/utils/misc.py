import pandas as pd
import pickle
import numpy as np
import toml
from box import Box


def to_bool(x):
    return {"True": True, "False": False}[x]


def load_vars():
    return toml.load("data/variables/consts.toml", Box)


def load_paths():
    return toml.load("data/variables/paths.toml", Box)


def splitp(x):
    return x.split("?")


def joinp(lst):
    return "?".join([str(x) for x in lst])


def filter_survival_endpoints(survival, endpoints):
    return survival[
        [
            x
            for x in survival.columns
            if any(x.startswith(endpoint) for endpoint in endpoints)
        ]
    ]


PATHS = load_paths()
VARS = load_vars()


def load_endpoints():
    return pd.read_csv(PATHS.MANUAL_ENDPOINTS, header=None).squeeze().tolist() + [
        "all_cause_death"
    ]


def load_splits():
    return list(range(len(VARS.CENTER_SPLITS)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_iids(split, exclusion_fn=None):
    tiids, viids, ttiids = [
        pd.read_csv(f"data/splits/{split}_{dset}.csv", header=None).squeeze().values
        for dset in ["train", "val", "test"]
    ]
    if exclusion_fn is not None:
        lt, lv, ltt = len(tiids), len(viids), len(ttiids)
        if not isinstance(exclusion_fn, list):
            exclusion_fn = [exclusion_fn]

        for excl in exclusion_fn:
            tiids, viids, ttiids = filter_for_exclusions(excl, tiids, viids, ttiids)
        print(
            f"Excluded {lt - len(tiids)}/{lt} train, {lv - len(viids)}/{lv} val, {ltt - len(ttiids)}/{ltt} test"
        )
    return tiids, viids, ttiids


def pick_best_model(input_files, pick_by="val_cindex"):
    all_res = []
    all_val_scores = []
    for path in input_files:
        res = pickle.load(open(path, "rb"))
        all_res.append(res)
        all_val_scores.append(res[pick_by])
    best_model = all_res[np.argmax(all_val_scores)]
    return best_model


def filter_for_exclusions(exclusion_fn, *args):
    exclusions = set(pd.read_csv(exclusion_fn, header=None).squeeze().values)
    new_args = []
    for iid_set in args:
        new_args.append(np.array([iid for iid in iid_set if iid not in exclusions]))
    if len(new_args) == 1:
        return new_args[0]
    return new_args


ENDPOINT_NAMES = {
    "J10_ASTHMA": "Asthma",
    "I9_CHD": "Coronary Heart Disease",
    "F5_DEPRESSIO": "Depression",
    "KNEE_ARTHROSIS": "Knee Arthrosis",
    "M13_RHEUMA": "Rheumatoid Arthritis",
    "T2D": "Type 2 Diabetes",
    "M13_GOUT": "Gout",
    "M13_OSTEOPOROSIS": "Osteoporosis",
    "I9_AF": "Atrial Fibrillation",
    "I9_VTE": "Venous Thromboembolism",
    "M13_ARTHTROSIS_COX": "Coxarthrosis",
    "N14_CHRONKIDNEYDIS": "Chronic Kidney Disease",
    "H7_GLAUCOMA": "Glaucoma",
    "I9_MI": "Myocardial Infarction",
    "I9_HEARTFAIL_NS": "Heart Failure",
    "L12_PSORIASIS": "Psoriasis",
    "I9_STR": "Stroke",
    "G6_SLEEPAPNO": "Sleep Apnea",
    "K11_IBD_STRICT": "Inflammatory Bowel Disease",
    "G6_EPLEPSY": "Epilepsy",
    "AUD_SWEDISH": "Alcohol Use Disorder",
    "all_cause_death": "All-Cause Mortality",
}
