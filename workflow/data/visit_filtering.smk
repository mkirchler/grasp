import sys
import h5py
import pandas as pd
from os.path import join
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.utils.misc import (
    load_vars,
    load_paths,
    load_iids,
)

paths = load_paths()
consts = load_vars()


TABLES = [
    "condition_occurrence",
    "procedure_occurrence",
    "drug_exposure",
]


rule extract_min_available:
    input:
        tables=expand(join(paths.OMOP_PATH, "omop_{table}.txt"), table=TABLES),
        meta=paths.META_PREPROCESSED,
        records=[f"data/embeddings/{table}/records.h5" for table in TABLES],
    output:
        "data/misc/available_data.csv",
    params:
        debug_nrows=None,
    run:
        meta = pd.read_csv(
            input.meta, index_col=0, parse_dates=["first_assessment_date"]
        )
        avails = pd.DataFrame(index=meta.index)
        for table in TABLES:
            datecol = {
                "condition_occurrence": "condition_start_date",
                "procedure_occurrence": "procedure_date",
                "drug_exposure": "drug_exposure_start_date",
            }[table]
            print(f"processing {table}")
            data = pd.read_csv(
                join(paths.OMOP_PATH, f"omop_{table}.txt"),
                sep="\t",
                usecols=["eid", datecol],
                nrows=params.debug_nrows,
                parse_dates=[datecol],
                dayfirst=True,
            )
            data["first_assessment_date"] = meta.first_assessment_date.loc[
                data.eid
            ].values
            print(f"before filtering: {data.shape}, {data.eid.nunique()} unique eids")
            data = data[data[datecol] < data.first_assessment_date]
            print(f"after filtering: {data.shape}, {data.eid.nunique()} unique eids")
            avails[table] = False
            avails.loc[data.eid.unique(), table] = True

            # discrepancy due to 8y exposure time cutoff currently --> use {table}_processed instead
            records = h5py.File(f"data/embeddings/{table}/records.h5", "r")
            keys = list(records.keys())
            lens = {int(key): len(records[key]) for key in tqdm(keys)}
            avails[f"{table}_processed"] = lens


        avails.to_csv(output[0])


## outdated --> visits are very incomplete, so not really relevant?
rule filter_available_visits:
    input:
        meta=paths.META_PREPROCESSED,
        visit_table=join(paths.OMOP_PATH, "omop_visit_occurrence.txt"),
    output:
        "data/misc/available_visits.csv",
    run:
        meta = pd.read_csv(input.meta, index_col=0)
        visits = pd.read_csv(
            input.visit_table,
            sep="\t",
            nrows=None,
        )


        hes = visits[visits.visit_concept_id == 9201]
        prim_care = visits[visits.visit_concept_id == 262]
        assessment = visits[visits.visit_concept_id == 32693]

        df = pd.DataFrame(
            index=meta.index,
        )
        for name, sdf in [
            ("hes", hes),
            ("prim_care", prim_care),
            ("assessment", assessment),
        ]:
            ids = set(sdf.eid.unique())
            df[f"{name}_available"] = df.index.isin(ids).astype(int)
        df.to_csv(output[0])
