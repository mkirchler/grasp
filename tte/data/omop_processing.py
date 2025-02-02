import pickle
from scipy import sparse
from tqdm import tqdm
import pandas as pd

import numpy as np

def ingredients_to_counts(
    main_table,
    baseline_dates,
    save_fn,
    remove_baseline_date_data,
    max_exposure_days,
    drug_strength_path,
    drop_missing_ingredients,
):
    date_field = "drug_exposure_start_date"
    id_col =  "drug_concept_id"

    main_table[date_field] = pd.to_datetime(main_table[date_field], dayfirst=True)

    unique_drugs = np.unique(main_table[id_col])
    ingredients = drugs_to_ingredients(
        unique_drugs, drop_missing=drop_missing_ingredients, drug_strength_path=drug_strength_path
    )
    unique_ingredients = np.unique(
        np.concatenate(list(ingredients.values())).astype(int)
    )

    coded = dict()
    for iid, sub_df in tqdm(main_table.groupby("eid", sort=False)):
        baseline_date = baseline_dates.loc[iid]
        row_date_field = pd.to_datetime(sub_df[date_field], dayfirst=True)
        if remove_baseline_date_data:
            sub_df = sub_df.loc[row_date_field < baseline_date]
        else:
            sub_df = sub_df.loc[row_date_field <= baseline_date]
        
        sub_df = sub_df.loc[
            (baseline_date - sub_df[date_field]).dt.days < max_exposure_days, id_col
        ]
        sub_ingredients = (
            np.concatenate([ingredients[drug_id] for drug_id in sub_df])
            if len(sub_df) > 0
            else np.array([])
        )
        vals = (sub_ingredients[None].T == unique_ingredients).sum(0)
        coded[iid] = sparse.csr_matrix(vals)

    unique_ingredients = unique_ingredients.flatten()
    iids = np.array(sorted(coded.keys()))
    coded = sparse.vstack([coded[key] for key in iids])

    pickle.dump([iids, unique_ingredients, coded], open(save_fn, "wb"))


def table_to_counts(
    source,
    main_table,
    baseline_dates,
    save_fn,
    remove_baseline_date_data,
    max_exposure_days,
):
    assert source in [
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure"
    ], "this function only works for conditions and procedures and drugs"

    date_field = {
        "condition_occurrence": "condition_start_date",
        "procedure_occurrence": "procedure_date",
        "drug_exposure": "drug_exposure_start_date",
    }[source]
    main_table[date_field] = pd.to_datetime(main_table[date_field], dayfirst=True)

    id_col = {
        "condition_occurrence": "condition_concept_id",
        "procedure_occurrence": "procedure_concept_id",
        "drug_exposure": "drug_concept_id",
    }[source]

    concepts = main_table[id_col].unique()
    concepts = concepts[None]

    coded = dict()

    coding_fn = individual_to_sparse_default

    for iid, sub_df in tqdm(main_table.groupby("eid", sort=False)):
        baseline_date = baseline_dates.loc[iid]
        row_date_field = pd.to_datetime(sub_df[date_field], dayfirst=True)
        if remove_baseline_date_data:
            sub_df = sub_df.loc[row_date_field < baseline_date]
        else:
            sub_df = sub_df.loc[row_date_field <= baseline_date]
        
        sub_df = sub_df.loc[
            (baseline_date - sub_df[date_field]).dt.days < max_exposure_days, id_col
        ]


        coded[iid] = coding_fn(sub_df, concepts)

    concepts = concepts.flatten()
    iids = np.array(sorted(coded.keys()))
    coded = sparse.vstack([coded[key] for key in iids])

    pickle.dump([iids, concepts, coded], open(save_fn, "wb"))

def individual_to_sparse_default(sub_df, concepts):
    return sparse.csr_matrix((sub_df.values[None].T == concepts).sum(0))


def drugs_to_ingredients(drug_concept_ids, drug_strength_path, drop_missing):
    drug_strength = pd.read_csv(drug_strength_path, sep='\t')

    ret = dict()

    for id in tqdm(drug_concept_ids):
        strength = drug_strength[drug_strength.drug_concept_id == id]
        ingredient_ids = list(strength.ingredient_concept_id)
        if not drop_missing and len(ingredient_ids) == 0:
            ingredient_ids = [id]
        ret[id] = ingredient_ids
    return ret
