import numpy as np
from tqdm import tqdm
import h5py
import pickle
import pandas as pd


from tte.utils.misc import PATHS
from tte.data.omop_processing import drugs_to_ingredients

CONCEPTS = PATHS.CONCEPTS
DAYS_IN_YEAR = 365.25
NAN_VALUE = -9.0


def prefetch_ingredient_table(
    main_table,
    baseline_dates,
    birth_dates,
    save_h5fn,
    remove_baseline_date_data,
    max_exposure_days,
    drop_missing_ingredients,
    drug_strength_path,
    remove_invalid_concepts,
):
    """reads in the raw omop data

    output:
        list of all relevant concepts to subset
        h5 file with entry for each iid. for each iid, a dict of {concept_name: (date, value)}; for binary concepts, value is always 1
    """
    date_field = "drug_exposure_start_date"
    id_col = "drug_concept_id"

    main_table[date_field] = pd.to_datetime(main_table[date_field], dayfirst=True)

    vlen_str_dtype = h5py.special_dtype(vlen=str)
    dtype = np.dtype(
        [("concept_name", vlen_str_dtype), ("age", np.float32), ("value", np.int8)]
    )

    unique_drugs = np.unique(main_table[id_col])
    ingredients = drugs_to_ingredients(
        unique_drugs,
        drop_missing=drop_missing_ingredients,
        drug_strength_path=drug_strength_path,
    )
    unique_ingredients = np.unique(
        np.concatenate(list(ingredients.values())).astype(int)
    )

    # concepts = main_table[id_col].unique()
    lookup = vocab_lookup_and_filter(unique_ingredients, remove_invalid_concepts=remove_invalid_concepts)

    all_concepts = set()
    with h5py.File(save_h5fn, "w") as f:
        for iid, sub_df in tqdm(main_table.groupby("eid", sort=False)):
            baseline_date = baseline_dates.loc[iid]
            row_date_field = pd.to_datetime(sub_df[date_field], dayfirst=True)
            if remove_baseline_date_data:
                sub_df = sub_df.loc[row_date_field < baseline_date]
            else:
                sub_df = sub_df.loc[row_date_field <= baseline_date]

            sub_df = sub_df.loc[
                (baseline_date - sub_df[date_field]).dt.days < max_exposure_days
            ]

            data = []
            for _, row in sub_df.iterrows():
                row_ingredients = ingredients[row[id_col]]
                age = (row[date_field] - birth_dates.loc[iid]).days / DAYS_IN_YEAR
                for ingredient in row_ingredients:
                    concept_name = lookup[ingredient]
                    if not concept_name.startswith("unknown_concept_"):
                        data.append((concept_name, age, NAN_VALUE))
                        all_concepts.update([concept_name])
            data = np.array(data, dtype=dtype)

            # sub_df["concept_name"] = sub_df[id_col].apply(lambda x: lookup[x]).values

            # data = np.array(
            #     [
            #         (concept_name, (date - birth_dates.loc[iid]).days / DAYS_IN_YEAR, NAN_VALUE)
            #         for date, concept_name in zip(
            #             sub_df[date_field], sub_df["concept_name"]
            #         )
            #     ],
            #     dtype=dtype,
            # )
            f.create_dataset(str(iid), data=data)
            # all_concepts.update(sub_df["concept_name"].unique())

        print("filling up missing iids with empty dict")
        handled_iids = set(main_table.eid.unique())
        missing_iids = np.array(
            [iid for iid in baseline_dates.index if iid not in handled_iids]
        )
        print(f"n elements before: {len(f)}")

        for iid in tqdm(missing_iids):
            f.create_dataset(str(iid), data=np.array([], dtype=dtype))
        print(f"n elements after: {len(f)}")
    return all_concepts


def prefetch_extra_table(
    meta,
    save_h5fn,
):
    vlen_str_dtype = h5py.special_dtype(vlen=str)
    dtype = np.dtype(
        [("concept_name", vlen_str_dtype), ("age", np.float32), ("value", np.float32)]
    )
    with h5py.File(save_h5fn, "w") as f:
        for iid, row in tqdm(meta.iterrows()):
            age = row.age_at_recruitment
            normalized_age = row.age_at_baseline
            data = np.array(
                [
                    ("sex_male" if row.sex_male == 1 else "sex_female", age, NAN_VALUE),
                    ("age_at_baseline", age, normalized_age),
                ],
                dtype=dtype,
            )
            f.create_dataset(str(iid), data=data)
    return ["age_at_baseline", "sex_male", "sex_female"]


def prefetch_source_table(
    source,
    main_table,
    baseline_dates,
    birth_dates,
    save_h5fn,
    remove_baseline_date_data,
    max_exposure_days=8 * 365,
    remove_invalid_concepts=True,
):
    """reads in the raw omop data

    output:
        list of all relevant concepts to subset
        h5 file with entry for each iid. for each iid, a dict of {concept_name: (date, value)}; for binary concepts, value is always 1
    """

    assert source in [
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure",
    ], "this function only works for conditions and procedures"

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
    vlen_str_dtype = h5py.special_dtype(vlen=str)
    dtype = np.dtype(
        [("concept_name", vlen_str_dtype), ("age", np.float32), ("value", np.int8)]
    )

    concepts = main_table[id_col].unique()
    lookup = vocab_lookup_and_filter(concepts, remove_invalid_concepts=remove_invalid_concepts)
    all_concepts = set()
    with h5py.File(save_h5fn, "w") as f:
        for iid, sub_df in tqdm(main_table.groupby("eid", sort=False)):
            baseline_date = baseline_dates.loc[iid]
            row_date_field = pd.to_datetime(sub_df[date_field], dayfirst=True)
            if remove_baseline_date_data:
                sub_df = sub_df.loc[row_date_field < baseline_date]
            else:
                sub_df = sub_df.loc[row_date_field <= baseline_date]

            sub_df = sub_df.loc[
                (baseline_date - sub_df[date_field]).dt.days < max_exposure_days
            ]

            sub_df["concept_name"] = sub_df[id_col].apply(lambda x: lookup[x]).values
            sub_df = sub_df[~sub_df.concept_name.apply(lambda x: str(x).startswith("unknown_concept_"))]
            if len(sub_df) > 0:
                data = np.array(
                    [
                        (concept_name, (date - birth_dates.loc[iid]).days / DAYS_IN_YEAR, NAN_VALUE)
                        for date, concept_name in zip(
                            sub_df[date_field], sub_df["concept_name"]
                        )
                    ],
                    dtype=dtype,
                )
                all_concepts.update(sub_df["concept_name"].unique())
            else:
                data = np.array([], dtype=dtype)

            f.create_dataset(str(iid), data=data)

        print("filling up missing iids with empty dict")
        handled_iids = set(main_table.eid.unique())
        missing_iids = np.array(
            [iid for iid in baseline_dates.index if iid not in handled_iids]
        )
        print(f"n elements before: {len(f)}")

        for iid in tqdm(missing_iids):
            f.create_dataset(str(iid), data=np.array([], dtype=dtype))
        print(f"n elements after: {len(f)}")
    return all_concepts


def vocab_lookup_and_filter(concept_ids, remove_invalid_concepts=True):
    full_vocab = pd.read_csv(CONCEPTS, sep="\t", index_col=0)
    if remove_invalid_concepts:
        full_vocab = full_vocab[
            (full_vocab.invalid_reason.isna())
            & (full_vocab.valid_end_date == 20991231)
            & (full_vocab.standard_concept == "S")
        ]
    joint = np.intersect1d(concept_ids, full_vocab.index)

    return {
        idx: (
            str(full_vocab.loc[idx, "concept_name"])
            if idx in joint
            else f"unknown_concept_{idx}"
        )
        for idx in concept_ids
    }
