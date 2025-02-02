from os.path import join
import pickle
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.omop_processing import table_to_counts, ingredients_to_counts

from tte.utils.misc import load_vars, load_paths

consts = load_vars()
paths = load_paths()


COUNT_TABLES = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]


rule all:
    input:
        expand(
            paths.PROCESSED_OMOP,
            source=COUNT_TABLES + ["ingredient_exposure", "extra"],
        ),
        # paths.PROCESSED_OMOP.format(source="condition_occurrence"),
        # str(PROCESSED_OMOP).format(source='procedure_occurrence')


rule process_extra_table:
    input:
        meta=paths.META_PREPROCESSED,
    output:
        paths.PROCESSED_EXTRA,
        paths.EXTRA_METADATA,
    params:
        debug_nrows=None,
    run:
        meta = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "sex_male", "age_at_recruitment"],
            nrows=params.debug_nrows,
        )
        meta.rename({"age_at_recruitment": "age_at_baseline"}, axis=1, inplace=True)
        mean, std = meta.age_at_baseline.mean(), meta.age_at_baseline.std()

        metadata = {"age_at_baseline": {"mean": mean, "std": std}}
        meta.age_at_baseline = (meta.age_at_baseline - mean) / std

        meta.to_csv(output[0])
        with open(output[1], "wb") as f:
            pickle.dump(metadata, f)


rule process_ingredient_table:
    input:
        meta=paths.META_PREPROCESSED,
        main_table=join(paths.OMOP_PATH, "omop_drug_exposure.txt"),
        drug_strength=paths.DRUG_STRENGTH,
    output:
        paths.PROCESSED_OMOP.format(source="ingredient_exposure"),
    params:
        debug_nrows=None,
        remove_baseline_date_data=consts.REMOVE_BASELINE_DATE_DATA,
        max_exposure_days=consts.MAX_EXPOSURE_DAYS,
        # drop_missing_ingredients=True,
        drop_missing_ingredients=False,
    run:
        main_table = pd.read_csv(input.main_table, nrows=params.debug_nrows, sep="\t")
        baseline_dates = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "first_assessment_date"],
            parse_dates=["first_assessment_date"],
            dayfirst=False,
        ).squeeze()
        ingredients_to_counts(
            main_table=main_table,
            baseline_dates=baseline_dates,
            save_fn=output[0],
            remove_baseline_date_data=params.remove_baseline_date_data,
            max_exposure_days=params.max_exposure_days,
            drug_strength_path=input.drug_strength,
            drop_missing_ingredients=params.drop_missing_ingredients,
        )


rule process_count_table:
    input:
        meta=paths.META_PREPROCESSED,
        main_table=join(paths.OMOP_PATH, "omop_{source}.txt"),
    output:
        paths.PROCESSED_OMOP,
    params:
        debug_nrows=None,
        # remove_baseline_date_data=True,
        remove_baseline_date_data=consts.REMOVE_BASELINE_DATE_DATA,
        max_exposure_days=consts.MAX_EXPOSURE_DAYS,
    run:
        main_table = pd.read_csv(input.main_table, nrows=params.debug_nrows, sep="\t")
        baseline_dates = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "first_assessment_date"],
            parse_dates=["first_assessment_date"],
            dayfirst=False,
        ).squeeze()
        table_to_counts(
            source=wildcards.source,
            main_table=main_table,
            baseline_dates=baseline_dates,
            save_fn=output[0],
            remove_baseline_date_data=params.remove_baseline_date_data,
            max_exposure_days=params.max_exposure_days,
        )
