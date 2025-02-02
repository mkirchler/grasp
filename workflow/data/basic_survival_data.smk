from pathlib import Path
from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys
import pandas as pd

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.basic_survival_data import load_survival_endpoints, full_exclusion
from tte.utils.misc import load_vars, load_paths

paths = load_paths()
consts = load_vars()

CENTERS = load_vars().CENTERS
OLD_ENDPOINTS = load_vars().ENDPOINTS
ENDPOINTS_FROM_OMOP = (
    pd.read_csv(paths.MANUAL_ENDPOINTS, header=None).squeeze().tolist()
)


SPLITS = list(range(len(consts.CENTER_SPLITS)))


rule prepare_full_data_splits:
    input:
        [f"data/splits/{split}_train.csv" for split in SPLITS],
        [f"data/splits/{split}_val.csv" for split in SPLITS],
        [f"data/splits/{split}_test.csv" for split in SPLITS],


rule prepare_full_data_split:
    input:
        meta=paths.META_PREPROCESSED,
    output:
        train="data/splits/{split}_train.csv",
        val="data/splits/{split}_val.csv",
        test="data/splits/{split}_test.csv",
    params:
        seed_minus_split=42,
        val_pct=0.1,
    run:
        split = int(wildcards.split)
        centers = consts.CENTER_SPLITS[split]
        meta = pd.read_csv(input.meta, index_col=0)
        train_iids = meta[~meta.center.isin(centers)].index.values
        test_iids = meta[meta.center.isin(centers)].index.values

        seed = params.seed_minus_split + split
        np.random.RandomState(seed).shuffle(train_iids)
        m = int(len(train_iids) * (1 - params.val_pct))
        val_iids = train_iids[m:]
        train_iids = train_iids[:m]

        pd.Series(train_iids).to_csv(output.train, index=False, header=False)
        pd.Series(val_iids).to_csv(output.val, index=False, header=False)
        pd.Series(test_iids).to_csv(output.test, index=False, header=False)


rule prepare_data_splits:
    input:
        [f"data/splits/train_{center}.csv" for center in CENTERS],
        [f"data/splits/val_{center}.csv" for center in CENTERS],
        [f"data/splits/test_{center}.csv" for center in CENTERS],


rule prepare_data_split:
    input:
        meta=paths.META_PREPROCESSED,
        # meta=paths.META_PREPROCESSED_FROM_OMOP,
    output:
        train="data/splits/train_{center}.csv",
        val="data/splits/val_{center}.csv",
        test="data/splits/test_{center}.csv",
    params:
        seed_minus_center=42,
        val_pct=0.1,
    run:
        center = int(wildcards.center)
        meta = pd.read_csv(input.meta, index_col=0)
        train_iids = meta[meta.center != center].index.values
        test_iids = meta[meta.center == center].index.values

        seed = params.seed_minus_center + center
        np.random.RandomState(seed).shuffle(train_iids)
        m = int(len(train_iids) * (1 - params.val_pct))
        val_iids = train_iids[m:]
        train_iids = train_iids[:m]

        pd.Series(train_iids).to_csv(
            output.train.format(center=center), index=False, header=False
        )
        pd.Series(val_iids).to_csv(
            output.val.format(center=center), index=False, header=False
        )
        pd.Series(test_iids).to_csv(
            output.test.format(center=center), index=False, header=False
        )


rule process_survival_endpoints_from_omop:
    input:
        # endpoint_file=paths.UKB_ENDPOINTS_FROM_OMOP.format(cm="wcm"),
        endpoint_file="data/endpoints/endpoints_direct_from_omop.csv",
        pheno_file=paths.UKB_PHENO_FILE,
        omop_deaths_file=paths.OMOP_DEATHS_FILE,
    output:
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        meta=paths.META_PREPROCESSED,
        endpoint_specific_exclusions=paths.ENDPOINT_SPECIFIC_EXCLUSIONS_FROM_OMOP,
    params:
        input_selection=None,
        debug_nrows=None,
        baseline_date=None,  # None means use the assessment date
        washout_days=consts.WASHOUT_DAYS,  # should be at least 1 day
        censoring_offset_days=consts.CENSORING_OFFSET_DAYS,  # how long to follow up after the washout period
        # censoring_offset_days="max_all",  # "max_all" means use the max date across all endpoints, otherwise int 
        overall_end_of_followup=consts.OVERALL_END_OF_FOLLOWUP,  # e.g., "2020-12-31"
        clip_at_zero=True,
    run:
        assert (
            params.washout_days >= 1
        ), "washout_days must be at least 1 day, otherwise will include data from baseline date"
        survival_endpoints, metadata, endpoint_exclusions = load_survival_endpoints(
            endpoint_file=input.endpoint_file,
            pheno_file=input.pheno_file,
            endpoint_selection=params.input_selection,
            iids=None,
            debug_nrows=params.debug_nrows,
            baseline_date=params.baseline_date,
            censoring_offset_days=params.censoring_offset_days,
            washout_days=params.washout_days,
            omop_deaths_file=input.omop_deaths_file,
            overall_end_of_followup=params.overall_end_of_followup,
            clip_at_zero=params.clip_at_zero,
        )
        survival_endpoints.to_csv(output.survival, index=True)
        metadata.to_csv(output.meta, index=True)
        with open(output.endpoint_specific_exclusions, "wb") as f:
            pickle.dump(endpoint_exclusions, f)


# rule exclusion_propagation_from_omop:
#     input:
#         endpoint_file="data/endpoints/endpoints_direct_from_omop.csv",
#         pheno_file=paths.UKB_PHENO_FILE,
#         omop_deaths_file=paths.OMOP_DEATHS_FILE,
#     output:
#         # survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
#     params:
#         input_selection=None,
#         debug_nrows=None,
#         baseline_date=None,  # None means use the assessment date
#         washout_days=0,  # should be at least 1 day
#         censoring_offset_days=consts.CENSORING_OFFSET_DAYS,  # how long to follow up after the washout period
#         # censoring_offset_days="max_all",  # "max_all" means use the max date across all endpoints, otherwise int
#         overall_end_of_followup=consts.OVERALL_END_OF_FOLLOWUP,  # e.g., "2020-12-31"
#     run:
#         assert (
#             params.washout_days >= 1
#         ), "washout_days must be at least 1 day, otherwise will include data from baseline date"
#         survival_endpoints, metadata, endpoint_exclusions = load_survival_endpoints(
#             endpoint_file=input.endpoint_file,
#             pheno_file=input.pheno_file,
#             endpoint_selection=params.input_selection,
#             iids=None,
#             debug_nrows=params.debug_nrows,
#             baseline_date=params.baseline_date,
#             censoring_offset_days=params.censoring_offset_days,
#             washout_days=params.washout_days,
#             omop_deaths_file=input.omop_deaths_file,
#             overall_end_of_followup=params.overall_end_of_followup,
#         )
#         # survival_endpoints.to_csv(output.survival, index=True)
#         # metadata.to_csv(output.meta, index=True)
#         # with open(output.endpoint_specific_exclusions, "wb") as f:
#         #     pickle.dump(endpoint_exclusions, f)


rule t2d_extra_exclusions:
    # finds all eids that:
    # - never have a T2D endpoint event AND
    # - have a diabetes-related endpoint (e.g., T1D or general diabetes) AND
    # - were not excluded before
    # additionally, finds all eids that:
    # - have a T2D *after* baseline
    # - have a diabetes-related endpoint **before** baseline (e.g., T1D or general diabetes) AND
    # - were not excluded before
    input:
        control_endpoints="data/endpoints/direct_endpoints/control_exclusions_t2d.csv",
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        endpoint_specific_exclusions=paths.ENDPOINT_SPECIFIC_EXCLUSIONS_FROM_OMOP,
        meta=paths.META_PREPROCESSED,
    output:
        "data/endpoints/exclusions/t2d_extra_exclusions.csv",
    run:
        control = pd.read_csv(input.control_endpoints)
        survival = pd.read_csv(input.survival, index_col=0)
        excl = pickle.load(open(input.endpoint_specific_exclusions, "rb"))["T2D"]
        event = survival["T2D_event"].astype(bool)
        meta = pd.read_csv(input.meta, index_col=0)

        # control_endpoints = "data/endpoints/direct_endpoints/control_exclusions_t2d.csv"
        # survival = paths.SURVIVAL_PREPROCESSED_FROM_OMOP
        # endpoint_specific_exclusions = paths.ENDPOINT_SPECIFIC_EXCLUSIONS_FROM_OMOP
        # meta = paths.META_PREPROCESSED
        # control = pd.read_csv(control_endpoints)
        # survival = pd.read_csv(survival, index_col=0)
        # excl = pickle.load(open(endpoint_specific_exclusions, "rb"))["T2D"]
        # meta = pd.read_csv(meta, index_col=0)

        control = control[~control.eid.isin(excl)]

        control.date = pd.to_datetime(control.date)
        control.sort_values(["eid", "date"], inplace=True)
        control.drop_duplicates(subset=["eid"], keep="first", inplace=True)


        control_eids = control.eid.values
        control_exclusions = control_eids[np.where(~event.loc[control_eids])[0]]


        control = control.merge(
            meta.first_assessment_date, left_on="eid", right_index=True
        )
        t2d_case_after = event.index[np.where(event)[0]]
        d_case_before = control.eid[control.date < control.first_assessment_date].values
        case_exclusions = np.intersect1d(t2d_case_after, d_case_before)

        new_excl = np.union1d(control_exclusions, case_exclusions)
        pd.Series(new_excl).to_csv(output[0], index=False, header=False)


rule full_exclusions_from_omop:
    input:
        meta=paths.META_PREPROCESSED,
        endpoint_specific_exclusions=paths.ENDPOINT_SPECIFIC_EXCLUSIONS_FROM_OMOP,
    output:
        exclusions=paths.FULL_EXCLUSIONS_FROM_OMOP,
    params:
        min_age_at_baseline=30,
        max_age_at_baseline=75,
    run:
        meta = pd.read_csv(input.meta, index_col=0)
        endpoint_specific = pickle.load(open(input.endpoint_specific_exclusions, "rb"))[
            wildcards.endpoint
        ]
        everyone_to_exclude = full_exclusion(
            meta,
            wildcards.endpoint,
            endpoint_specific,
            min_age_at_baseline=params.min_age_at_baseline,
            max_age_at_baseline=params.max_age_at_baseline,
        )
        pd.Series(everyone_to_exclude).to_csv(
            output.exclusions, index=False, header=False
        )


rule all_endpoint_steps_from_omop:
    input:
        survival=paths.SURVIVAL_PREPROCESSED_FROM_OMOP,
        # splits=[f"data/splits/train_{center}.csv" for center in CENTERS],
        splits=[f"data/splits/{split}_train.csv" for split in SPLITS],
        exclusions=expand(paths.FULL_EXCLUSIONS_FROM_OMOP, endpoint=ENDPOINTS_FROM_OMOP),
        t2d_extra_exclusions="data/endpoints/exclusions/t2d_extra_exclusions.csv",
    default_target: True


###### TODO: move somewhere else?


rule date_ranges_in_endpoints:
    input:
        endpoint_file=paths.UKB_ENDPOINTS,
    output:
        "reports/endpoint_date_ranges.png",
    run:
        dates = pd.read_csv(
            input.endpoint_file, usecols=["date"], parse_dates=["date"]
        ).squeeze()
        min_year, max_year = dates.min().year, dates.max().year
        percentages = dict()
        for year in range(min_year, max_year + 1):
            pct = (dates.dt.year == year).mean() * 100
            print(f"{year}: {pct:.2f}%")
            percentages[year] = pct
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(
            list(percentages.keys()),
            list(percentages.values()),
            color="black",
        )
        ax.set_title("All endpoints")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent of records")
        plt.tight_layout()
        plt.savefig(output[0])


rule date_ranges_in_omop:
    output:
        paths.OMOP_DATE_RANGES,
        "reports/omop_date_ranges.png",
    params:
        sources=[
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "visit_occurrence",
        ],
        debug_nrows=None,
    run:
        percentages = dict()
        for source in params.sources:
            print(source)
            percentages[source] = dict()
            date_col = {
                "condition_occurrence": "condition_start_date",
                "drug_exposure": "drug_exposure_start_date",
                "measurement": "measurement_date",
                "observation": "observation_date",
                "procedure_occurrence": "procedure_date",
                "visit_occurrence": "visit_start_date",
            }[source]
            df = pd.read_csv(
                f"{paths.OMOP_PATH}/omop_{source}.txt",
                sep="\t",
                usecols=[date_col],
                parse_dates=[date_col],
                nrows=params.debug_nrows,
            ).squeeze()
            min_year, max_year = df.min().year, df.max().year
            for year in range(min_year, max_year + 1):
                pct = (df.dt.year == year).mean() * 100
                print(f"{year}: {pct:.2f}%")
                percentages[source][year] = pct
        n = len(params.sources)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(5 + n * 8, 5), sharey=True)
        plt.title("OMOP data")

        for i, source in enumerate(params.sources):
            ax = axes[i]
            ax.bar(
                list(percentages[source].keys()),
                list(percentages[source].values()),
                color="black",
            )
            ax.set_title(source)
            ax.set_xlabel("Year")
            ax.set_ylabel("Percent of records")
        plt.tight_layout()
        plt.savefig(output[1])
        with open(output[0], "wb") as f:
            pickle.dump(percentages, f)


############# old endpoints
# deprecated
# rule all_endpoint_steps_old:
#     input:
#         survival=paths.SURVIVAL_PREPROCESSED,
#         splits=[f"data/splits/train_{center}.csv" for center in CENTERS],
#         exclusions=expand(paths.FULL_EXCLUSIONS, endpoint=OLD_ENDPOINTS),
# deprecated!
# rule process_survival_endpoints:
#     input:
#         endpoint_file=paths.UKB_ENDPOINTS,
#         pheno_file=paths.UKB_PHENO_FILE,
#         omop_deaths_file=paths.OMOP_DEATHS_FILE,
#     output:
#         survival=paths.SURVIVAL_PREPROCESSED,
#         meta=paths.META_PREPROCESSED_OLD,
#         endpoint_specific_exclusions=paths.ENDPOINT_SPECIFIC_EXCLUSIONS,
#     params:
#         input_selection=None,
#         debug_nrows=None,
#         baseline_date=None,  # None means use the assessment date
#         washout_days=consts.WASHOUT_DAYS,  # should be at least 1 day
#         censoring_offset_days=consts.CENSORING_OFFSET_DAYS,  # how long to follow up after the washout period
#         # censoring_offset_days="max_all",  # "max_all" means use the max date across all endpoints, otherwise int
#         overall_end_of_followup=consts.OVERALL_END_OF_FOLLOWUP,  # e.g., "2020-12-31"
#     run:
#         assert (
#             params.washout_days >= 1
#         ), "washout_days must be at least 1 day, otherwise will include data from baseline date"
#         survival_endpoints, metadata, endpoint_exclusions = load_survival_endpoints(
#             endpoint_file=input.endpoint_file,
#             pheno_file=input.pheno_file,
#             endpoint_selection=params.input_selection,
#             iids=None,
#             debug_nrows=params.debug_nrows,
#             baseline_date=params.baseline_date,
#             censoring_offset_days=params.censoring_offset_days,
#             washout_days=params.washout_days,
#             omop_deaths_file=input.omop_deaths_file,
#             overall_end_of_followup=params.overall_end_of_followup,
#         )
#         survival_endpoints.to_csv(output.survival, index=True)
#         metadata.to_csv(output.meta, index=True)
#         with open(output.endpoint_specific_exclusions, "wb") as f:
#             pickle.dump(endpoint_exclusions, f)
# deprecated
# rule full_exclusions:
#     input:
#         meta=paths.META_PREPROCESSED_OLD,
#         endpoint_specific_exclusions=paths.ENDPOINT_SPECIFIC_EXCLUSIONS,
#     output:
#         exclusions=paths.FULL_EXCLUSIONS,
#     params:
#         min_age_at_baseline=30,
#         max_age_at_baseline=75,
#     run:
#         meta = pd.read_csv(input.meta, index_col=0)
#         endpoint_specific = pickle.load(open(input.endpoint_specific_exclusions, "rb"))[
#             wildcards.endpoint
#         ]
#         everyone_to_exclude = full_exclusion(
#             meta,
#             wildcards.endpoint,
#             endpoint_specific,
#             min_age_at_baseline=params.min_age_at_baseline,
#             max_age_at_baseline=params.max_age_at_baseline,
#         )
#         pd.Series(everyone_to_exclude).to_csv(
#             output.exclusions, index=False, header=False
#         )
