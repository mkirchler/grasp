import pandas as pd
import numpy as np

DAYS_IN_YEAR = 365.25
OMOP_DEATHS_FILE = "~/tmp_ukb_mirror/omop/omop_death.txt"



def load_survival_endpoints_old(
    endpoint_file,
    pheno_file,
    endpoint_selection=None,
    iids=None,
    debug_nrows=None,
    baseline_date=None,
    censoring_offset_days="max_all",
    washout_days=0,
    omop_deaths_file=OMOP_DEATHS_FILE,
    overall_end_of_followup=None,
):

    endpoints = pd.read_csv(
        endpoint_file,
        index_col=0,
        parse_dates=["date"],
        # only if direct from omop there's an additional source_concept_id col
        usecols=["eid", "date", "endpoint"],
    )
    if not endpoint_selection is None:
        endpoint_selection = set(endpoint_selection)
        endpoints = endpoints[endpoints["endpoint"].isin(endpoint_selection)]
    endpoint_selection = np.unique(endpoints.endpoint)

    if overall_end_of_followup is None:
        overall_end_of_followup = "2099-12-31"
    overall_end_of_followup = pd.to_datetime(overall_end_of_followup)
    n_before = len(endpoints)
    endpoints = endpoints[endpoints.date <= overall_end_of_followup]
    print(
        f"filtered out {n_before - len(endpoints)} endpoints that were after overall end of followup"
    )

    if not iids is None:
        iids = set(iids)
        endpoints = endpoints[endpoints.index.isin(iids)]

    meta = get_birth_and_assessment_dates(
        pheno_file,
        # only_basics=True,
        debug_nrows=debug_nrows,
        omop_deaths_file=omop_deaths_file,
    )

    if baseline_date is None:
        baseline_date = meta.first_assessment_date
    else:
        baseline_date = pd.Series(pd.Timestamp(baseline_date), index=meta.index)
    washout_date = baseline_date + pd.Timedelta(days=washout_days)
    if censoring_offset_days == "max_all":
        censoring_date = endpoints.date.max()
    else:
        censoring_date = washout_date + pd.Timedelta(days=censoring_offset_days)
    print()
    print()
    print("censoring dates before min of censoring date and overall end of followup")
    print(censoring_date.describe())
    censoring_date = censoring_date.apply(lambda x: min(x, overall_end_of_followup))
    print("censoring dates after min of censoring date and overall end of followup")
    print(censoring_date.describe())
    print()
    print()
    # else:
    #     censoring_date = pd.to_datetime(default_censoring_date)
    meta["censoring_date"] = censoring_date
    meta["censoring_date"] = meta[["censoring_date", "death_date"]].min(axis=1)

    metaindex = set(meta.index)
    available_iids = [iid for iid in endpoints.index.unique() if iid in metaindex]
    endpoints = endpoints.loc[available_iids]
    endpoints["baseline_date"] = baseline_date.loc[endpoints.index]
    endpoints["washout_date"] = washout_date.loc[endpoints.index]
    endpoints["censoring_date"] = censoring_date.loc[endpoints.index]
    endpoints["iid"] = endpoints.index

    # endpoints_prior_baseline = endpoints[
    #     endpoints["date"] <= endpoints["baseline_date"]
    # ]
    # endpoints_during_washout = endpoints[
    #     (endpoints["date"] > baseline_date)
    #     & (endpoints["date"] <= endpoints["washout_date"])
    # ]
    endpoints_up_to_washout_end = endpoints[
        (endpoints["date"] <= endpoints["washout_date"])
    ]
    endpoints_during_followup = endpoints[
        (endpoints["date"] > endpoints["washout_date"])
        & (endpoints["date"] <= endpoints["censoring_date"])
    ]
    # endpoints_after_followup = endpoints[
    #     endpoints["date"] > endpoints["censoring_date"]
    # ]

    endpoint_exclusions = {
        "all_cause_death": meta.loc[
            meta.death_date <= washout_date.loc[meta.index]
        ].index.values
    }
    for endpoint in endpoint_selection:
        endpoint_exclusions[endpoint] = compute_endpoint_exclusion_criteria(
            endpoint,
            endpoints_up_to_washout_end,
            endpoint_exclusions["all_cause_death"],
        )

    # endpoints = endpoints[endpoints["date"] > endpoints["baseline_date"]]

    # process actual endpoints used for survival analysis
    endpoints_during_followup = (
        endpoints_during_followup.sort_values(["iid", "endpoint", "date"])
        .drop_duplicates(["iid", "endpoint"], keep="first")
        .drop("iid", axis=1)
    )

    wide_endpoints = dict()
    for endpoint in endpoint_selection:
        sub_endpoints = endpoints_during_followup[
            endpoints_during_followup.endpoint == endpoint
        ]
        wide_endpoint = meta.loc[:, ["censoring_date", "first_assessment_date"]]
        wide_endpoint["event"] = 0
        wide_endpoint.loc[sub_endpoints.index, "event"] = 1
        wide_endpoint.loc[sub_endpoints.index, "censoring_date"] = sub_endpoints.date
        wide_endpoint["duration"] = (
            wide_endpoint.censoring_date - wide_endpoint.first_assessment_date
        ).apply(lambda x: x.days / DAYS_IN_YEAR)

        wide_endpoint = wide_endpoint[["event", "duration"]].rename(
            {"event": f"{endpoint}_event", "duration": f"{endpoint}_duration"}, axis=1
        )
        wide_endpoints[endpoint] = wide_endpoint
    wide_endpoints = pd.concat(wide_endpoints.values(), axis=1)
    wide_endpoints["all_cause_death_event"] = 1 * (~meta.death_date.isna())
    wide_endpoints["all_cause_death_duration"] = (
        meta.censoring_date - meta.first_assessment_date
    ).apply(lambda x: x.days / DAYS_IN_YEAR)

    meta = meta[
        [
            "sex_male",
            "first_assessment_date",
            "age_at_recruitment",
            "center",
            "birth_date",
        ]
    ]

    return wide_endpoints, meta, endpoint_exclusions
def load_survival_endpoints(
    endpoint_file,
    pheno_file,
    endpoint_selection=None,
    iids=None,
    debug_nrows=None,
    baseline_date=None,
    censoring_offset_days="max_all",
    washout_days=0,
    omop_deaths_file=OMOP_DEATHS_FILE,
    overall_end_of_followup=None,
    clip_at_zero=True,
):

    endpoints = pd.read_csv(
        endpoint_file,
        index_col=0,
        parse_dates=["date"],
        # only if direct from omop there's an additional source_concept_id col
        usecols=["eid", "date", "endpoint"],
    )
    if not endpoint_selection is None:
        endpoint_selection = set(endpoint_selection)
        endpoints = endpoints[endpoints["endpoint"].isin(endpoint_selection)]
    endpoint_selection = np.unique(endpoints.endpoint)

    if overall_end_of_followup is None:
        overall_end_of_followup = "2099-12-31"
    overall_end_of_followup = pd.to_datetime(overall_end_of_followup)
    n_before = len(endpoints)
    endpoints = endpoints[endpoints.date <= overall_end_of_followup]
    print(
        f"filtered out {n_before - len(endpoints)} endpoints that were after overall end of followup"
    )

    if not iids is None:
        iids = set(iids)
        endpoints = endpoints[endpoints.index.isin(iids)]

    meta = get_birth_and_assessment_dates(
        pheno_file,
        # only_basics=True,
        debug_nrows=debug_nrows,
        omop_deaths_file=omop_deaths_file,
    )

    if baseline_date is None:
        baseline_date = meta.first_assessment_date
    else:
        baseline_date = pd.Series(pd.Timestamp(baseline_date), index=meta.index)
    washout_date = baseline_date + pd.Timedelta(days=washout_days)
    if censoring_offset_days == "max_all":
        censoring_date = endpoints.date.max()
    else:
        censoring_date = washout_date + pd.Timedelta(days=censoring_offset_days)
    print()
    print()
    print("censoring dates before min of censoring date and overall end of followup")
    print(censoring_date.describe())
    censoring_date = censoring_date.apply(lambda x: min(x, overall_end_of_followup))
    print("censoring dates after min of censoring date and overall end of followup")
    print(censoring_date.describe())
    print()
    print()
    # else:
    #     censoring_date = pd.to_datetime(default_censoring_date)
    meta["censoring_date"] = censoring_date
    meta["censoring_date"] = meta[["censoring_date", "death_date"]].min(axis=1)

    metaindex = set(meta.index)
    available_iids = [iid for iid in endpoints.index.unique() if iid in metaindex]
    endpoints = endpoints.loc[available_iids]
    endpoints["baseline_date"] = baseline_date.loc[endpoints.index]
    endpoints["washout_date"] = washout_date.loc[endpoints.index]
    endpoints["censoring_date"] = censoring_date.loc[endpoints.index]
    endpoints["iid"] = endpoints.index

    # endpoints_up_to_washout_end = endpoints[
    #     (endpoints["date"] <= endpoints["washout_date"])
    # ]
    # endpoints_during_followup = endpoints[
    #     (endpoints["date"] > endpoints["washout_date"])
    #     & (endpoints["date"] <= endpoints["censoring_date"])
    # ]
    # endpoints_after_followup = endpoints[
    #     endpoints["date"] > endpoints["censoring_date"]
    # ]

    # endpoint_exclusions = {
    #     "all_cause_death": meta.loc[
    #         meta.death_date <= washout_date.loc[meta.index]
    #     ].index.values
    # }
    # for endpoint in endpoint_selection:
    #     endpoint_exclusions[endpoint] = compute_endpoint_exclusion_criteria(
    #         endpoint,
    #         endpoints_up_to_washout_end,
    #         endpoint_exclusions["all_cause_death"],
    #     )

    # endpoints = endpoints[endpoints["date"] > endpoints["baseline_date"]]

    # process actual endpoints used for survival analysis
    endpoints = endpoints.sort_values(['iid', 'endpoint', 'date']).drop_duplicates(['iid', 'endpoint'], keep='first').drop('iid', axis=1)
    # endpoints_during_followup = (
    #     endpoints_during_followup.sort_values(["iid", "endpoint", "date"])
    #     .drop_duplicates(["iid", "endpoint"], keep="first")
    #     .drop("iid", axis=1)
    # )

    wide_endpoints = dict()
    for endpoint in endpoint_selection:
        sub_endpoints = endpoints[ endpoints.endpoint == endpoint ]
        wide_endpoint = meta.loc[:, ["censoring_date", "first_assessment_date"]]
        wide_endpoint["event"] = 0
        wide_endpoint.loc[sub_endpoints.index, "event"] = 1
        wide_endpoint.loc[sub_endpoints.index, "censoring_date"] = sub_endpoints.date
        wide_endpoint["duration"] = (
            wide_endpoint.censoring_date - wide_endpoint.first_assessment_date
        ).apply(lambda x: x.days / DAYS_IN_YEAR)

        wide_endpoint = wide_endpoint[["event", "duration"]].rename(
            {"event": f"{endpoint}_event", "duration": f"{endpoint}_duration"}, axis=1
        )
        wide_endpoints[endpoint] = wide_endpoint
    wide_endpoints = pd.concat(wide_endpoints.values(), axis=1)
    wide_endpoints["all_cause_death_event"] = 1 * (~meta.death_date.isna())
    wide_endpoints["all_cause_death_duration"] = (
        meta.censoring_date - meta.first_assessment_date
    ).apply(lambda x: x.days / DAYS_IN_YEAR)

    death_excl = meta.loc[
            meta.death_date <= washout_date.loc[meta.index]
        ].index.values
    endpoint_exclusions = compute_endpoint_exclusion_criteria(wide_endpoints, death_excl, washout_days=washout_days)

    meta = meta[
        [
            "sex_male",
            "first_assessment_date",
            "age_at_recruitment",
            "center",
            "birth_date",
        ]
    ]
    if clip_at_zero:
        wide_endpoints = wide_endpoints.clip(lower=0)

    return wide_endpoints, meta, endpoint_exclusions

def compute_endpoint_exclusion_criteria(
    wide_endpoints,
    dead_iids,
    washout_days=2*365+1
):
    """returns all iids that had endpoint before washout end"""
    duration_cols = [col for col in wide_endpoints.columns if col.endswith("_duration")]
    exclusions = {col.split('_duration')[0]: wide_endpoints.index[wide_endpoints[col] < washout_days / DAYS_IN_YEAR] for col in duration_cols}
    wide_endpoints[duration_cols] < washout_days / DAYS_IN_YEAR
    exclusions = {endpoint: np.unique(np.concatenate([exclusions[endpoint], dead_iids])) for endpoint in exclusions}
    return exclusions

    # endpoints = endpoints_up_to_washout_end[
    #     endpoints_up_to_washout_end.endpoint == endpoint
    # ]
    # iids_to_exclude = (
    #     endpoints.sort_values(["iid", "date"]).drop_duplicates(["iid"], keep="first")
    # ).iid.values
    # iids_to_exclude = np.unique(np.concatenate([iids_to_exclude, dead_iids]))
    # return iids_to_exclude



def full_exclusion(
    meta, endpoint, endpoint_specific, min_age_at_baseline=30, max_age_at_baseline=75
):
    too_young = meta.index[meta.age_at_recruitment < min_age_at_baseline]
    too_old = meta.index[meta.age_at_recruitment > max_age_at_baseline]
    if endpoint == "C3_BREAST":
        wrong_sex = meta.index[meta.sex_male == 1]
    elif endpoint == "C3_PROSTATE":
        wrong_sex = meta.index[meta.sex_male == 0]
    else:
        wrong_sex = []
    everyone_to_exclude = np.unique(
        np.concatenate([too_young, too_old, wrong_sex, endpoint_specific])
    ).astype(int)
    return everyone_to_exclude


def compute_endpoint_exclusion_criteria_old(
    endpoint,
    endpoints_up_to_washout_end,
    dead_iids,
):
    """returns all iids that had endpoint before washout end"""
    endpoints = endpoints_up_to_washout_end[
        endpoints_up_to_washout_end.endpoint == endpoint
    ]
    iids_to_exclude = (
        endpoints.sort_values(["iid", "date"]).drop_duplicates(["iid"], keep="first")
    ).iid.values
    iids_to_exclude = np.unique(np.concatenate([iids_to_exclude, dead_iids]))
    return iids_to_exclude


def get_deaths(omop_deaths_file, meta):
    death_cols = ["eid", "death_date"]
    deaths = (
        pd.read_csv(
            omop_deaths_file,
            usecols=death_cols,
            parse_dates=["death_date"],
            dayfirst=True,
            sep="\t",
        )
        .sort_values(death_cols)
        .drop_duplicates(death_cols, keep="first")
    )
    deaths = deaths[deaths.eid.isin(meta.index)]
    deaths.set_index("eid", inplace=True)
    return deaths


def get_birth_and_assessment_dates(
    pheno_file,
    omop_deaths_file,
    debug_nrows=None,  # only_basics=True
):
    # if only_basics:
    #     columns = ["eid", "21022-0.0", "53-0.0", "31-0.0", "54-0.0"]
    #     colnames = ["age_at_recruitment", "first_assessment_date", "sex_male", "center"]
    # else:
    columns = [
        "eid",
        "34-0.0",
        "52-0.0",
        # "21022-0.0",
        "53-0.0",
        "31-0.0",
        "54-0.0",
    ]
    colnames = [
        "birth_year",
        "birth_month",
        # "age_at_recruitment",
        "first_assessment_date",
        "sex_male",
        "center",
    ]
    meta = pd.read_csv(
        pheno_file,
        index_col=0,
        nrows=debug_nrows,
        usecols=columns,
    ).rename(
        {key: name for key, name in zip(columns[1:], colnames)},
        axis=1,
    )
    meta.first_assessment_date = pd.to_datetime(meta.first_assessment_date)

    deaths = get_deaths(omop_deaths_file, meta)
    meta.loc[deaths.index, "death_date"] = deaths.death_date

    # if not only_basics:
    meta = meta[~meta["birth_year"].isna() & ~meta["birth_month"].isna()]
    meta["birth_date"] = meta.apply(
        lambda row: pd.Timestamp(
            year=int(row["birth_year"]), month=int(row["birth_month"]), day=15
        ),
        axis=1,
    )
    meta["age_at_recruitment"] = (meta.first_assessment_date - meta.birth_date).apply(
        lambda x: x.days / DAYS_IN_YEAR
    )

    meta.dropna(
        subset=[
            "sex_male",
            "first_assessment_date",
            "center",
            "age_at_recruitment",
            "birth_date",
        ],
        inplace=True,
    )
    return meta
