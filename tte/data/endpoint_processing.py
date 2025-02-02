import numpy as np
import pandas as pd

ICD10_CODE = 41270
ICD10_DATE = 41280

ICD9_CODE = 41271
ICD9_DATE = 41281


def load_icd_codes_long(
    pheno_file,
    debug_nrows=None,
    icd9=True,
    icd10=True,
    code_name="CODE1",
    date_name="DATE",
    eid_name="eid",
):
    sniff = pd.read_csv(pheno_file, nrows=1)
    template_codes = []
    if icd9:
        template_codes += [ICD9_CODE, ICD9_DATE]
    if icd10:
        template_codes += [ICD10_CODE, ICD10_DATE]

    cols = ["eid"] + [
        c
        for c in sniff.columns
        if any([c.startswith(f"{code}-") for code in template_codes])
    ]
    dtypes = {
        c: str
        for c in cols
        if c.startswith(f"{ICD10_CODE}-") or c.startswith(f"{ICD9_CODE}-")
    } | {"eid": int}
    dates = [
        c
        for c in cols
        if c.startswith(f"{ICD10_DATE}-") or c.startswith(f"{ICD9_DATE}-")
    ]
    icds = pd.read_csv(
        pheno_file,
        usecols=cols,
        nrows=debug_nrows,
        index_col=0,
        dtype=dtypes,
        parse_dates=dates,
    )
    if icd9:
        num_pairs = len(
            [col for col in icds.columns if col.startswith(f"{ICD9_CODE}-")]
        )
        col_pairs = [
            (f"{ICD9_CODE}-0.{i}", f"{ICD9_DATE}-0.{i}") for i in range(num_pairs)
        ]
        long_icd9 = pd.concat(
            [
                icds.loc[:, pair]
                .dropna()
                .rename({pair[0]: code_name, pair[1]: date_name}, axis=1)
                for pair in col_pairs
            ],
            axis=0,
        )
        long_icd9[code_name] = long_icd9[code_name].apply(lambda x: f"9x{x}")
    if icd10:
        num_pairs = len(
            [col for col in icds.columns if col.startswith(f"{ICD10_CODE}-")]
        )
        col_pairs = [
            (f"{ICD10_CODE}-0.{i}", f"{ICD10_DATE}-0.{i}") for i in range(num_pairs)
        ]
        long_icd10 = pd.concat(
            [
                icds.loc[:, pair]
                .dropna()
                .rename({pair[0]: code_name, pair[1]: date_name}, axis=1)
                for pair in col_pairs
            ],
            axis=0,
        )
        long_icd10[code_name] = long_icd10[code_name].apply(lambda x: f"10x{x}")
    icds_long = pd.concat([long_icd9, long_icd10], axis=0)

    return icds_long
