from pathlib import Path
import numpy as np
import sys
import pandas as pd
from collections import defaultdict
from os.path import join
from tqdm import tqdm

from tte.utils.misc import load_paths

paths = load_paths()


def load_prelim(vocabularies=["ICD10", "ICD10CM"]):
    ukb_endpoints_from_omop = paths.UKB_ENDPOINTS_FROM_OMOP
    ukb_endpoints_from_icd = paths.UKB_ENDPOINTS
    omop = join(paths.OMOP_PATH, "omop_condition_occurrence.txt")
    concepts = paths.CONCEPTS
    concept_relationship = paths.CONCEPT_RELATIONSHIP
    print("reading endpoints...")
    from_omop = pd.read_csv(ukb_endpoints_from_omop)
    from_icd = pd.read_csv(ukb_endpoints_from_icd)
    print("reading omop...")
    main_table = pd.read_csv(
        omop,
        sep="\t",
        parse_dates=["condition_start_date"],
    )
    print("reading concepts...")
    concepts = pd.read_csv(
        concepts,
        sep="\t",
        usecols=["concept_id", "concept_name", "vocabulary_id", "concept_code"],
    )
    print("reading relationships...")
    R = pd.read_csv(
        concept_relationship,
        sep="\t",
    )
    R = R[R.relationship_id == "Maps to"]
    d = create_dict(concepts, R, vocabularies=vocabularies)
    raw_icd = load_icd_codes_long()
    return from_omop, from_icd, main_table, concepts, d, raw_icd


# from_omop, from_icd, main_table, concepts, d, raw_icd = load_prelim()
# compare_endpoints(
#     from_omop=from_omop,
#     from_icd=from_icd,
#     endpoint='T2D',
#     main_table=main_table,
#     raw_icd=raw_icd,
#     n_examples=25,
#     vocabularies=["ICD10", "ICD10CM"],
#     keywords=['diabetes'],
#     concepts=concepts,
#     d=d,
#     paths=paths,
# )


def compare_endpoints(
    from_omop,
    from_icd,
    endpoint,
    main_table,
    raw_icd,
    filter_dates_icd=None,
    icd_keys=["e11"],
    n_examples=25,
    vocabularies=["ICD10", "ICD10CM"],
    keywords=[],
    concepts=None,
    d=None,
    paths=paths,
):
    if paths is None:
        from tte.utils.misc import load_paths

        paths = load_paths()
    if filter_dates_icd is not None:
        from_icd = from_icd[from_icd.date <= filter_dates_icd]
    from_omop = from_omop[from_omop.endpoint == endpoint].eid.unique()
    from_icd = from_icd[from_icd.endpoint == endpoint].eid.unique()
    print(f"OMOP: {len(from_omop)}; ICD: {len(from_icd)}")
    omop_only = len(set(from_omop) - set(from_icd))
    icd_only = len(set(from_icd) - set(from_omop))
    both = len(set(from_omop) & set(from_icd))
    total = len(set(from_omop) | set(from_icd))
    all_iids = set(from_omop) | set(from_icd)
    print(f"OMOP: {omop_only}; ICD: {icd_only}; BOTH: {both}, TOTAL: {total}")
    if main_table is None:
        print("reading omop...")
        main_table = pd.read_csv(
            join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
            nrows=None,
            sep="\t",
            parse_dates=["condition_start_date"],
        )
    if concepts is None:
        concepts = pd.read_csv(
            paths.CONCEPTS,
            sep="\t",
            usecols=["concept_id", "concept_name"],
        )

    print("filtering...")
    main_table = main_table[main_table.eid.isin(all_iids)]
    main_table["concept_name"] = main_table.condition_concept_id.map(
        dict(zip(concepts.concept_id, concepts.concept_name))
    )
    if d is None:
        print("reading relationships...")
        R = pd.read_csv(
            paths.CONCEPT_RELATIONSHIP,
            sep="\t",
        )
        R = R[R.relationship_id == "Maps to"]
        d = create_dict(concepts, R, vocabularies=vocabularies)
    print(len(main_table))
    print(main_table.head())

    print("indivs only in OMOP endpoints:")
    from_omop_only = list(set(from_omop) - set(from_icd))
    for i in range(n_examples):
        iid = from_omop_only[i]
        print(f"Individual: {iid}")
        conditions = main_table[main_table.eid == iid].concept_name.unique()
        keys = [any(key in c.lower() for key in keywords) for c in conditions]
        if any(keys):
            print("  Has keywords: ", conditions[keys])
        else:
            print("  No keywords: ")
            mt = main_table[main_table.eid == iid][
                ["condition_concept_id", "concept_name"]
            ].drop_duplicates()
            for row in mt.itertuples():
                icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
                print(f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}")

        print()
        print()
        print("=======" * 5)

    print()
    print()
    print()
    print("**********" * 5)
    print("**********" * 5)
    print("**********" * 5)
    print()
    print()
    print()
    print("indivs only in ICD endpoints:")
    from_icd_only = list(set(from_icd) - set(from_omop))
    for i in range(n_examples):
        iid = from_icd_only[i]
        print(f"Individual: {iid}")
        conditions = main_table[main_table.eid == iid].concept_name.unique()
        keys = [any(key in c.lower() for key in keywords) for c in conditions]
        if any(keys):
            print("  Has keywords: ", conditions[keys])
        else:
            print("  No keywords: ")
            mt = main_table[main_table.eid == iid][
                ["condition_concept_id", "concept_name"]
            ].drop_duplicates()
            for row in mt.itertuples():
                icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
                print(f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}")

            print("  raw icd codes:")
            all_codes = raw_icd.loc[iid]
            print(all_codes)
            keyd = all_codes[
                np.any(
                    [
                        all_codes.CODE1.str.lower().str.contains(icd_key)
                        for icd_key in icd_keys
                    ],
                    axis=0,
                )
            ]
            print(keyd)

        print()
        print()
        print("=======" * 5)


def create_dict(concept, relationships, vocabularies=["ICD10", "ICD10CM"]):
    concept = concept[concept.vocabulary_id.isin(vocabularies)]
    available_from_concepts = set(concept.concept_id)

    def parse_code(code):
        if not "." in code:
            return code.replace(".", "")
        x1, x2 = code.split(".")
        if x2.isdigit():
            return code.replace(".", "")
        code = x1
        for c in x2:
            if c.isdigit():
                code += c
            else:
                return code

    code_from_concept = {
        id: "10x" + parse_code(code)
        for id, code in zip(concept.concept_id, concept.concept_code)
    }
    # code_from_concept = dict(zip(concept.concept_id, concept.concept_code))
    relationships = relationships[relationships.relationship_id == "Maps to"]
    relationships = relationships[
        relationships.concept_id_1.isin(available_from_concepts)
    ]
    d = defaultdict(list)
    for row in relationships.itertuples():
        d[row.concept_id_2].append(code_from_concept[row.concept_id_1])
    d = defaultdict(list, {k: list(set(v)) for k, v in d.items()})
    return d


ICD10_CODE = 41270
ICD10_DATE = 41280

ICD9_CODE = 41271
ICD9_DATE = 41281


def load_icd_codes_long(
    pheno_file=paths.UKB_PHENO_FILE,
    debug_nrows=None,
    icd9=True,
    icd10=True,
    code_name="CODE1",
    date_name="DATE",
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
    if icd9 and icd10:
        icds_long = pd.concat([long_icd9, long_icd10], axis=0)
    elif icd9:
        icds_long = long_icd9
    elif icd10:
        icds_long = long_icd10
    # icds_long = pd.concat([long_icd9, long_icd10], axis=0)

    return icds_long


##### manual code removal

from io import StringIO

REMOVALS = {
    "T2D": pd.read_csv(
        StringIO(
            """
omop_id,count,concept_name
134398,309,Periodontal disease
443735,274,Coma due to diabetes mellitus
4042502,179,Disease of mouth
442793,125,Complication due to diabetes mellitus
380097,67,Macular edema due to diabetes mellitus
376114,25,Severe nonproliferative retinopathy due to diabetes mellitus
4131908,11,Peripheral angiopathy due to diabetes mellitus
4009303,5,Diabetic ketoacidosis without coma"""
        )
    ),
    "J10_ASTHMA": pd.read_csv(
        StringIO(
            """
omop_id,count,concept_name
"""
        )
    ),
    # TODO: this seems more tricky --> need to look at the codes in detail; icd10cm definitely has some false positives, such as angina pectoris
    "I9_CHD": pd.read_csv(
        StringIO(
            """
omop_id,count,concept_name
"""
        )
    ),
    "F5_DEPRESSIO": pd.read_csv(
        StringIO(
            """
omop_id,count,concept_name
"""))
}
