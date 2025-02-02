from pathlib import Path
import pickle
import numpy as np
import sys
import pandas as pd
from collections import defaultdict
from os.path import join
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.endpoint_processing import load_icd_codes_long
from tte.utils.misc import load_paths
from tte.data.endpoint_from_omop_utils import (
    create_dict,
    load_prelim,
    compare_endpoints,
)

DEBUG_NROWS = None

paths = load_paths()


rule extract_direct_origin_codes:
    input:
        main_table=join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
        concepts=paths.CONCEPTS,
        ukb_definitions=paths.ENDPOINT_DEFINITIONS,
    output:
        "data/endpoints/endpoint_selection/origin_codes/{endpoint}.pkl",
    params:
        debug_nrows=None,
    run:
        definitions = pd.read_csv(input.ukb_definitions, index_col=0)
        regex = definitions[definitions.NAME == wildcards.endpoint].COD_ICD_10.iloc[0]
        if not isinstance(regex, str):
            print(f"Endpoint {wildcards.endpoint} has no direct ICD10 codes")
            pickle.dump(
                {"nocm": [], "wcm": []},
                open(output[0], "wb"),
            )
        else:
            R = pd.read_csv(
                input.concept_relationship,
                sep="\t",
            )
            R = R[R.relationship_id == "Maps to"]
            concept = pd.read_csv(
                input.concepts,
                sep="\t",
            )
            d_nocm = create_dict(concept, R, vocabularies=["ICD10"])
            d_wcm = create_dict(concept, R, vocabularies=["ICD10", "ICD10CM"])
            main_table = pd.read_csv(
                input.main_table,
                nrows=params.debug_nrows,
                sep="\t",
                parse_dates=["condition_start_date"],
            )
            all_endpoints_nocm = []
            all_endpoints_wcm = []
            print(main_table)
            for row in tqdm(main_table.itertuples(), total=len(main_table)):
                icd_codes = [
                    x.split("10x")[1] for x in d_nocm[row.condition_concept_id]
                ]

                if any([re.match(regex, s) for s in icd_codes]):
                    all_endpoints_nocm.append(row.condition_concept_id)

                    # icd_codes = d_wcm[row.condition_concept_id]
                icd_codes = [x.split("10x")[1] for x in d_wcm[row.condition_concept_id]]
                if any([re.match(regex, s) for s in icd_codes]):
                    all_endpoints_wcm.append(row.condition_concept_id)
            pickle.dump(
                {"nocm": all_endpoints_nocm, "wcm": all_endpoints_wcm},
                open(output[0], "wb"),
            )


ALL_ENDPOINTS = (
    pd.read_csv("external_data/endpoints/UKBB_definitions_demo_TEST.csv")
    .NAME.dropna()
    .tolist()
)


rule print_origin_codes:
    input:
        # origins="tmp{endpoint}.pkl",
        direct_origins=expand(
            "data/endpoints/endpoint_selection/origin_codes/{endpoint}.pkl",
            endpoint=ALL_ENDPOINTS,
        ),
        concepts=paths.CONCEPTS,
    output:
        nocm="data/endpoints/endpoint_selection/origin_codes/{endpoint}_nocm.csv",
        wcm="data/endpoints/endpoint_selection/origin_codes/{endpoint}_wcm.csv",
    run:
        all_d = dict()
        for endpoint in ALL_ENDPOINTS:
            fn = f"data/endpoints/endpoint_selection/origin_codes/{endpoint}.pkl"
            all_d[endpoint] = pickle.load(open(fn, "rb"))

        nocm = all_d[wildcards.endpoint]["nocm"]
        wcm = all_d[wildcards.endpoint]["wcm"]
        definitions = pd.read_csv(
            "external_data/endpoints/UKBB_definitions_demo_TEST.csv"
        )
        definition = definitions[definitions.NAME == wildcards.endpoint]

        includes = (
            []
            if not isinstance(definition.INCLUDE.iloc[0], str)
            else definition.INCLUDE.iloc[0].split("|")
        )
        extra_includes = []
        print(f"n raw: {len(nocm)}, w raw: {len(wcm)}")

        for include in includes:
            print(f"including {include}")
            nocm += all_d[include]["nocm"]
            wcm += all_d[include]["wcm"]
            sub_incl = definitions[definitions.NAME == include].INCLUDE.iloc[0]
            if isinstance(sub_incl, str):
                print(f"  also including {sub_incl}")
                sub_incl = sub_incl.split("|")
                extra_includes += sub_incl
        print(f"new n: {len(nocm)}, new w: {len(wcm)}")

        extra_includes = [incl for incl in set(extra_includes) if not incl in includes]
        for include in extra_includes:
            print(f"sub-including {include}")
            nocm += all_d[include]["nocm"]
            wcm += all_d[include]["wcm"]
            # only three levels, so not necessary to go deeper
        print(f"final n: {len(nocm)}, final w: {len(wcm)}")


        concepts = pd.read_csv(input.concepts, sep="\t")
        cdict = dict(zip(concepts.concept_id, concepts.concept_name))
        nocm = pd.DataFrame(pd.Series(nocm).value_counts().sort_values(ascending=False))
        nocm["concept_name"] = nocm.index.map(cdict)
        wcm = pd.DataFrame(pd.Series(wcm).value_counts().sort_values(ascending=False))
        wcm["concept_name"] = wcm.index.map(cdict)
        print(f"ICD10 only")
        print(nocm)
        print(f"ICD10 + ICD10CM")
        print(wcm)
        nocm.to_csv(output.nocm)
        wcm.to_csv(output.wcm)


fn = "data/endpoints/endpoint_selection/manual_endpoint_selection.txt"
# if False:
if os.path.isfile(fn):
    MANUAL_ENDPOINTS = pd.read_csv(fn, header=None).squeeze().tolist()
else:
    # print("manual endpoints not yet defined?")
    MANUAL_ENDPOINTS = ["AUD_SWEDISH"]
# print(MANUAL_ENDPOINTS)


rule extract_all_origin_codes:
    input:
        expand(
            "data/endpoints/endpoint_selection/origin_codes/{endpoint}_nocm.csv",
            endpoint=MANUAL_ENDPOINTS,
        ),


# concepts=paths.CONCEPTS
# endpoint='F5_DEPRESSIO'
# origin_codes=f"data/endpoints/endpoint_selection/origin_codes/{endpoint}_wcm.csv"
# removed_origin_codes=f"data/endpoints/endpoint_selection/removed_origin_codes_old/{endpoint}_wcm.csv"
# manual=f"data/endpoints/endpoint_selection/manual_inclusion/{endpoint}.csv"
# origin_codes = pd.read_csv(origin_codes, index_col=0)
# removed_origin_codes = pd.read_csv(removed_origin_codes, index_col=0)
# include = {x for x in origin_codes.index if x not in removed_origin_codes.index}
# concepts = pd.read_csv(concepts, sep="\t")
# ccdict = dict(zip(concepts.concept_name, concepts.concept_id))
# manual = pd.read_csv(manual, header=None)
# manual_ids = manual[0].apply(lambda x: ccdict[x])
# mids = np.unique(sum([concepts[concepts.concept_name == concept].concept_id.tolist() for concept in tqdm(manual[0].values)], []))
# include2 = include | set(mids)


rule direct_omop_to_endpoints:
    input:
        main_table=join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
        concepts=paths.CONCEPTS,
        origin_codes="data/endpoints/endpoint_selection/origin_codes/{endpoint}_wcm.csv",
        removed_origin_codes="data/endpoints/endpoint_selection/removed_origin_codes_old/{endpoint}_wcm.csv",
        manual="data/endpoints/endpoint_selection/manual_inclusion/{endpoint}.csv",
    output:
        with_cm="data/endpoints/direct_endpoints/{endpoint}.csv",
    params:
        debug_nrows=None,
    run:
        main_table = pd.read_csv(
            input.main_table,
            nrows=params.debug_nrows,
            sep="\t",
            parse_dates=["condition_start_date"],
        )
        origin_codes = pd.read_csv(input.origin_codes, index_col=0)
        removed_origin_codes = pd.read_csv(input.removed_origin_codes, index_col=0)
        include = {x for x in origin_codes.index if x not in removed_origin_codes.index}


        concepts = pd.read_csv(input.concepts, sep="\t")

        try:
            manual = pd.read_csv(input.manual, header=None)
            # ccdict = dict(zip(concepts.concept_name, concepts.concept_id))
            # manual_ids = manual[0].apply(lambda x: ccdict[x])
            manual_ids = np.unique(
                sum(
                    [
                        concepts[concepts.concept_name == concept].concept_id.tolist()
                        for concept in tqdm(manual[0].values)
                    ],
                    [],
                )
            )

        except:
            manual = pd.DataFrame()
            manual_ids = pd.DataFrame()
            # manual = pd.DataFrame([np.nan])
            # manual_ids = pd.DataFrame([''])
        include = include | set(manual_ids)

        print(
            f"Including {len(include)} codes for endpoint {wildcards.endpoint}: {include}"
        )

        cdict = dict(zip(concepts.concept_id, concepts.concept_name))

        main_table = main_table[main_table.condition_concept_id.isin(include)]
        main_table["source_concept_name"] = main_table.condition_concept_id.map(cdict)
        main_table["endpoint"] = wildcards.endpoint
        main_table.rename(
            {
                "condition_start_date": "date",
                "condition_concept_id": "source_concept_id",
            },
            axis=1,
            inplace=True,
        )
        main_table[
            [
                "eid",
                "date",
                "source_concept_id",
                # "source_concept_name",
                # "endpoint",
            ]
        ].drop_duplicates().to_csv(output.with_cm, index=False)


rule create_t2d_control_exclusions:
    # special exclusions for T2D only --> collect, but do nothing
    input:
        main_table=join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
        concepts=paths.CONCEPTS,
        exclusions="data/endpoints/endpoint_selection/manual_exclusions/T2D.csv",
    output:
        "data/endpoints/direct_endpoints/control_exclusions_t2d.csv",
    params:
        debug_nrows=None,
    run:
        main_table = pd.read_csv(
            input.main_table,
            nrows=params.debug_nrows,
            sep="\t",
            parse_dates=["condition_start_date"],
        )
        exclusions = pd.read_csv(input.exclusions, header=None)

        concepts = pd.read_csv(input.concepts, sep="\t")
        # ccdict = dict(zip(concepts.concept_name, concepts.concept_id))
        # manual_ids = exclusions[0].apply(lambda x: ccdict[x])
        manual_ids = np.unique(
            sum(
                [
                    concepts[concepts.concept_name == concept].concept_id.tolist()
                    for concept in tqdm(exclusions[0].values)
                ],
                [],
            )
        )
        include = set(manual_ids)

        print(f"Including {len(include)} codes for T2D exclusions")

        cdict = dict(zip(concepts.concept_id, concepts.concept_name))

        main_table = main_table[main_table.condition_concept_id.isin(include)]
        main_table["source_concept_name"] = main_table.condition_concept_id.map(cdict)
        main_table["endpoint"] = "T2D"
        main_table.rename(
            {
                "condition_start_date": "date",
                "condition_concept_id": "source_concept_id",
            },
            axis=1,
            inplace=True,
        )
        main_table[
            [
                "eid",
                "date",
                "source_concept_id",
                # "source_concept_name",
                # "endpoint",
            ]
        ].drop_duplicates().to_csv(output[0], index=False)


rule aggregate_direct_omop_to_endpoints:
    input:
        expand(
            "data/endpoints/direct_endpoints/{endpoint}.csv",
            endpoint=MANUAL_ENDPOINTS,
        ),
        t2d_control="data/endpoints/direct_endpoints/control_exclusions_t2d.csv",
    output:
        "data/endpoints/endpoints_direct_from_omop.csv",
    params:
        endpoints=MANUAL_ENDPOINTS,
    run:
        dfs = []
        for endpoint in params.endpoints:
            fn = f"data/endpoints/direct_endpoints/{endpoint}.csv"
            sub_df = pd.read_csv(fn)
            sub_df["endpoint"] = endpoint
            dfs.append(sub_df)
        df = pd.concat(dfs)
        df.to_csv(output[0], index=False)


# deprecated
"""
rule extract_codes:
    input:
        main_table=join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
        concepts=paths.CONCEPTS,
    output:
        no_cm=paths.UKB_ICD_FROM_OMOP_LONGFORMAT_NOCM,
        with_cm=paths.UKB_ICD_FROM_OMOP_LONGFORMAT_WCM,
    params:
        debug_nrows=DEBUG_NROWS,
    run:
        R = pd.read_csv(
            input.concept_relationship,
            sep="\t",
        )
        R = R[R.relationship_id == "Maps to"]
        concept = pd.read_csv(
            input.concepts,
            sep="\t",
        )
        d_nocm = create_dict(concept, R, vocabularies=["ICD10"])
        d_wcm = create_dict(concept, R, vocabularies=["ICD10", "ICD10CM"])

        main_table = pd.read_csv(
            input.main_table,
            nrows=params.debug_nrows,
            sep="\t",
            parse_dates=["condition_start_date"],
        )
        all_endpoints_nocm = []
        all_endpoints_wcm = []
        print(main_table)
        for row in tqdm(main_table.itertuples(), total=len(main_table)):
            icd_codes = d_nocm[row.condition_concept_id]
            for code in icd_codes:
                all_endpoints_nocm.append(
                    {"eid": row.eid, "CODE1": code, "DATE": row.condition_start_date}
                )
            icd_codes = d_wcm[row.condition_concept_id]
            for code in icd_codes:
                all_endpoints_wcm.append(
                    {"eid": row.eid, "CODE1": code, "DATE": row.condition_start_date}
                )

        all_endpoints_nocm = pd.DataFrame(all_endpoints_nocm).drop_duplicates()
        all_endpoints_nocm.to_csv(output.no_cm, index=False)
        all_endpoints_wcm = pd.DataFrame(all_endpoints_wcm).drop_duplicates()
        all_endpoints_wcm.to_csv(output.with_cm, index=False)
"""


rule all:
    input:
        rules.aggregate_direct_omop_to_endpoints.output,
    default_target: True


# def create_dict(concept, relationships, vocabularies=["ICD10", "ICD10CM"]):
#     concept = concept[concept.vocabulary_id.isin(vocabularies)]
#     available_from_concepts = set(concept.concept_id)

#     def parse_code(code):
#         if not "." in code:
#             return code.replace(".", "")
#         x1, x2 = code.split(".")
#         if x2.isdigit():
#             return code.replace(".", "")
#         code = x1
#         for c in x2:
#             if c.isdigit():
#                 code += c
#             else:
#                 return code

#     code_from_concept = {
#         id: "10x" + parse_code(code)
#         for id, code in zip(concept.concept_id, concept.concept_code)
#     }
#     # code_from_concept = dict(zip(concept.concept_id, concept.concept_code))
#     relationships = relationships[relationships.relationship_id == "Maps to"]
#     relationships = relationships[
#         relationships.concept_id_1.isin(available_from_concepts)
#     ]
#     d = defaultdict(list)
#     for row in relationships.itertuples():
#         d[row.concept_id_2].append(code_from_concept[row.concept_id_1])
#     d = defaultdict(list, {k: list(set(v)) for k, v in d.items()})
#     return d


# deprecated
# rule:
#     input:
#         expand("tmp_{cm}.ttt", cm=["nocm", "wcm"]),

# rule generate_ukb_phenotypes_from_omop:
#     input:
#         ukb_definitions=paths.ENDPOINT_DEFINITIONS,
#         ukb_longformat=paths.UKB_ICD_FROM_OMOP_LONGFORMAT,
#     output:
#         ukb_endpoints=paths.UKB_ENDPOINTS_FROM_OMOP_INTERM,
#     params:
#         first_date_only=0,
#     conda:
#         "r_env"
#     shell:
#         r"""
#         Rscript external_scripts/UKBBPhenotyper.R \
#             --ukb_definitions {input.ukb_definitions} \
#             --ukb_longitudinal {input.ukb_longformat} \
#             --first_date_only {params.first_date_only} \
#             --output {output.ukb_endpoints}
#         """


# deprecated
# rule slacken_ukb_phenotypes_from_omop:
#     input:
#         ukb_endpoints=paths.UKB_ENDPOINTS_FROM_OMOP_INTERM,
#         flagship_endpoints=paths.FLAGSHIP_ENDPOINTS,
#     output:
#         ukb_endpoints=paths.UKB_ENDPOINTS_FROM_OMOP,
#     params:
#         # use_levels=[1],
#         use_levels=[1, 2, 3],
#         filter_flagship=False,
#     run:
#         df = pd.read_csv(
#             input.ukb_endpoints,
#             # index_col=0,
#             # usecols=["eid", "DATE", "ENDPOINT_LEVEL_1"],
#         )
#         df.rename(columns={"DATE": "date"}, inplace=True)
#         if params.filter_flagship:
#             flagship_endpoints = set(
#                 pd.read_csv(input.flagship_endpoints)["FinnGen endpoint"]
#             )
#         levels = []
#         for level in params.use_levels:
#             if params.filter_flagship:
#                 dff = df[df[f"ENDPOINT_LEVEL_{level}"].isin(flagship_endpoints)].copy()
#             else:
#                 dff = df.dropna(subset=[f"ENDPOINT_LEVEL_{level}"]).copy()
#             dff.rename(columns={f"ENDPOINT_LEVEL_{level}": "endpoint"}, inplace=True)
#             levels.append(dff[["eid", "date", "endpoint"]])
#         df = pd.concat(levels)
#         df.drop_duplicates(inplace=True)
#         df.to_csv(output.ukb_endpoints, index=False)


# deprecated
"""
rule all:
    input:
        expand(paths.UKB_ENDPOINTS_FROM_OMOP, cm=["nocm", "wcm"]),


# deprecated
rule compare_against_original_endpoints:
    input:
        ukb_endpoints_from_omop=paths.UKB_ENDPOINTS_FROM_OMOP,
        ukb_endpoints_from_icd=paths.UKB_ENDPOINTS,
    output:
        "tmp_{cm}.ttt",
    run:
        from_omop = pd.read_csv(input.ukb_endpoints_from_omop)
        from_icd = pd.read_csv(input.ukb_endpoints_from_icd)
        endpoints = set(from_icd.endpoint)
        cols = ["eid", "date"]
        for endpoint in endpoints:
            print(f"Endpoint: {endpoint}")
            omop = (
                from_omop[from_omop.endpoint == endpoint][cols].drop_duplicates().copy()
            )
            icd = from_icd[from_icd.endpoint == endpoint][cols].drop_duplicates().copy()
            print("events:")
            print(f"  OMOP: {len(omop)}; ICD: {len(icd)}")
            merged_left = omop.merge(icd, on=cols, how="left", indicator=True)
            merged_right = omop.merge(icd, on=cols, how="right", indicator=True)
            counts = dict(zip(*np.unique(merged_left._merge, return_counts=True)))
            counts["right_only"] = merged_right._merge.value_counts()["right_only"]

            for k, v in counts.items():
                n = {"left_only": "OMOP", "both": "BOTH", "right_only": "ICD"}[k]
                print(f"    {n}: {v}")

            print("individuals:")
            omop = omop.eid.unique()
            icd = icd.eid.unique()
            print(f"  OMOP: {len(omop)}; ICD: {len(icd)}")
            omop_only = len(set(omop) - set(icd))
            icd_only = len(set(icd) - set(omop))
            both = len(set(omop) & set(icd))
            print(f"    OMOP: {omop_only}; ICD: {icd_only}; BOTH: {both}")
            print("----" * 5)
        Path(output[0]).touch()


# deprecated
rule compare_t2d:
    input:
        ukb_endpoints_from_omop=paths.UKB_ENDPOINTS_FROM_OMOP,
        ukb_endpoints_from_icd=paths.UKB_ENDPOINTS,
        omop=join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
        concepts=paths.CONCEPTS,
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
    output:
        "tmp2.ttt",
    params:
        debug_nrows=1000,
        n_examples=5,
        vocabularies=["ICD10", "ICD10CM"],
    run:
        from_omop = pd.read_csv(input.ukb_endpoints_from_omop)
        from_icd = pd.read_csv(input.ukb_endpoints_from_icd)
        from_omop = from_omop[from_omop.endpoint == "T2D"].eid.unique()
        from_icd = from_icd[from_icd.endpoint == "T2D"].eid.unique()
        print(f"OMOP: {len(from_omop)}; ICD: {len(from_icd)}")
        omop_only = len(set(from_omop) - set(from_icd))
        icd_only = len(set(from_icd) - set(from_omop))
        both = len(set(from_omop) & set(from_icd))
        total = len(set(from_omop) | set(from_icd))
        all_iids = set(from_omop) | set(from_icd)
        print(f"OMOP: {omop_only}; ICD: {icd_only}; BOTH: {both}, TOTAL: {total}")
        print("reading omop...")
        main_table = pd.read_csv(
            input.omop,
            nrows=params.debug_nrows,
            sep="\t",
            parse_dates=["condition_start_date"],
        )
        print("reading concepts...")
        concepts = pd.read_csv(
            input.concepts,
            sep="\t",
            usecols=["concept_id", "concept_name"],
        )
        print("filtering...")
        main_table = main_table[main_table.eid.isin(all_iids)]
        main_table["concept_name"] = main_table.condition_concept_id.map(
            dict(zip(concepts.concept_id, concepts.concept_name))
        )
        print("reading relationships...")
        R = pd.read_csv(
            input.concept_relationship,
            sep="\t",
        )
        R = R[R.relationship_id == "Maps to"]
        d = create_dict(concepts, R, vocabularies=params.vocabularies)
        print(len(main_table))
        print(main_table.head())

        print("indivs only in OMOP endpoints:")
        from_omop_only = list(set(from_omop) - set(from_icd))
        for i in range(params.n_examples):
            iid = from_omop_only[i]
            print(f"Individual: {iid}")
            conditions = main_table[main_table.eid == iid].concept_name.unique()
            diabs = ["diabetes" in c.lower() for c in conditions]
            if any(diabs):
                print("  Has diabetes: ", conditions[diabs])
            else:
                print("  No diabetes, conditions: ", conditions)
                mt = main_table[main_table.eid == iid][
                    ["condition_concept_id", "concept_name"]
                ].drop_duplicates()
                for row in mt.itertuples():
                    icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
                    print(
                        f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}"
                    )

            print()
            print()
            print("=======" * 5)

        print("indivs only in ICD endpoints:")
        from_icd_only = list(set(from_icd) - set(from_omop))
        for i in range(params.n_examples):
            iid = from_icd_only[i]
            print(f"Individual: {iid}")
            conditions = main_table[main_table.eid == iid].concept_name.unique()
            diabs = ["diabetes" in c.lower() for c in conditions]
            if any(diabs):
                print("  Has diabetes: ", conditions[diabs])
            else:
                print("  No diabetes, conditions: ", conditions)
                mt = main_table[main_table.eid == iid][
                    ["condition_concept_id", "concept_name"]
                ].drop_duplicates()
                for row in mt.itertuples():
                    icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
                    print(
                        f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}"
                    )

            print()
            print()
            print("=======" * 5)

        Path(output[0]).touch()
"""
# def load_prelim():
#     ukb_endpoints_from_omop = paths.UKB_ENDPOINTS_FROM_OMOP
#     ukb_endpoints_from_icd = paths.UKB_ENDPOINTS
#     omop = join(paths.OMOP_PATH, "omop_condition_occurrence.txt")
#     concepts = paths.CONCEPTS
#     concept_relationship = paths.CONCEPT_RELATIONSHIP
#     print("reading endpoints...")
#     from_omop = pd.read_csv(ukb_endpoints_from_omop)
#     from_icd = pd.read_csv(ukb_endpoints_from_icd)
#     print("reading omop...")
#     main_table = pd.read_csv(
#         omop,
#         nrows=params.debug_nrows,
#         sep="\t",
#         parse_dates=["condition_start_date"],
#     )
#     print("reading concepts...")
#     concepts = pd.read_csv(
#         concepts,
#         sep="\t",
#         usecols=["concept_id", "concept_name"],
#     )
#     print("reading relationships...")
#     R = pd.read_csv(
#         concept_relationship,
#         sep="\t",
#     )
#     R = R[R.relationship_id == "Maps to"]
#     d = create_dict(concepts, R, vocabularies=vocabularies)
#     return from_omop, from_icd, main_table, concepts, d
# from_omop, from_icd, main_table, concepts, d = load_prelim()
# def compare_endpoints(
#     from_omop,
#     from_icd,
#     endpoint,
#     main_table,
#     n_examples=25,
#     vocabularies=["ICD10", "ICD10CM"],
#     keywords=[],
#     concepts=None,
#     d=None,
#     paths=None,
# ):
#     if paths is None:
#         from tte.utils.misc import load_paths
#         paths = load_paths()
#     from_omop = from_omop[from_omop.endpoint == endpoint].eid.unique()
#     from_icd = from_icd[from_icd.endpoint == endpoint].eid.unique()
#     print(f"OMOP: {len(from_omop)}; ICD: {len(from_icd)}")
#     omop_only = len(set(from_omop) - set(from_icd))
#     icd_only = len(set(from_icd) - set(from_omop))
#     both = len(set(from_omop) & set(from_icd))
#     total = len(set(from_omop) | set(from_icd))
#     all_iids = set(from_omop) | set(from_icd)
#     print(f"OMOP: {omop_only}; ICD: {icd_only}; BOTH: {both}, TOTAL: {total}")
#     if main_table is None:
#         print("reading omop...")
#         main_table = pd.read_csv(
#             join(paths.OMOP_PATH, "omop_condition_occurrence.txt"),
#             nrows=None,
#             sep="\t",
#             parse_dates=["condition_start_date"],
#         )
#     if concepts is None:
#         concepts = pd.read_csv(
#             paths.CONCEPTS,
#             sep="\t",
#             usecols=["concept_id", "concept_name"],
#         )
#     print("filtering...")
#     main_table = main_table[main_table.eid.isin(all_iids)]
#     main_table["concept_name"] = main_table.condition_concept_id.map(
#         dict(zip(concepts.concept_id, concepts.concept_name))
#     )
#     if d is None:
#         print("reading relationships...")
#         R = pd.read_csv(
#             paths.CONCEPT_RELATIONSHIP,
#             sep="\t",
#         )
#         R = R[R.relationship_id == "Maps to"]
#         d = create_dict(concepts, R, vocabularies=vocabularies)
#     print(len(main_table))
#     print(main_table.head())
#     print("indivs only in OMOP endpoints:")
#     from_omop_only = list(set(from_omop) - set(from_icd))
#     for i in range(n_examples):
#         iid = from_omop_only[i]
#         print(f"Individual: {iid}")
#         conditions = main_table[main_table.eid == iid].concept_name.unique()
#         keys = [any(key in c.lower() for key in keywords) for c in conditions]
#         if any(keys):
#             print("  Has keywords: ", conditions[keys])
#         else:
#             print("  No keywords: ")
#             mt = main_table[main_table.eid == iid][
#                 ["condition_concept_id", "concept_name"]
#             ].drop_duplicates()
#             for row in mt.itertuples():
#                 icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
#                 print(f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}")
#         print()
#         print()
#         print("=======" * 5)
#     print("indivs only in ICD endpoints:")
#     from_icd_only = list(set(from_icd) - set(from_omop))
#     for i in range(n_examples):
#         iid = from_icd_only[i]
#         print(f"Individual: {iid}")
#         conditions = main_table[main_table.eid == iid].concept_name.unique()
#         keys = [any(key in c.lower() for key in keywords) for c in conditions]
#         if any(keys):
#             print("  Has keywords: ", conditions[keys])
#         else:
#             print("  No keywords: ")
#             mt = main_table[main_table.eid == iid][
#                 ["condition_concept_id", "concept_name"]
#             ].drop_duplicates()
#             for row in mt.itertuples():
#                 icd_codes = [x.split("10x")[1] for x in d[row.condition_concept_id]]
#                 print(f"  {row.concept_name} - {row.condition_concept_id}: {icd_codes}")
#         print()
#         print()
#         print("=======" * 5)
