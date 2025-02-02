from os.path import join
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd
from snakemake.utils import Paramspace

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.data.prefetch_set_data import (
    prefetch_source_table,
    prefetch_ingredient_table,
    prefetch_extra_table,
)

from tte.utils.misc import load_vars, load_paths

consts = load_vars()
paths = load_paths()

# EMPTY_CONCEPTS = load_vars().EMPTY_CONCEPTS
EMPTY_CONCEPTS = {
    source: []
    for source in [
        "condition_occurrence",
        "procedure_occurrence",
        "ingredient_exposure",
        "drug_exposure",
    ]
}


embed_params = pd.DataFrame(
    [
        {
            "provider": "openai_new",
            "model": "none",
            "input_type": "none",
            "pca_dim": 0,
        },
        {
            "provider": "openai",
            "model": "none",
            "input_type": "none",
            "pca_dim": 0,
        },
    ]
)
embed_params = embed_params[sorted(embed_params.columns)]
embed_paramspace = Paramspace(embed_params)


def raw_embeddings_fn(wildcards):
    provider, model, input_type, source = (
        wildcards.provider,
        wildcards.model,
        wildcards.input_type,
        wildcards.source,
    )
    if provider == "openai":
        p = paths.OPENAI_EMBEDDINGS.format(source=source)
    elif provider == "openai_onto":
        p = paths.OPENAI_ONTO_EMBEDDINGS.format(source=source)
    elif provider == "cohere":
        p = paths.COHERE_EMBEDDINGS.format(
            source=source, model=model, input_type=input_type
        )
    elif provider == "voyage-large-2-instruct":
        p = paths.VOYAGE_EMBEDDINGS.format(
            prefix=input_type, model="voyage-large-2-instruct", source=source
        )
    elif provider == "rand":
        # raise Exception
        p = []
    elif provider == "onehot":
        p = []
    else:
        # p = "XXX"
        # p = []
        # print("\n\n\n")
        # print(f"provider {provider} not supported")
        raise
        # print("\n\n\n")
    return p


rule all_process_set_data:
    input:
        expand(
            join(
                paths.PROCESSED_SET_DATA_DIR,
                "{source}",
                "records.h5",
            ),
            source=consts.SOURCES,
        ),


rule all_filter_for_active_embeddings:
    input:
        expand(
            join(
                paths.PROCESSED_SET_DATA_DIR,
                "{embedding_pattern}",
                "{source}_active_embeddings.h5",
            ),
            embedding_pattern=embed_paramspace.instance_patterns,
            source=consts.SOURCES,
        ),
        # expand(
        #     join(
        #         paths.PROCESSED_SET_DATA_DIR,
        #         rand_embed_subdir,
        #         "{source}_active_embeddings.h5",
        #     ),
        #     source=consts.SOURCES,
        # ),
        # expand(
        #     join(
        #         paths.PROCESSED_SET_DATA_DIR,
        #         onehot_embed_subdir,
        #         "{source}_active_embeddings.h5",
        #     ),
        #     source=consts.SOURCES,
        # ),


rule all:
    input:
        rules.all_process_set_data.input,
        rules.all_filter_for_active_embeddings.input,


rule all_embeddings_filter_rand:
    input:
        [
            join(
                paths.PROCESSED_SET_DATA_DIR,
                embed_paramspace.wildcard_pattern.format(
                    provider="rand",
                    model="none",
                    input_type="none",
                    pca_dim=0,
                ),
                f"{source}_active_embeddings.h5",
            )
            for source in ["condition_occurrence"]
        ],


# rule filter_for_active_embeddings_rand:
#     input:
#         active_embeddings=join(
#             paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
#         ),
#         raw_embeddings=raw_embeddings_fn,
#     output:
#         join(
#             paths.PROCESSED_SET_DATA_DIR,
#             rand_embed_subdir,
#             "{source}_active_embeddings.h5",
#         ),
#     params:
#         extra_concepts=EMPTY_CONCEPTS,
#         rand_dist="stdnorm",
#         emb_dim=1536,
#     run:
#         print("reading active")
#         active = set(pd.read_csv(input.active_embeddings, header=None).squeeze().values)
#         active.update(params.extra_concepts[wildcards.source])
#         print("sampling random embeddings")
#         if params.rand_dist == "stdnorm":
#             rand_embeddings = np.random.normal(
#                 size=(len(active), params.emb_dim)
#             ).astype(np.float32)
#         else:
#             raise NotImplementedError("only stdnorm supported atm")

#         print("dumping active embeddings to h5")
#         with h5py.File(output[0], "w") as store:
#             for i, concept_name in enumerate(active):
#                 concept_name = concept_name.replace("/", "_")
#                 store.create_dataset(
#                     concept_name, data=rand_embeddings[i].astype(np.float32)
#                 )


# rule filter_for_active_embeddings_onehot:
#     input:
#         records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
#     output:
#         join(
#             paths.PROCESSED_SET_DATA_DIR,
#             onehot_embed_subdir,
#             "{source}_active_embeddings.h5",
#         ),
#     params:
#         extra_concepts=EMPTY_CONCEPTS,
#         max_concepts_total=1536,
#         n_sources=2,
#         unique_per_person=True,
#         sources=consts.SOURCES,
#     run:
#         records = h5py.File(input.records, "r")
#         counts = dict()
#         for iid in tqdm(records):
#             concepts = [x.decode().replace("/", "_") for x, _, _ in records[iid][()]]
#             if params.unique_per_person:
#                 concepts = set(concepts)

#             for concept in concepts:
#                 if concept in counts:
#                     counts[concept] += 1
#                 else:
#                     counts[concept] = 1
#         head = params.max_concepts_total // params.n_sources
#         extra = params.extra_concepts[wildcards.source]
#         head = head - len(extra)
#         counts = pd.Series(counts).sort_values(ascending=False).head(head)
#         counts = pd.concat([counts, pd.Series(0, index=extra)])

#         index_of_concept = params.sources.index(wildcards.source)

#         print("dumping active embeddings to h5")
#         with h5py.File(output[0], "w") as store:
#             for i, concept_name in enumerate(counts.index):
#                 concept_name = concept_name.replace("/", "_")
#                 embedding = np.zeros(params.max_concepts_total, dtype=np.float32)
#                 embedding[i + index_of_concept * head] = 1
#                 store.create_dataset(concept_name, data=embedding)


# rule tmptmpextra_filter_for_active:
#     input:
#         join(
#             paths.PROCESSED_SET_DATA_DIR,
#             "input_type~none",
#             "model~none",
#             "pca_dim~0",
#             "provider~openai_new",
#             "procedure_occurrence_active_embeddings.h5",
#         ),
#         join(
#             paths.PROCESSED_SET_DATA_DIR,
#             "input_type~none",
#             "model~none",
#             "pca_dim~0",
#             "provider~openai_new",
#             "condition_occurrence_active_embeddings.h5",
#         ),
#         join(
#             paths.PROCESSED_SET_DATA_DIR,
#             "input_type~none",
#             "model~none",
#             "pca_dim~0",
#             "provider~openai_new",
#             "extra_active_embeddings.h5",
#         ),


rule filter_for_active_embeddings_rand:
    input:
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
        records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern.format(
                provider="rand",
                model="none",
                input_type="none",
                pca_dim=0,
            ),
            "{source}_active_embeddings.h5",
        ),
    params:
        extra_concepts=EMPTY_CONCEPTS,
        rand_dist="stdnorm",
        emb_dim=3072,
        base_seed=987,
    run:
        SOURCE_SEED_OFFSETS = {
            "condition_occurrence": 999,
            "procedure_occurrence": 888,
            "ingredient_exposure": 777,
            "drug_exposure": 666,
            "extra": 555,
        }
        seed = params.base_seed + SOURCE_SEED_OFFSETS[wildcards.source]
        print("reading active")
        active = set(pd.read_csv(input.active_embeddings, header=None).squeeze().values)
        if wildcards.source != "extra" and wildcards.source in params.extra_concepts:
            active.update(params.extra_concepts[wildcards.source])
        print("sampling random embeddings")
        if params.rand_dist == "stdnorm":
            rand_embeddings = (
                np.random.RandomState(seed)
                .normal(size=(len(active), params.emb_dim))
                .astype(np.float32)
            )
        else:
            raise NotImplementedError("only stdnorm supported atm")

        print("dumping active embeddings to h5")
        with h5py.File(output[0], "w") as store:
            for i, concept_name in enumerate(active):
                concept_name = str(concept_name)
                concept_name = concept_name.replace("/", "_")
                store.create_dataset(
                    concept_name, data=rand_embeddings[i].astype(np.float32)
                )


rule filter_for_active_embeddings_onehot:
    input:
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
        records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern.format(
                provider="onehot", model="none", input_type="none", pca_dim=0
            ),
            "{source}_active_embeddings.h5",
        ),
    params:
        extra_concepts=EMPTY_CONCEPTS,
        rand_dist="stdnorm",
        emb_dim=1536,
        # max_concepts_total=1536,
        onehot_unique_per_person=True,
        onehot_sources=[
            "condition_occurrence",
            "procedure_occurrence",
            "ingredient_exposure",
        ],
    run:
        records = h5py.File(input.records, "r")
        counts = dict()
        for iid in tqdm(records):
            concepts = [x.decode().replace("/", "_") for x, _, _ in records[iid][()]]
            if params.onehot_unique_per_person:
                concepts = set(concepts)

            for concept in concepts:
                if concept in counts:
                    counts[concept] += 1
                else:
                    counts[concept] = 1
        n_sources = len(params.onehot_sources)
        head = params.emb_dim // n_sources
        extra = params.extra_concepts[wildcards.source]
        head = head - len(extra)
        counts = pd.Series(counts).sort_values(ascending=False).head(head)
        counts = pd.concat([counts, pd.Series(0, index=extra)])

        index_of_concept = params.onehot_sources.index(wildcards.source)

        print("dumping active embeddings to h5")
        with h5py.File(output[0], "w") as store:
            for i, concept_name in enumerate(counts.index):
                concept_name = str(concept_name)
                concept_name = concept_name.replace("/", "_")
                embedding = np.zeros(params.emb_dim, dtype=np.float32)
                embedding[i + index_of_concept * head] = 1
                store.create_dataset(concept_name, data=embedding)


rule filter_for_active_embeddings_openai:
    input:
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
        raw_embeddings=paths.OPENAI_EMBEDDINGS,
        records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern.format(
                provider="openai",
                model="{model}",
                input_type="{input_type}",
                pca_dim="{pca_dim}",
            ),
            "{source}_active_embeddings.h5",
        ),
    params:
        extra_concepts=EMPTY_CONCEPTS,
    run:
        openai_filter(input, output, wildcards, params)


rule filter_for_active_embeddings_openai_new:
    input:
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
        raw_embeddings=paths.OPENAI_NEW_EMBEDDINGS,
        records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern.format(
                provider="openai_new",
                model="{model}",
                input_type="{input_type}",
                pca_dim="{pca_dim}",
            ),
            "{source}_active_embeddings.h5",
        ),
    params:
        extra_concepts=EMPTY_CONCEPTS,
    run:
        openai_filter(input, output, wildcards, params)


rule filter_for_active_embeddings_openai_onto:
    input:
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
        raw_embeddings=paths.OPENAI_ONTO_EMBEDDINGS,
        records=join(paths.PROCESSED_SET_DATA_DIR, "{source}", "records.h5"),
    output:
        join(
            paths.PROCESSED_SET_DATA_DIR,
            embed_paramspace.wildcard_pattern.format(
                provider="openai_onto",
                model="{model}",
                input_type="{input_type}",
                pca_dim="{pca_dim}",
            ),
            "{source}_active_embeddings.h5",
        ),
    params:
        extra_concepts=EMPTY_CONCEPTS,
    run:
        openai_filter(input, output, wildcards, params)


def openai_filter(input, output, wildcards, params):
    print("reading active")
    active = set(pd.read_csv(input.active_embeddings, header=None).squeeze().values)
    print("sniffing names of available embeddings")
    all_embeddings = pd.read_csv(input.raw_embeddings, usecols=[0])

    if wildcards.source != "extra" and wildcards.source in params.extra_concepts:
        active = active.union(params.extra_concepts[wildcards.source])
    userows = [0] + list(
        all_embeddings.index[all_embeddings.isin(active).values.flatten()] + 1
    )
    userows = set(userows)
    skiprows = [x for x in range(len(all_embeddings)) if x not in userows]
    print(f"using {len(userows)} rows, skipping {len(skiprows)} rows")
    print("reading full embeddings")
    filtered_embeddings = pd.read_csv(
        input.raw_embeddings, skiprows=skiprows, index_col=0
    )
    print("dumping active embeddings to h5")
    with h5py.File(output[0], "w") as store:
        for concept_name, values in tqdm(
            filtered_embeddings.iterrows(), total=len(filtered_embeddings)
        ):
            # if not concept_name in stored_concepts:
            concept_name = str(concept_name)
            concept_name = concept_name.replace("/", "_")
            if not concept_name in store:
                store.create_dataset(
                    concept_name, data=values.values.astype(np.float32)
                )


rule process_extra_set_data:
    input:
        meta=paths.META_PREPROCESSED,
    output:
        out_h5fn=join(
            paths.PROCESSED_SET_DATA_DIR,
            "extra",
            "records.h5",
        ),
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR,
            "extra",
            "active_embeddings.csv",
        ),
    params:
        debug_nrows=None,
    run:
        meta = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "sex_male", "age_at_recruitment"],
            nrows=params.debug_nrows,
        )
        mean, std = meta.age_at_recruitment.mean(), meta.age_at_recruitment.std()
        meta["age_at_baseline"] = (meta.age_at_recruitment - mean) / std
        active_embeddings = prefetch_extra_table(meta, output.out_h5fn)
        pd.Series(sorted(active_embeddings)).to_csv(
            output.active_embeddings, index=False, header=False
        )

        # not necessary - already done by process_extra_table in omop_processing.smk
        # metadata = {"age_at_baseline": {"mean": mean, "std": std}}
        # with open(output[1], "wb") as f:
        #     pickle.dump(metadata, f)



rule process_ingredient_set_data:
    input:
        meta=paths.META_PREPROCESSED,
        main_table=join(paths.OMOP_PATH, "omop_drug_exposure.txt"),
        drug_strength_path=paths.DRUG_STRENGTH,
    output:
        out_h5fn=join(
            paths.PROCESSED_SET_DATA_DIR,
            "ingredient_exposure",
            "records.h5",
        ),
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR,
            "ingredient_exposure",
            "active_embeddings.csv",
        ),
    params:
        debug_nrows=None,
        # remove_baseline_date_data=True,
        remove_baseline_date_data=consts.REMOVE_BASELINE_DATE_DATA,
        max_exposure_days=consts.MAX_EXPOSURE_DAYS,
        drop_missing_ingredients=False,
        remove_invalid_concepts=True,
    run:
        main_table = pd.read_csv(input.main_table, nrows=params.debug_nrows, sep="\t")
        meta = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "first_assessment_date", "birth_date"],
            parse_dates=["first_assessment_date", "birth_date"],
            dayfirst=False,
        ).squeeze()
        active_embeddings = prefetch_ingredient_table(
            main_table=main_table,
            baseline_dates=meta.first_assessment_date,
            birth_dates=meta.birth_date,
            save_h5fn=output.out_h5fn,
            remove_baseline_date_data=params.remove_baseline_date_data,
            max_exposure_days=params.max_exposure_days,
            drop_missing_ingredients=params.drop_missing_ingredients,
            drug_strength_path=input.drug_strength_path,
            remove_invalid_concepts=params.remove_invalid_concepts,
        )
        pd.Series(sorted(active_embeddings)).to_csv(
            output.active_embeddings, index=False, header=False
        )


rule process_set_data:
    input:
        meta=paths.META_PREPROCESSED,
        main_table=join(paths.OMOP_PATH, "omop_{source}.txt"),
    output:
        out_h5fn=join(
            paths.PROCESSED_SET_DATA_DIR,
            "{source}",
            "records.h5",
        ),
        active_embeddings=join(
            paths.PROCESSED_SET_DATA_DIR, "{source}", "active_embeddings.csv"
        ),
    params:
        debug_nrows=None,
        # remove_baseline_date_data=True,
        remove_baseline_date_data=consts.REMOVE_BASELINE_DATE_DATA,
        max_exposure_days=consts.MAX_EXPOSURE_DAYS,
        remove_invalid_concepts=True,
    run:
        main_table = pd.read_csv(input.main_table, nrows=params.debug_nrows, sep="\t")
        meta = pd.read_csv(
            input.meta,
            index_col=0,
            usecols=["eid", "first_assessment_date", "birth_date"],
            parse_dates=["first_assessment_date", "birth_date"],
            dayfirst=False,
        ).squeeze()
        active_embeddings = prefetch_source_table(
            source=wildcards.source,
            main_table=main_table,
            baseline_dates=meta.first_assessment_date,
            birth_dates=meta.birth_date,
            save_h5fn=output.out_h5fn,
            remove_baseline_date_data=params.remove_baseline_date_data,
            max_exposure_days=params.max_exposure_days,
            remove_invalid_concepts=params.remove_invalid_concepts,
        )
        pd.Series(sorted(active_embeddings)).to_csv(
            output.active_embeddings, index=False, header=False
        )


# ruleorder: process_ingredient_set_data > process_set_data
