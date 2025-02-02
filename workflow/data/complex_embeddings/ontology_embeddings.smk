import numpy as np
from os.path import join
import os
from pathlib import Path
import sys
import pandas as pd
import pickle
import math
import openai
import time

import json
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.utils.misc import (
    load_vars,
    load_paths,
)

from tte.embedding_loading.openai_new_emb import init_openai, get_embeddings

paths = load_paths()
consts = load_vars()


rule all:
    input:
        # expand(
        #     paths.OPENAI_ONTO_EMBEDDINGS_ROBUST,
        #     source=["drug_exposure"],
        # ),
        # expand(paths.OPENAI_ONTO_BATCH_FINISHED, source=["condition_occurrence"]),
        # expand(paths.OPENAI_ONTO_EMBEDDINGS, source=["condition_occurrence"]),
        expand(
            paths.OPENAI_ONTO_EMBEDDINGS,
            source=["drug_exposure"],
            #source=["procedure_occurrence", "drug_exposure"],
        ),
        # expand(paths.OPENAI_ONTO_EMBEDDINGS, source=["condition_occurrence", "extra"]),
        # paths.OPENAI_ONTO_EMBEDDINGS,


rule cp_extra:
    input:
        paths.OPENAI_NEW_EMBEDDINGS.format(source="extra"),
    output:
        paths.OPENAI_ONTO_EMBEDDINGS.format(source="extra"),
    run:
        os.system(f"cp {input[0]} {output[0]}")


# rule download_openai_embeddings_source:
#     input:
#         concepts=paths.CONCEPTS,
#         text="data/embeddings/ontology_text/{source}.csv",
#     output:
#         paths.OPENAI_ONTO_EMBEDDINGS,
#     params:
#         debug_nrows=500,
#         batch_size=256,
#         model="text-embedding-3-large",
#     run:
#         raise
#         client = init_openai()
#         text = pd.read_csv(input.text, index_col=0)
#         concepts = pd.read_csv(input.concepts, sep="\t", index_col=0)
#         text["concept_name"] = text.index.map(concepts.concept_name)

#         text = text.head(params.debug_nrows)

#         n_batches = math.ceil(len(text) / params.batch_size)
#         embeddings = []
#         for batch in tqdm(range(n_batches)):
#             embeddings.append(
#                 get_embeddings(
#                     strings=text.concept_text.iloc[
#                         batch * params.batch_size : (batch + 1) * params.batch_size
#                     ],
#                     client=client,
#                     model=params.model,
#                 )
#             )
#         embeddings = np.concatenate(embeddings, axis=0)
#         df = pd.DataFrame(
#             embeddings,
#             index=text.concept_name,
#             columns=[f"embdim_{i}" for i in range(embeddings.shape[1])],
#         )
#         df.to_csv(output[0])


rule fill_up_missing_batches:
    input:
        # meta=paths.OPENAI_ONTO_BATCH,
        finished_embeddings=paths.OPENAI_ONTO_EMBEDDINGS_ROBUST,
        missing_embeddings="data/raw_embeddings/tmp_batch/missing_embeddings_{source}.pkl",
    output:
        finished_embeddings=paths.OPENAI_ONTO_EMBEDDINGS,
    run:
        previous_embeddings = pd.read_csv(input.finished_embeddings, index_col=0)
        with open(input.missing_embeddings, "rb") as f:
            batch_metadata = pickle.load(f)
        client = init_openai()
        embeddings = []
        concepts = []
        for batch in tqdm(batch_metadata):
            embeddings.append(
                get_embeddings(
                    strings=batch["strings"],
                    client=client,
                    model="text-embedding-3-large",
                )
            )
            concepts.append(batch["concept_names"])
        embeddings = pd.DataFrame(
            np.concatenate(embeddings, axis=0),
            # np.concatenate(ee, axis=0),
            index=np.concatenate(concepts),
            columns=[f"embdim_{i}" for i in range(embeddings[0].shape[1])],
            # columns=[f"embdim_{i}" for i in range(ee[0].shape[1])],
        )
        full_embeddings = pd.concat(
            [previous_embeddings, embeddings], axis=0
        ).sort_index()
        full_embeddings.to_csv(output[0])


rule download_finished_batches_robust:
    input:
        paths.OPENAI_ONTO_BATCH,
        # paths.OPENAI_ONTO_BATCH_FINISHED,
    output:
        finished_embeddings=paths.OPENAI_ONTO_EMBEDDINGS_ROBUST,
        missing_embeddings="data/raw_embeddings/tmp_batch/missing_embeddings_{source}.pkl",
    params:
        tmp_dir="data/raw_embeddings/tmp_batch/tmpfiles/{source}/",
    run:
        with open(input[0], "rb") as f:
            batch_metadata = pickle.load(f)
        client = init_openai()
        concepts = pd.read_csv(paths.CONCEPTS, sep="\t", index_col=0)
        names = dict(zip(concepts.index, concepts.concept_name))
        all_embeddings = []
        missing_embeddings = []
        for batch in tqdm(batch_metadata):
            retval = client.batches.retrieve(batch["metadata"].id)
            if retval.status != "completed":
                missing_embeddings.append(batch)
                print(f"Batch {batch['metadata'].id} not finished yet")
                continue
            outfile_id = retval.output_file_id
            content = client.files.content(outfile_id)
            lines = [json.loads(line) for line in content.iter_lines()]
            ids = [line["custom_id"] for line in lines]
            omop_ids = [int(s.split("-")[-1]) for s in ids]

            embeddings = np.array(
                [line["response"]["body"]["data"][0]["embedding"] for line in lines]
            )

            embeddings = pd.DataFrame(
                embeddings,
                index=[names[i] for i in omop_ids],
                columns=[f"embdim_{i}" for i in range(embeddings.shape[1])],
            )
            all_embeddings.append(embeddings)
        all_embeddings = pd.concat(all_embeddings, axis=0)
        all_embeddings.to_csv(output.finished_embeddings)
        with open(output.missing_embeddings, "wb") as f:
            pickle.dump(missing_embeddings, f)


# rule download_finished_batches:
#     input:
#         paths.OPENAI_ONTO_BATCH,
#         paths.OPENAI_ONTO_BATCH_FINISHED,
#     output:
#         paths.OPENAI_ONTO_EMBEDDINGS,
#     params:
#         tmp_dir="data/raw_embeddings/tmp_batch/tmpfiles/{source}/",
#     run:
#         with open(input[0], "rb") as f:
#             batch_metadata = pickle.load(f)
#         client = init_openai()
#         concepts = pd.read_csv(paths.CONCEPTS, sep="\t", index_col=0)
#         names = dict(zip(concepts.index, concepts.concept_name))
#         all_embeddings = []
#         for batch in tqdm(batch_metadata):
#             retval = client.batches.retrieve(batch["metadata"].id)
#             outfile_id = retval.output_file_id
#             content = client.files.content(outfile_id)
#             lines = [json.loads(line) for line in content.iter_lines()]
#             ids = [line["custom_id"] for line in lines]
#             omop_ids = [int(s.split("-")[-1]) for s in ids]

#             embeddings = np.array(
#                 [line["response"]["body"]["data"][0]["embedding"] for line in lines]
#             )

#             embeddings = pd.DataFrame(
#                 embeddings,
#                 index=[names[i] for i in omop_ids],
#                 columns=[f"embdim_{i}" for i in range(embeddings.shape[1])],
#             )
#             all_embeddings.append(embeddings)
#         all_embeddings = pd.concat(all_embeddings, axis=0)
#         all_embeddings.to_csv(output[0])


rule check_openai_batch_finished:
    input:
        paths.OPENAI_ONTO_BATCH,
    output:
        paths.OPENAI_ONTO_BATCH_FINISHED,
    run:
        with open(input[0], "rb") as f:
            batch_metadata = pickle.load(f)
        client = init_openai()
        all_status = [
            client.batches.retrieve(batch["metadata"].id).status == "completed"
            for batch in batch_metadata
        ]
        print(f"Finished: {sum(all_status)} / {len(all_status)}")
        if all(all_status):
            Path(output[0]).touch()
        else:
            raise ValueError("Not all batches are finished yet")


rule purge_tmp:
    run:
        os.system("rm -r data/raw_embeddings/tmp_batch/tmpfiles/condition_occurrence")


rule start_openai_batch_embeddings:
    input:
        concepts=paths.CONCEPTS,
        text="data/embeddings/ontology_text/{source}.csv",
    output:
        paths.OPENAI_ONTO_BATCH,
    params:
        debug_nrows=None,
        batch_size=256,
        model="text-embedding-3-large",
        tmp_dir="data/raw_embeddings/tmp_batch/tmpfiles/{source}/",
    run:
        os.makedirs(params.tmp_dir, exist_ok=True)
        client = init_openai()
        text = pd.read_csv(input.text, index_col=0)
        concepts = pd.read_csv(input.concepts, sep="\t", index_col=0)
        text["concept_name"] = text.index.map(concepts.concept_name)

        text = text.head(params.debug_nrows)

        n_batches = math.ceil(len(text) / params.batch_size)
        batch_metadata = []
        for batch in tqdm(range(n_batches)):

            strings = text.concept_text.iloc[
                batch * params.batch_size : (batch + 1) * params.batch_size
            ]
            fn = join(params.tmp_dir, f"{batch}.jsonl")
            metadata = setup_batch(
                strings=strings,
                client=client,
                model=params.model,
                ids=text.index[
                    batch * params.batch_size : (batch + 1) * params.batch_size
                ],
                fn=fn,
            )
            batch_metadata.append(
                {
                    "metadata": metadata,
                    "strings": strings,
                    "concept_names": text.concept_name.iloc[
                        batch * params.batch_size : (batch + 1) * params.batch_size
                    ],
                }
            )
        with open(output[0], "wb") as f:
            pickle.dump(batch_metadata, f)


def setup_batch(strings, client, ids, fn, model="text-embedding-ada-002", **kwargs):

    to_jsonl(strings, model, ids, fn)
    batch_input_file = client.files.create(file=open(fn, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id

    count = 0
    max_repeats = 60
    while count < max_repeats:
        count += 1
        try:
            metadata = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )
            break
        except openai.RateLimitError:
            print(f"Rate limit reached, count {count}, waiting 1 minute")
            time.sleep(60)
            # metadata = client.batches.create(
            #     input_file_id=batch_input_file_id,
            #     endpoint="/v1/embeddings",
            #     completion_window="24h",
            # )
    if count == max_repeats:
        raise ValueError("Max repeats reached")
    return metadata


def to_jsonl(strings, model, ids, fn):

    body = [
        {
            "custom_id": f"{fn}-{id}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": model, "input": string, "encoding_format": "float"},
        }
        for id, string in zip(ids, strings)
    ]
    with open(fn, "w") as f:
        for line in body:
            f.write(json.dumps(line) + "\n")


ALL_EXCLUDE = pd.read_csv("data/misc/ontology_manual_exclusion.csv", header=None)[
    0
].tolist()


rule all_texts:
    input:
        expand(
            "data/embeddings/ontology_text/{source}.csv",
            source=["procedure_occurrence", "drug_exposure"],
        ),
        # expand("data/embeddings/ontology_text/{source}.csv", source=['condition_occurrence', 'procedure_occurrence', 'drug_exposure']),


rule condition_texts:
    input:
        concepts=paths.CONCEPTS,
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
        synonyms=paths.CONCEPT_SYNONYM,
        relationship=paths.RELATIONSHIP,
    output:
        "data/embeddings/ontology_text/condition_occurrence.csv",
    params:
        c2c="1to2",
        rel_filter={
            "include": None,
            "exclude": ALL_EXCLUDE,
        },
    run:
        concepts = pd.read_csv(input.concepts, sep="\t", index_col=0)
        relationships = pd.read_csv(input.concept_relationship, sep="\t")
        synonyms = pd.read_csv(input.synonyms, sep="\t")
        rel_definitions = pd.read_csv(input.relationship, sep="\t")
        rel_definitions.relationship_name = (
            rel_definitions.relationship_name.str.replace(r"\s*\(.*\)$", "", regex=True)
        )

        rel_definitions = dict(
            zip(rel_definitions.relationship_id, rel_definitions.relationship_name)
        )

        # concepts = pd.read_csv(paths.CONCEPTS, sep="\t", index_col=0)
        # relationships = pd.read_csv(paths.CONCEPT_RELATIONSHIP, sep="\t")

        conditions = concepts[
            (concepts.domain_id == "Condition")
            & (concepts.standard_concept == "S")
            & (concepts.invalid_reason.isna())
        ]
        ids = set(conditions.index)
        rels = relationships[
            (relationships.concept_id_1.isin(ids))
            | (relationships.concept_id_2.isin(ids))
        ]
        if params.rel_filter["include"]:
            rels = rels[rels.relationship_id.isin(params.rel_filter["include"])]
        if params.rel_filter["exclude"]:
            rels = rels[~rels.relationship_id.isin(params.rel_filter["exclude"])]

        synonyms = synonyms[synonyms.concept_id.isin(ids)]


        names = dict(zip(concepts.index, concepts.concept_name))

        # conditions = conditions[75_500:90000]

        data = dict()
        for row in tqdm(conditions.itertuples(), total=len(conditions)):
            S = synonyms[synonyms.concept_id == row.Index]
            if True:
                # if params.c2c == "1to2":
                R = rels[rels.concept_id_1 == row.Index]
            elif params.c2c == "2to1":
                R = rels[rels.concept_id_2 == row.Index]
            else:
                R = rels[
                    (rels.concept_id_1 == row.Index) | (rels.concept_id_2 == row.Index)
                ]

            texts = [f"{row.concept_name}."]
            for s in S.itertuples():
                texts.append(f"Synonym: {s.concept_synonym_name}.")
            for r in R.itertuples():
                rel_name = rel_definitions[r.relationship_id]
                if r.concept_id_1 == row.Index and r.concept_id_2 in names:
                    name = names[r.concept_id_2]
                elif r.concept_id_2 == row.Index and r.concept_id_1 in names:
                    name = names[r.concept_id_1]
                else:
                    print(f"missing name for {r}")
                    continue
                texts.append(f"{rel_name}: {name}.")
            text = " ".join(texts)
            data[row.Index] = text
        data = pd.Series(data, name="concept_text")
        data.to_csv(output[0], index=True)


rule procedure_texts:
    input:
        concepts=paths.CONCEPTS,
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
        synonyms=paths.CONCEPT_SYNONYM,
        relationship=paths.RELATIONSHIP,
    output:
        "data/embeddings/ontology_text/procedure_occurrence.csv",
    params:
        c2c="1to2",
        # rel_filter={"include": None, "exclude": ["Maps to", "Mapped from", "Subsumes"]},
        rel_filter={
            "include": None,
            "exclude": ALL_EXCLUDE,
        },
    run:
        concepts = pd.read_csv(input.concepts, sep="\t", index_col=0)
        relationships = pd.read_csv(input.concept_relationship, sep="\t")
        synonyms = pd.read_csv(input.synonyms, sep="\t")
        rel_definitions = pd.read_csv(input.relationship, sep="\t")

        # concepts = pd.read_csv(paths.CONCEPTS, sep="\t", index_col=0)
        # relationships = pd.read_csv(paths.CONCEPT_RELATIONSHIP, sep="\t")
        # synonyms = pd.read_csv(paths.CONCEPT_SYNONYM, sep="\t")
        # rel_definitions = pd.read_csv(paths.RELATIONSHIP, sep="\t")

        rel_definitions.relationship_name = (
            rel_definitions.relationship_name.str.replace(r"\s*\(.*\)$", "", regex=True)
        )

        rel_definitions = dict(
            zip(rel_definitions.relationship_id, rel_definitions.relationship_name)
        )

        procedures = concepts[
            (concepts.domain_id == "Procedure")
            & (concepts.standard_concept == "S")
            & (concepts.invalid_reason.isna())
        ]
        ids = set(procedures.index)
        rels = relationships[
            (relationships.concept_id_1.isin(ids))
            | (relationships.concept_id_2.isin(ids))
        ]
        if params.rel_filter["include"]:
            rels = rels[rels.relationship_id.isin(params.rel_filter["include"])]
        if params.rel_filter["exclude"]:
            # EX = ["Maps to", "Mapped from", "Subsumes"]
            # rels = rels[~rels.relationship_id.isin(EX)]
            rels = rels[~rels.relationship_id.isin(params.rel_filter["exclude"])]

        synonyms = synonyms[synonyms.concept_id.isin(ids)]


        names = dict(zip(concepts.index, concepts.concept_name))

        # procedures = procedures[:5_000]

        data = dict()
        for row in tqdm(procedures.itertuples(), total=len(procedures)):
            S = synonyms[synonyms.concept_id == row.Index]
            if True:
                # if params.c2c == "1to2":
                R = rels[rels.concept_id_1 == row.Index]
            elif params.c2c == "2to1":
                R = rels[rels.concept_id_2 == row.Index]
            else:
                R = rels[
                    (rels.concept_id_1 == row.Index) | (rels.concept_id_2 == row.Index)
                ]

            texts = [f"{row.concept_name}."]
            for s in S.itertuples():
                texts.append(f"Synonym: {s.concept_synonym_name}.")
            for r in R.itertuples():
                rel_name = rel_definitions[r.relationship_id]
                if r.concept_id_1 == row.Index and r.concept_id_2 in names:
                    name = names[r.concept_id_2]
                elif r.concept_id_2 == row.Index and r.concept_id_1 in names:
                    name = names[r.concept_id_1]
                else:
                    print(f"missing name for {r}")
                    continue
                texts.append(f"{rel_name}: {name}.")
            text = " ".join(texts)
            data[row.Index] = text
        data = pd.Series(data, name="concept_text")
        data.to_csv(output[0], index=True)


rule drugs_texts:
    input:
        concepts=paths.CONCEPTS,
        concept_relationship=paths.CONCEPT_RELATIONSHIP,
        synonyms=paths.CONCEPT_SYNONYM,
        relationship=paths.RELATIONSHIP,
    output:
        "data/embeddings/ontology_text/drug_exposure.csv",
    params:
        c2c="1to2",
        max_synonyms=10,
        rel_filter={
            "include": None,
            "exclude": ALL_EXCLUDE,
        },
    run:
        concepts = pd.read_csv(input.concepts, sep="\t", index_col=0)
        relationships = pd.read_csv(input.concept_relationship, sep="\t")
        synonyms = pd.read_csv(input.synonyms, sep="\t")
        rel_definitions = pd.read_csv(input.relationship, sep="\t")

        # concepts = pd.read_csv(paths.CONCEPTS, sep="\t", index_col=0)
        # relationships = pd.read_csv(paths.CONCEPT_RELATIONSHIP, sep="\t")
        # synonyms = pd.read_csv(paths.CONCEPT_SYNONYM, sep="\t")
        # rel_definitions = pd.read_csv(paths.RELATIONSHIP, sep="\t")

        rel_definitions.relationship_name = (
            rel_definitions.relationship_name.str.replace(r"\s*\(.*\)$", "", regex=True)
        )

        rel_definitions = dict(
            zip(rel_definitions.relationship_id, rel_definitions.relationship_name)
        )

        drugs = concepts[
            (concepts.domain_id == "Drug")
            & (concepts.standard_concept == "S")
            & (concepts.invalid_reason.isna())
        ]
        ids = set(drugs.index)
        rels = relationships[
            (relationships.concept_id_1.isin(ids))
            | (relationships.concept_id_2.isin(ids))
        ]
        if params.rel_filter["include"]:
            rels = rels[rels.relationship_id.isin(params.rel_filter["include"])]
        if params.rel_filter["exclude"]:
            rels = rels[~rels.relationship_id.isin(params.rel_filter["exclude"])]

        synonyms = synonyms[synonyms.concept_id.isin(ids)]


        names = dict(zip(concepts.index, concepts.concept_name))

        data = dict()
        for row in tqdm(drugs.itertuples(), total=len(drugs)):
            S = synonyms[synonyms.concept_id == row.Index]
            if True:
                # if params.c2c == "1to2":
                R = rels[rels.concept_id_1 == row.Index]
            elif params.c2c == "2to1":
                R = rels[rels.concept_id_2 == row.Index]
            else:
                R = rels[
                    (rels.concept_id_1 == row.Index) | (rels.concept_id_2 == row.Index)
                ]

            texts = [f"{row.concept_name}."]
            for s in S.head(params.max_synonyms).itertuples():
                texts.append(f"Synonym: {s.concept_synonym_name}.")
            for r in R.itertuples():
                rel_name = rel_definitions[r.relationship_id]
                if r.concept_id_1 == row.Index and r.concept_id_2 in names:
                    name = names[r.concept_id_2]
                elif r.concept_id_2 == row.Index and r.concept_id_1 in names:
                    name = names[r.concept_id_1]
                else:
                    print(f"missing name for {r}")
                    continue
                texts.append(f"{rel_name}: {name}.")
            text = " ".join(texts)
            data[row.Index] = text
        data = pd.Series(data, name="concept_text")
        data.to_csv(output[0], index=True)
