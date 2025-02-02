import numpy as np
import math
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from tte.utils.misc import VARS, PATHS


DOMAINS = {
    "condition_occurrence": ["Condition"],
    "procedure_occurrence": ["Procedure"],
    "drug_exposure": ["Drug"],
}


def init_openai():
    key_fn = PATHS.OPENAI_KEY
    org = VARS.OPENAI_ORG
    key = open(key_fn, "r").read().strip()
    client = OpenAI(api_key=key, organization=org)
    return client


MODELS = ["text-embedding-ada-002", "text-embedding-3-large", "text-embedding-3-small"]


def fetch_extra_embeddings(
    concepts,
    save_fn,
    client,
    model,
):
    """
    concepts: dict of {concept: "string description to get embedding for"}
    """

    embeddings = pd.DataFrame(
        {
            concept: get_embeddings(strings=[string], client=client, model=model)[0]
            for concept, string in tqdm(concepts.items())
        }
    ).T
    embeddings.columns = [f"embdim_{i}" for i in range(embeddings.shape[1])]
    embeddings.to_csv(save_fn)

def collect_all_embeddings(source, only_valid_standard=True):
    concepts_fn = PATHS.CONCEPTS
    full_vocab = pd.read_csv(concepts_fn, sep="\t", index_col=0)
    domains = DOMAINS[source]
    source_vocab = full_vocab[full_vocab.domain_id.isin(domains)]
    if only_valid_standard:
        # filter for not in vocab
        source_vocab = source_vocab[
            (source_vocab.invalid_reason.isna())
            & (source_vocab.valid_end_date == 20991231)
            & (source_vocab.standard_concept == "S")
        ]
    return source_vocab


def fetch_all_embeddings(
    source,
    save_fn,
    concepts_fn,
    client,
    model,
    batch_size=16,
    empty_concepts=[],
    only_valid_standard=False,
    debug_nrows=None,
):
    full_vocab = pd.read_csv(concepts_fn, sep="\t", index_col=0)
    domains = DOMAINS[source]
    source_vocab = full_vocab[full_vocab.domain_id.isin(domains)]

    if only_valid_standard:
        # filter for not in vocab
        source_vocab = source_vocab[
            (source_vocab.invalid_reason.isna())
            & (source_vocab.valid_end_date == 20991231)
            & (source_vocab.standard_concept == "S")
        ]
    concepts = source_vocab.concept_name.to_list()[:debug_nrows]
    concepts += empty_concepts
    concepts = sorted(np.unique(concepts))

    n_batches = math.ceil(len(concepts) / batch_size)
    embeddings = []
    for batch in tqdm(range(n_batches)):
        embeddings.append(
            get_embeddings(
                strings=concepts[batch * batch_size : (batch + 1) * batch_size],
                client=client,
                model=model,
            )
        )
    embeddings = np.concatenate(embeddings, axis=0)

    df = pd.DataFrame(
        embeddings,
        index=concepts,
        columns=[f"embdim_{i}" for i in range(embeddings.shape[1])],
    )
    df.to_csv(save_fn)


def get_embeddings(strings, client, model="text-embedding-ada-002", **kwargs):
    E = client.embeddings.create(input=strings, model=model, **kwargs)
    return np.stack([np.array(x.embedding) for x in E.data])
