import numpy as np
import os
from pathlib import Path
from lifelines.utils import concordance_index
import sys
import pandas as pd
from lifelines import CoxPHFitter
import pickle

sys.path.insert(0, str(Path.cwd().absolute()))
from tte.utils.misc import (
    load_vars,
    load_paths,
)
from tte.embedding_loading.openai_new_emb import (
    init_openai,
    fetch_all_embeddings,
    fetch_extra_embeddings,
)


paths = load_paths()
consts = load_vars()


rule download_all_openai_new_embeddings:
    input:
        expand(paths.OPENAI_NEW_EMBEDDINGS, source=consts.RAW_SOURCES),
        paths.OPENAI_NEW_EMBEDDINGS.format(source="ingredient_exposure"),
        paths.OPENAI_NEW_EMBEDDINGS.format(source="extra"),


rule create_openai_new_embeddings_ingredients:
    input:
        paths.OPENAI_NEW_EMBEDDINGS.format(source="drug_exposure"),
    output:
        paths.OPENAI_NEW_EMBEDDINGS.format(source="ingredient_exposure"),
    run:
        ainput = os.path.abspath(input[0])
        aoutput = os.path.abspath(output[0])
        os.system(f"ln -s {ainput} {aoutput}")


rule download_openai_new_embeddings_source:
    input:
        concepts=paths.CONCEPTS,
    output:
        paths.OPENAI_NEW_EMBEDDINGS,
    params:
        empty=consts.EMPTY_CONCEPTS,
        only_valid_standard=True,
        debug_nrows=None,
        batch_size=256,
        model="text-embedding-3-large",
    run:
        client = init_openai()
        fetch_all_embeddings(
            source=wildcards.source,
            save_fn=output[0],
            concepts_fn=input.concepts,
            client=client,
            model=params.model,
            empty_concepts=params.empty[wildcards.source],
            only_valid_standard=params.only_valid_standard,
            debug_nrows=params.debug_nrows,
            batch_size=params.batch_size,
        )


rule download_openai_new_embeddings_extra:
    output:
        paths.OPENAI_NEW_EMBEDDINGS.format(source="extra"),
    params:
        extra_concepts=consts.EXTRA_CONCEPTS,
        model="text-embedding-3-large",
    run:
        client = init_openai()
        fetch_extra_embeddings(
            concepts=params.extra_concepts,
            save_fn=output[0],
            client=client,
            model=params.model,
        )
