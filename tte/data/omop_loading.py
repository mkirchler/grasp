import numpy as np
from tqdm import tqdm
import h5py
import pickle
import pandas as pd
from scipy import sparse as sp
from tte.utils.misc import PATHS

CONCEPTS = PATHS.CONCEPTS

SOURCE_NAMES = {
    PATHS.PROCESSED_EXTRA: "extra",
} | {
    PATHS.PROCESSED_OMOP.format(source=source): source
    for source in [
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure",
        "ingredient_exposure",
    ]
}


def propagate_ontology(sp_arr, concept_ids, max_sep=2, extra_cols=[] ,binarize_counts=True):
    assert binarize_counts, "only binarize_counts=True is supported"
    anc = pd.read_csv(PATHS.ANCESTORS, sep="\t")
    ancestors = anc[anc.min_levels_of_separation<=max_sep]
    all_ids = set(concept_ids)
    ancestors = ancestors[ancestors.descendant_concept_id.isin(all_ids)]


    extra = sp.lil_matrix(sp_arr.shape, dtype=float)
    extra[:, extra_cols] = sp_arr[:, extra_cols].copy().tocsr()

    sp_arr = sp_arr.astype(bool).tolil()
    sp_arr[:, extra_cols] = 0
    sp_arr = sp_arr.tocsr()
    concept_id_lookup = {c: i for i, c in enumerate(concept_ids)}

    data = []
    rows = []
    cols = []
    for i in tqdm(range(sp_arr.shape[0])):
        row = sp_arr[i]
        inds = row.indices
        desc = set(concept_ids[inds])
        ance = set(ancestors[ancestors.descendant_concept_id.isin(desc)].ancestor_concept_id)
        ance = {c for c in ance if c in all_ids}
        full = desc | set(ance)

        indices = np.array([concept_id_lookup[c] for c in full])

        for ind in indices:
            data.append(True)
            rows.append(i)
            cols.append(ind)
    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)
    new_arr = sp.coo_matrix((data, (rows, cols)), shape=sp_arr.shape).tocsr().astype(float)
    new_arr = new_arr + extra

    return new_arr
    


        





def load_sparse_omop(
    # sources,
    input_fns,
    train_iids,
    min_count_per_concept=25,
    remove_invalid_concepts=True,
    binarize_counts=True,
):
    full_vocab = pd.read_csv(CONCEPTS, sep="\t", index_col=0)
    sources = sorted(input_fns.keys())

    strain = set(train_iids)
    # input_fns = [
    #     (
    #         PATHS.PROCESSED_OMOP.format(source=source)
    #         if source != "extra"
    #         else PATHS.PROCESSED_EXTRA
    #     )
    #     for source in sources
    # ]
    arrs = dict()
    iids = dict()
    cols_concepts = dict()
    cols = dict()
    # for source, fn in zip(sources, input_fns):
    for source in sources:
        fn = input_fns[source]
        if source == "extra":
            source_df = pd.read_csv(fn, index_col=0)
            arrs[source] = sp.csr_matrix(source_df.values)
            iids[source] = source_df.index.values
            cols_concepts[source] = source_df.columns.values
            cols[source] = [-10000000 * i for i in range(1, 1 + len(source_df.columns))]
        else:
            source_iids, source_concepts, source_coded = pickle.load(open(fn, "rb"))
            if binarize_counts:
                source_coded = source_coded.astype(bool).astype(int)

            train_bind = np.array([iid in strain for iid in source_iids])
            include_columns = np.array(
                (source_coded[train_bind].sum(0) >= min_count_per_concept)
            ).flatten()
            source_coded = source_coded[:, include_columns]
            source_concepts = source_concepts[include_columns]

            lookup = vocab_lookup_and_filter(
                source_concepts,
                remove_invalid_concepts=remove_invalid_concepts,
                cached_full_vocab=full_vocab.copy(),
            )
            concepts_bind = np.array([c in lookup for c in source_concepts])
            source_concepts = source_concepts[concepts_bind]
            source_coded = source_coded[:, concepts_bind]

            concepts = np.array([lookup[c] for c in source_concepts])

            arrs[source] = source_coded
            iids[source] = source_iids
            cols_concepts[source] = concepts
            cols[source] = source_concepts

    all_iids = np.unique([iid for source in iids.values() for iid in source])
    iid_lookups = {
        source: {iid: i for i, iid in enumerate(iids[source])} for source in sources
    }
    print("stitching data")
    rows = []
    for iid in tqdm(all_iids):
        iid_rows = []
        for source in sources:
            if iid in iid_lookups[source]:
                row = arrs[source][iid_lookups[source][iid]]
            else:
                row = sp.csr_matrix((1, arrs[source].shape[1]))
            iid_rows.append(row)
        row = sp.hstack(iid_rows)
        rows.append(row)
    arr = sp.vstack(rows)
    cols_concepts = np.concatenate([cols_concepts[source] for source in sources])
    cols = np.concatenate([cols[source] for source in sources])
    return arr, all_iids, cols_concepts, cols

    return arrs, iids, cols_concepts, cols


# deprecated! (I think!?)
def load_omop(
    out_fn_h5,
    input_fns,
    # sources,
    train_iids,
    val_iids,
    test_iids,
    binarize_counts=True,
    min_count_per_concept=25,
    min_var_per_concept=0.01,
    remove_invalid_concepts=True,
    fill_up_missing_with_zeros=True,
):
    f = h5py.File(out_fn_h5, "w")
    strain = set(train_iids)
    all_iids = np.concatenate([train_iids, val_iids, test_iids])
    sources = [SOURCE_NAMES[fn] for fn in input_fns]
    # source_dfs = []
    for source, fn in zip(sources, input_fns):
        print(f"loading {source}")
        f.create_group(source)
        pth = fn
        # pth = f"data/omop/{source}.pkl"
        if source == "extra":
            source_df = pd.read_csv(pth, index_col=0)
            f[f"{source}/columns"] = np.array(source_df.columns)
            f[f"{source}/description"] = "direct"
            for iid, row in tqdm(source_df.iterrows(), total=len(source_df)):
                f[f"{source}/{iid}"] = row.values
        else:
            source_iids, source_concepts, source_coded = pickle.load(open(pth, "rb"))
            if binarize_counts:
                source_coded = source_coded.astype(bool).astype(int)

            train_bind = np.array([iid in strain for iid in source_iids])

            var = (
                np.array(source_coded[train_bind].power(2).mean(0)).flatten()
                - np.array(source_coded[train_bind].mean(0)).flatten() ** 2
            )
            include_columns = np.array(
                (source_coded[train_bind].sum(0) >= min_count_per_concept)
                & (var > min_var_per_concept)
            ).flatten()

            source_coded = source_coded[:, include_columns]
            source_concepts = source_concepts[include_columns]

            lookup = vocab_lookup_and_filter(
                source_concepts, remove_invalid_concepts=remove_invalid_concepts
            )
            concepts_bind = np.array([c in lookup for c in source_concepts])
            source_concepts = source_concepts[concepts_bind]
            source_coded = source_coded[:, concepts_bind]

            concepts = np.array([lookup[c] for c in source_concepts])
            source_df = pd.DataFrame(
                index=source_iids,
                columns=concepts,
                data=source_coded.toarray(),
            )
            if fill_up_missing_with_zeros:
                missing_iids = all_iids[~np.isin(all_iids, source_iids)]
                print(f"filling up {len(missing_iids)} missing iids with zeros")
                print(missing_iids)
                print(f"shape before: {source_df.shape}")
                source_df = pd.concat(
                    [
                        source_df,
                        pd.DataFrame(0, index=missing_iids, columns=source_df.columns),
                    ]
                )
                print(f"shape after: {source_df.shape}")
            print("saving to hdf5")
            f[f"{source}/columns"] = np.array(source_df.columns)
            f[f"{source}/description"] = "binary_index"
            for iid, row in tqdm(source_df.iterrows(), total=len(source_df)):
                f[f"{source}/{iid}"] = np.where(row.values > 0)[0]
            # source_dfs.append(source_df)
    # source_df = pd.concat(source_dfs, axis=1)

    # train = source_df.loc[train_iids]
    # val = source_df.loc[val_iids]
    # test = source_df.loc[test_iids]
    # return train, val, test


def vocab_lookup_and_filter(
    concept_ids, remove_invalid_concepts=True, cached_full_vocab=None
):
    if cached_full_vocab is not None:
        full_vocab = cached_full_vocab
    else:
        full_vocab = pd.read_csv(CONCEPTS, sep="\t", index_col=0)
    if remove_invalid_concepts:
        full_vocab = full_vocab[
            (full_vocab.invalid_reason.isna())
            & (full_vocab.valid_end_date == 20991231)
            & (full_vocab.standard_concept == "S")
        ]

    joint = np.intersect1d(concept_ids, full_vocab.index)

    return {idx: f'{full_vocab.loc[idx, "concept_name"]}_OMOP{idx}' for idx in joint}
