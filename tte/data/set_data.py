import h5py
from time import time
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

NAN_VALUE = -9.0

PRELOAD_OLINK = False
# PRELOAD_OLINK = True


def setup_set_dataloader(
    records_h5fns,
    embeddings_h5fns,
    survival,
    n_token_fill_or_sample=15,
    fill_value=0.0,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    binarize=False,
    empty_concepts=dict(),
    input_dim=None,
    load_embeddings_into_memory=True,
    load_records_into_memory=True,
    aux_survival=None,
):
    E = survival[[col for col in survival.columns if col.endswith("_event")]]
    D = survival[[col for col in survival.columns if col.endswith("_duration")]]
    if aux_survival is not None:
        aux_E = aux_survival[[col for col in aux_survival.columns if col.endswith("_event")]]
        aux_D = aux_survival[[col for col in aux_survival.columns if col.endswith("_duration")]]
    else:
        aux_E = None
        aux_D = None
    dataset = SetSurvivalData(
        records_h5fns=records_h5fns,
        embeddings_h5fns=embeddings_h5fns,
        duration=D,
        event=E,
        n_token_fill_or_sample=n_token_fill_or_sample,
        fill_value=fill_value,
        binarize=binarize,
        empty_concepts=empty_concepts,
        input_dim=input_dim,
        load_embeddings_into_memory=load_embeddings_into_memory,
        load_records_into_memory=load_records_into_memory,
        aux_duration=aux_D,
        aux_event=aux_E,
    )

    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return dl


class SetSurvivalData(Dataset):
    def __init__(
        self,
        records_h5fns,
        embeddings_h5fns,
        duration,
        event,
        empty_concepts,
        n_token_fill_or_sample=15,
        fill_value=0.0,
        binarize=False,
        input_dim=None,
        load_embeddings_into_memory=True,
        load_records_into_memory=True,
        aux_duration=None,
        aux_event=None,
    ):
        self.binarize = binarize
        self.sources = sorted(records_h5fns.keys())
        self.target_names = [col[:-6] for col in event.columns]
        self.n_targets = len(self.target_names)
        self.empty_concepts = empty_concepts
        self.input_dim = input_dim
        self.load_embeddings_into_memory = load_embeddings_into_memory
        self.load_records_into_memory = load_records_into_memory


        if load_embeddings_into_memory:
            self.embedding_stores = {
                source: load_mem_embedding_store(embeddings_h5fn)
                for source, embeddings_h5fn in embeddings_h5fns.items()
            }
        else:
            self.embedding_stores = {
                source: h5py.File(embeddings_h5fn, "r")
                for source, embeddings_h5fn in embeddings_h5fns.items()
            }

        if load_records_into_memory:
            endpoint_iids = set(
                [str(iid) for iid in np.intersect1d(duration.index, event.index)]
            )

            self.records = {
                source: load_mem_embedding_store(records_h5fn, filter_by=endpoint_iids)
                for source, records_h5fn in records_h5fns.items()
            }

        else:
            self.records = {
                source: h5py.File(records_h5fn, "r")
                for source, records_h5fn in records_h5fns.items()
            }
        self.embedding_store_keys = {
            source: set(store.keys()) for source, store in self.embedding_stores.items()
        }
        available_iids = set.intersection(
            *[{int(iid) for iid in store.keys()} for store in self.records.values()]
        )

        iids = np.array(
            list(available_iids.intersection(duration.index).intersection(event.index))
        )
        self.duration = duration.loc[iids]
        self.event = event.loc[iids]
        self.index_names = self.event.index
        if (not aux_duration is None) and (not aux_event is None):
            self.aux_target_names = [col[:-6] for col in aux_event.columns]
            self.n_aux_target = len(self.aux_target_names)
            self.aux_duration = aux_duration.loc[iids]
            self.aux_event = aux_event.loc[iids]
        else:
            self.aux_target_names = []
            self.n_aux_target = 0
            self.aux_duration = None
            self.aux_event = None

        self.token_size = self._load_tensor(0)[0].shape[1]
        if isinstance(fill_value, torch.Tensor):
            self.fill_value = fill_value
        else:
            self.fill_value = torch.tensor(
                [fill_value] * self.token_size, dtype=torch.float
            )
        self.n_token_fill_or_sample = n_token_fill_or_sample

        # print('\n'*2, '***'*10, 'WARNING! USING DUMMY DATA ONLY')
        # self.dummy_data = np.random.rand(100, self.token_size).astype(np.float32)
        # print('***'*10, '\n'*2)

    def __getitem__(self, idx):
        # print(f'item {idx}')
        tensor, ages, values, concept_names, iid = self._load_tensor(idx)
        # print('tensor loaded')
        if not self.n_token_fill_or_sample is None:
            if tensor.shape[0] < self.n_token_fill_or_sample:
                # print("pruning")
                tensor = torch.cat(
                    [
                        tensor,
                        self.fill_value.repeat(
                            self.n_token_fill_or_sample - tensor.shape[0], 1
                        ),
                    ],
                    dim=0,
                )
                ages = torch.cat(
                    [
                        ages,
                        torch.tensor(
                            [NAN_VALUE] * (self.n_token_fill_or_sample - ages.shape[0]),
                            dtype=torch.float,
                        ),
                    ],
                    dim=0,
                )
                values = torch.cat(
                    [
                        values,
                        torch.tensor(
                            [NAN_VALUE]
                            * (self.n_token_fill_or_sample - values.shape[0]),
                            dtype=torch.float,
                        ),
                    ],
                    dim=0,
                )
                concept_names = concept_names + ["FILL-UP-CONCEPTS"] * (
                    self.n_token_fill_or_sample - len(concept_names)
                )
            elif tensor.shape[0] > self.n_token_fill_or_sample:
                # print("sampling")
                token_idxs = torch.randperm(tensor.shape[0])[
                    : self.n_token_fill_or_sample
                ]
                tensor = tensor[token_idxs]
                ages = ages[token_idxs]
                values = values[token_idxs]
                concept_names = np.array(concept_names)[token_idxs].tolist()

        return {
            "data": tensor,
            "ages": ages,
            "values": values,
            "duration": self.duration.iloc[idx].values,
            "event": self.event.iloc[idx].values,
            "concept_names": concept_names,
            'iid': iid,
            **self._get_aux(idx),
        }
    def _get_aux(self, idx):
        # TODO
        if self.aux_duration is None:
            return dict()
        else:
            return {
                'aux_duration': self.aux_duration.iloc[idx].values,
                'aux_event': self.aux_event.iloc[idx].values,
            }

    def _load_subtensor(self, idx):
        iid = self.index_names[idx]
        tensors = []
        ages = []
        values = []
        total_concept_names = []
        for source in self.sources:
            t0 = time()
            source_records = get_records(
                self.records,
                source,
                iid,
                loaded_into_memory=self.load_records_into_memory,
            )
            t1 = time()
            if self.binarize:
                source_records = keep_only_last(source_records)
            t2 = time()

            if PRELOAD_OLINK:
                if "olink" in source:
                    concept_names = [
                        str(concept) for concept, age, value in source_records
                    ]

                else:
                    concept_names = [
                        concept.decode().replace("/", "_")
                        for concept, age, value in source_records
                    ]
            else:
                if "olink" in source:
                    concept_names = [
                        str(concept) for concept, age, value in source_records
                    ]
                else:

                    concept_names = [
                        concept.decode().replace("/", "_")
                        for concept, age, value in source_records
                    ]
            concepts_available = [
                name in self.embedding_store_keys[source] for name in concept_names
            ]
            t3 = time()
            concept_names = [
                name
                for name, available in zip(concept_names, concepts_available)
                if available
            ]
            concept_ages = [
                age
                for (_, age, _), available in zip(source_records, concepts_available)
                if available
            ]

            concept_values = [
                value
                for (_, _, value), available in zip(source_records, concepts_available)
                if available
            ]

            if len(concept_names) == 0:
                concept_names = self.empty_concepts[source]
                concept_ages = [NAN_VALUE] * len(concept_names)
                concept_values = [NAN_VALUE] * len(concept_names)
            t35 = time()

            if hasattr(self, "dummy_data"):
                tokens = self.dummy_data
            else:
                t36 = time()
                tokens = np.array(
                    [
                        get_embedding(
                            self.embedding_stores,
                            source,
                            concept,
                            loaded_into_memory=self.load_embeddings_into_memory,
                        )
                        for concept in concept_names
                    ]
                )
                t37 = time()
                token_ages = np.array(concept_ages)
                token_values = np.array(concept_values)
            t4 = time()
            tensors.append(torch.from_numpy(tokens).float())
            ages.append(torch.from_numpy(token_ages).float())
            values.append(torch.from_numpy(token_values).float())
            total_concept_names += list(concept_names)
        t5 = time()
        tensor = torch.cat(tensors)
        if not self.input_dim is None:
            tensor = tensor[:, : self.input_dim] / torch.norm(
                tensor[:, : self.input_dim], dim=1, keepdim=True
            )
        ages = torch.cat(ages)
        values = torch.cat(values)
        # print times:
        print(f"loading records: {1000*(t1-t0):.3f}")
        print(f"pruning records: {1000*(t2-t1):.3f}")
        print(f"loading embeddings: {1000*(t3-t2):.3f}")
        print(f"checking embeddings: {1000*(t35-t3):.3f}")
        print(f"pruning embeddings: {1000*(t36-t35):.3f}")
        print(f"loading embeddings: {1000*(t37-t36):.3f}")
        print(f"concatenating tensors: {1000*(t4-t37):.3f}")
        print(f"pruning embeddings: {1000*(t4-t3):.3f}")
        print(f"concatenating tensors: {1000*(t5-t4):.3f}")
        print(f"total time: {1000*(t5-t0):.3f}")
        return tensor, ages, values, total_concept_names

    def _load_tensor(self, idx):
        iid = self.index_names[idx]
        # print(f'loading iid {iid}')
        tensors = []
        ages = []
        values = []
        total_concept_names = []
        for source in self.sources:
            # print(f'loading source {source}')
            t0 = time()
            # source_records = self.records[source][str(iid)][()]
            source_records = get_records(
                self.records,
                source,
                iid,
                loaded_into_memory=self.load_records_into_memory,
            )
            t1 = time()
            if self.binarize:
                source_records = keep_only_last(source_records)
            t2 = time()

            if PRELOAD_OLINK:
                if "olink" in source:
                    concept_names = [
                        str(concept) for concept, age, value in source_records
                    ]

                else:
                    concept_names = [
                        concept.decode().replace("/", "_")
                        for concept, age, value in source_records
                    ]
            else:
                if "olink" in source:
                    concept_names = [
                        str(concept) for concept, age, value in source_records
                    ]
                else:

                    concept_names = [
                        concept.decode().replace("/", "_")
                        for concept, age, value in source_records
                    ]
            # if self.binarize:
            #     raise NotImplementedError("doesn't handle values and ages yet")
            #     concept_names = np.unique(concept_names)
            # print(concept_names)
            concepts_available = [
                name in self.embedding_store_keys[source] for name in concept_names
            ]
            t3 = time()
            concept_names = [
                # name.replace("/", "_")
                name
                for name, available in zip(concept_names, concepts_available)
                if available
            ]
            # print(concept_names)
            concept_ages = [
                age
                for (_, age, _), available in zip(source_records, concepts_available)
                if available
            ]

            concept_values = [
                value
                for (_, _, value), available in zip(source_records, concepts_available)
                if available
            ]

            if len(concept_names) == 0:
                concept_names = self.empty_concepts[source]
                concept_ages = [NAN_VALUE] * len(concept_names)
                concept_values = [NAN_VALUE] * len(concept_names)
            # print(concept_names)
            t35 = time()

            # dummy data for speed checks
            if hasattr(self, "dummy_data"):
                tokens = self.dummy_data
            else:
                # print('here actually loading real data!')
                t36 = time()
                tokens = np.array(
                    [
                        # self.embedding_stores[source][concept][()]
                        get_embedding(
                            self.embedding_stores,
                            source,
                            concept,
                            loaded_into_memory=self.load_embeddings_into_memory,
                        )
                        for concept in concept_names
                    ]
                )
                t37 = time()
                token_ages = np.array(concept_ages)
                token_values = np.array(concept_values)
            t4 = time()
            # print(f'loaded {len(tokens)} tokens')
            tensors.append(torch.from_numpy(tokens).float())
            ages.append(torch.from_numpy(token_ages).float())
            values.append(torch.from_numpy(token_values).float())
            total_concept_names += list(concept_names)

        t5 = time()
        tensor = torch.cat(tensors)
        if not self.input_dim is None:
            tensor = tensor[:, : self.input_dim] / torch.norm(
                tensor[:, : self.input_dim], dim=1, keepdim=True
            )
        ages = torch.cat(ages)
        values = torch.cat(values)
        # print times:
        # print(f"loading records: {t1-t0:.3f}")
        # print(f"pruning records: {t2-t1:.3f}")
        # print(f"loading embeddings: {t3-t2:.3f}")
        # print(f"checking embeddings: {t35-t3:.3f}")
        # print(f"pruning embeddings: {t36-t35:.3f}")
        # print(f"loading embeddings: {t37-t36:.3f}")
        # print(f"concatenating tensors: {t4-t37:.3f}")
        # print(f'pruning embeddings: {t4-t3:.3f}')
        # print(f"concatenating tensors: {t5-t4:.3f}")
        # print(f"total time: {t5-t0:.3f}")
        return tensor, ages, values, total_concept_names, iid

    def get_events(self):
        return self.event.rename(
            columns={
                col: name for col, name in zip(self.event.columns, self.target_names)
            }
        )

    def get_durations(self):
        return self.duration.rename(
            columns={
                col: name for col, name in zip(self.duration.columns, self.target_names)
            }
        )

    def __len__(self):
        return len(self.duration)


def get_embedding(embedding_stores, source, concept, loaded_into_memory=False):
    if "olink" in source and PRELOAD_OLINK:
        return embedding_stores[source][concept]
    elif loaded_into_memory:
        return embedding_stores[source][concept]
    else:
        return embedding_stores[source][concept][()]


def get_records(records, source, iid, loaded_into_memory=False):
    if "olink" in source and PRELOAD_OLINK:
        return records[source][str(iid)]
    elif loaded_into_memory:
        return records[source][str(iid)]
    else:
        return records[source][str(iid)][()]


def load_mem_embedding_from_pickle(embeddings_pkl, filter_by=None, return_key=None):
    print(f'loading {embeddings_pkl} pkl object instead of h5 object for better speed')
    import pickle

    with open(embeddings_pkl, "rb") as f:
        data, codec = pickle.load(f)
    reverse_codec = {v: k for k, v in codec.items()}

    vlen_str_dtype = h5py.special_dtype(vlen=str)
    dtype = np.dtype(
        [("concept_name", vlen_str_dtype), ("age", np.float32), ("value", np.int8)]
    )
    ret = dict()
    keys = list(data.keys())
    if not filter_by is None:
        keys = [key for key in keys if key in filter_by]
    for key in tqdm(data):
        arr = data[key]
        item = np.array(
            [
                (reverse_codec[concept_code], age, value)
                for concept_code, age, value in arr
            ],
            dtype=dtype,
        )
        ret[key] = item
    if return_key:
        return ret, return_key
    else:
        return ret


def load_mem_embedding_store(embeddings_h5fn, filter_by=None, return_key=None):
    if embeddings_h5fn.endswith(".pkl"):
        return load_mem_embedding_from_pickle(
            embeddings_h5fn, filter_by=filter_by, return_key=return_key
        )
    f = h5py.File(embeddings_h5fn, "r")
    keys = list(f.keys())
    if not filter_by is None:
        keys = [key for key in keys if key in filter_by]

    if return_key:
        ret = {key: f[key][()] for key in keys}
        return ret, return_key
    else:
        ret = {
            key: f[key][()]
            for key in tqdm(keys, desc=f"loading {embeddings_h5fn} into memory")
        }
        return ret


def load_olink_embedding_store(embeddings_h5fn):
    f = h5py.File(embeddings_h5fn, "r")
    keys = list(f.keys())
    return {key: f[key][()] for key in tqdm(keys, desc="loading olink")}


class SetSurvivalDataOLD(Dataset):
    def __init__(
        self,
        h5fn,
        duration,
        event,
        n_token_fill_or_sample=15,
        fill_value=0.0,
        use_cache=False,
        binarize=False,
    ):
        """
        n_token_fill_or_sample: if None, will return all tokens. Otherwise, will either fill up to n_token_fill_or_sample with fill_value (if fewer than n_token_fill_or_sample tokens), or sample n_token_fill_or_sample tokens (if more than n_token_fill_or_sample tokens)
        fill_value: float or torch.Tensor of same size as embedding
        """
        self.target_names = [col[:-6] for col in event.columns]
        self.n_targets = len(self.target_names)
        self.h5f = h5py.File(h5fn, "r")
        available_iids = {int(iid) for iid in self.h5f.keys()}
        iids = np.array(
            list(available_iids.intersection(duration.index).intersection(event.index))
        )
        self.use_cache = use_cache
        if self.use_cache:
            self.cache = dict()
        # for i, iid in tqdm(enumerate(iids), total=len(iids)):
        #     self.cache[i] = torch.tensor(np.array(f[str(iid)], dtype=torch.float32))
        self.duration = duration.loc[iids]
        self.event = event.loc[iids]
        self.index_names = self.event.index

        self.token_size = self._load_tensor(0)[0].shape[1]
        if isinstance(fill_value, torch.Tensor):
            self.fill_value = fill_value
        else:
            self.fill_value = torch.tensor(
                [fill_value] * self.token_size, dtype=torch.float32
            )
        self.n_token_fill_or_sample = n_token_fill_or_sample
        self.binarize = binarize

    def get_events(self):
        return self.event.rename(
            columns={
                col: name for col, name in zip(self.event.columns, self.target_names)
            }
        )

    def get_durations(self):
        return self.duration.rename(
            columns={
                col: name for col, name in zip(self.duration.columns, self.target_names)
            }
        )

    def __len__(self):
        return len(self.duration)

    def _load_tensor(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        else:
            iid = self.index_names[idx]
            # print(iid)
            # print(self.h5f[str(iid)].shape)
            tensor = torch.tensor(self.h5f[str(iid)][()], dtype=torch.float32)
            if self.use_cache:
                self.cache[idx] = tensor
            return tensor

    def __getitem__(self, idx):
        tensor = self._load_tensor(idx)
        if not self.n_token_fill_or_sample is None:
            if tensor.shape[0] < self.n_token_fill_or_sample:
                tensor = torch.cat(
                    [
                        tensor,
                        self.fill_value.repeat(
                            self.n_token_fill_or_sample - tensor.shape[0], 1
                        ),
                    ],
                    dim=0,
                )
            elif tensor.shape[0] > self.n_token_fill_or_sample:
                tensor = tensor[
                    torch.randperm(tensor.shape[0])[: self.n_token_fill_or_sample]
                ]

        return {
            "data": tensor,
            "duration": self.duration.iloc[idx].values,
            "event": self.event.iloc[idx].values,
        }


def keep_only_last(source_records):
    new_source_records = dict()
    for concept_name, age, value in source_records:
        if (
            concept_name in new_source_records
            and new_source_records[concept_name][0] < age
            or not concept_name in new_source_records
        ):
            new_source_records[concept_name] = (age, value)
    return np.array(
        [
            (concept_name, age, value)
            for concept_name, (age, value) in new_source_records.items()
        ],
        dtype=source_records.dtype,
    )
