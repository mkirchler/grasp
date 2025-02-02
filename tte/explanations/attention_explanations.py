import torch
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
import torch
# import torch.nn as nn
# from einops import rearrange
import pickle

from captum.attr import Occlusion

from tte.models.exp_transformers import (
    load_explainable_transformer_from_normal_transformer,
)

# from tte.explanations.vit_explanation_generator import LRP
from tte.explanations.explanation_generator import LRP
from tte.explanations.data_setup import get_loader


# def load_setup(bs=32, dev="cpu"):
#     center = 11011
#     dump = pickle.load(
#         open("export/attention_singletable/attention_singletable.pkl", "rb")
#     )
#     net = dump["models"][
#         f"models/attention_singletable/{center}/sources~condition_occurrence/best_model.pkl"
#     ]["net"]

#     dl = get_loader(
#         center, sources=["condition_occurrence"], split="test", bs=bs, num_workers=0
#     )
#     explainable_net = load_explainable_transformer_from_normal_transformer(net)
#     attribution_generator = LRP(explainable_net, dev=dev)
#     batch = next(iter(dl))
#     x = batch["data"]
#     concept_names = np.array(batch["concept_names"]).T

#     return attribution_generator, x, concept_names, dl


# def load_captum_setup(bs=32, dev="cpu", binarize=True):
#     center = 11011
#     dump = pickle.load(
#         open("export/attention_singletable/attention_singletable.pkl", "rb")
#     )
#     net = dump["models"][
#         f"models/attention_singletable/{center}/sources~condition_occurrence/best_model.pkl"
#     ]["net"]

#     dl = get_loader(
#         center,
#         sources=["condition_occurrence"],
#         split="test",
#         bs=bs,
#         num_workers=0,
#         binarize=binarize,
#     )
#     return net, dl

from tte.explanations.manual_occlusion import attribute_occlusion
def occlusion_multibatch_detail_all(net, dl, n_batches=None, ndim=22, zero_or_remove='zero'):
    assert zero_or_remove in ['zero', 'remove']
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)

    n = 0
    inds_data = [dict() for _ in range(ndim)]
    for batch in tqdm(dl, total=n_batches):
        concepts = np.array(batch["concept_names"]).T
        # input_token_dim = batch['data'].shape[-1]
        iids = batch["iid"]

        def fwd_fct(x):
            data = {
                "data": x.to(dev),
                "ages": batch["ages"].to(dev),
                "values": batch["values"].to(dev),
            }
            return net(data)

        attributions = attribute_occlusion(fwd_fct, batch['data'])

        for k in range(ndim):
            for i in range(len(attributions)):
                concept_names = concepts[i]
                ind_data = defaultdict(lambda: [])
                for j, concept in enumerate(concept_names):
                    ind_data[concept] += [attributions[i, j, k].item()]
                inds_data[k][iids[i].item()] = dict((key, np.mean(val)) for key, val in ind_data.items() )
        n += 1
        if (not n_batches is None) and (n == n_batches):
            break
    Is, Cs, Ms = [], [], []
    for k in range(ndim):
        I, C, M = attr_to_sparse(inds_data[k])
        Is.append(I)
        Cs.append(C)
        Ms.append(M)
    assert all([(Is[0] == I).all() for I in Is])
    assert all([(Cs[0] == C).all() for C in Cs])
    return Is[0], Cs[0], Ms

    #     inds_data[k] = pd.DataFrame(inds_data[k]).T
    # inds_data = [pd.DataFrame(I).T for I in inds_data]
    return inds_data

from scipy.sparse import csr_matrix, coo_matrix, coo_array

def attr_to_sparse(attr):
    iids = sorted(attr.keys())
    concepts = []
    for iid in iids:
        concepts += list(attr[iid].keys())
    concepts = np.unique(concepts)
    lookup = {c: i for i, c in enumerate(concepts)}
    rows = []
    cols = []
    data = []
    for i, iid in tqdm(enumerate(iids)):
        for concept in attr[iid]:
            rows.append(i)
            cols.append(lookup[concept])
            data.append(attr[iid][concept])

    # matrix = coo_array((data, (rows, cols)), shape=(len(iids), len(concepts)))
    matrix = coo_matrix((data, (rows, cols)), shape=(len(iids), len(concepts))).tocsr()
    return np.array(iids), concepts, matrix
        



        
    return concepts

    


def occlusion_multibatch_detail(net, dl, target=14, n_batches=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)

    n = 0
    inds_data = dict()
    for batch in tqdm(dl, total=n_batches):
        concepts = np.array(batch["concept_names"]).T
        input_token_dim = batch['data'].shape[-1]
        iids = batch["iid"]

        def fwd_fct(x):
            data = {
                "data": x.to(dev),
                "ages": batch["ages"].to(dev),
                "values": batch["values"].to(dev),
            }
            return net(data)

        occlusion = Occlusion(fwd_fct)
        attributions = occlusion.attribute(
            batch["data"],
            sliding_window_shapes=(1, input_token_dim),
            strides=(1, 1),
            target=target,
        ).to("cpu")
        # average out last dim -> is all identical
        attributions = attributions.mean(-1)

        for i in range(len(attributions)):
            concept_names = concepts[i]
            ind_data = defaultdict(lambda: [])
            for j, concept in enumerate(concept_names):
                ind_data[concept] += [attributions[i, j].item()]
            inds_data[iids[i].item()] = dict((key, np.mean(val)) for key, val in ind_data.items() )
        n += 1
        if (not n_batches is None) and (n == n_batches):
            break
    inds_data = pd.DataFrame(inds_data).T
    return inds_data

                
            

    #     for i in range(len(attributions)):
    #         concept_names = concepts[i]
    #         for j, concept in enumerate(concept_names):
    #             all_attributions[concept] += [attributions[i, j].item()]
    #     n += 1
    #     if (not n_batches is None) and (n == n_batches):
    #         break

    # res = dict()
    # for concept in all_attributions:
    #     res[concept] = dict(
    #         count=len(all_attributions[concept]),
    #         avg_val=np.mean(all_attributions[concept]),
    #         std=np.std(all_attributions[concept]),
    #     )
    # res = pd.DataFrame(res).T
    # return res


def occlusion_multibatch(net, dl, target=14, n_batches=2):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)

    n = 0
    all_attributions = defaultdict(lambda: [], dict())
    for batch in tqdm(dl, total=n_batches):
        concepts = np.array(batch["concept_names"]).T
        input_token_dim = batch['data'].shape[-1]

        def fwd_fct(x):
            data = {
                "data": x.to(dev),
                "ages": batch["ages"].to(dev),
                "values": batch["values"].to(dev),
            }
            return net(data)

        occlusion = Occlusion(fwd_fct)
        attributions = occlusion.attribute(
            batch["data"],
            sliding_window_shapes=(1, input_token_dim),
            strides=(1, 1),
            target=target,
        ).to("cpu")
        # average out last dim -> is all identical
        attributions = attributions.mean(-1)
        for i in range(len(attributions)):
            concept_names = concepts[i]
            for j, concept in enumerate(concept_names):
                all_attributions[concept] += [attributions[i, j].item()]
        n += 1
        if n == n_batches:
            break

    res = dict()
    for concept in all_attributions:
        res[concept] = dict(
            count=len(all_attributions[concept]),
            avg_val=np.mean(all_attributions[concept]),
            std=np.std(all_attributions[concept]),
        )
    res = pd.DataFrame(res).T
    return res


def occlusion_batch(net, batch, target=14):
    concepts = np.array(batch["concept_names"]).T

    def fwd_fct(x):
        data = {
            "data": x,
            "ages": batch["ages"],
            "values": batch["values"],
        }
        return net(data)

    occlusion = Occlusion(fwd_fct)
    attributions = occlusion.attribute(
        batch["data"], sliding_window_shapes=(1, 1536), strides=(1, 1), target=target
    )
    # average out last dim -> is all identical
    attributions = attributions.mean(-1)
    all_attributions = defaultdict(lambda: [], dict())
    for i in range(len(attributions)):
        concept_names = concepts[i]
        for j, concept in enumerate(concept_names):
            all_attributions[concept] += [attributions[i, j].item()]
    res = dict()
    for concept in all_attributions:
        res[concept] = dict(
            count=len(all_attributions[concept]),
            avg_val=np.mean(all_attributions[concept]),
            std=np.std(all_attributions[concept]),
        )
    res = pd.DataFrame(res).T
    return res


def aggregate_explanations(
    attribution_generator,
    dl,
    method="transformer_attribution",
    class_index=None,
    normalized=False,
    n_batches=2,
    const_mult=1e6,
    remove_fillup=True,
):
    res = []
    concept_counts = defaultdict(lambda: 0, dict())
    concept_indiv_counts = defaultdict(lambda: 0, dict())
    concept_avg_vals = defaultdict(lambda: 0, dict())
    concept_avg_logvals = defaultdict(lambda: 0, dict())

    n = 0
    for batch in tqdm(dl, total=n_batches):
        if n == n_batches:
            break
        res += explain_batch(
            batch,
            attribution_generator,
            method=method,
            class_index=class_index,
            normalized=normalized,
            remove_fillup=remove_fillup,
        )
        n += 1

        for r in res:
            # indiv_set = set()
            indiv_vals = defaultdict(lambda: [], dict())
            for c, v in zip(r["concept_names"], r["exp"]):
                if not c in indiv_vals:
                    # indiv_set.add(c)
                    concept_indiv_counts[c] += 1
                    indiv_vals[c] += [v]
                concept_counts[c] += 1

            for c in indiv_vals:
                concept_avg_vals[c] += np.mean(indiv_vals[c])
                concept_avg_logvals[c] += np.mean(
                    np.log1p(const_mult * np.array(indiv_vals[c]))
                )

    all_v = dict()
    for concept in concept_counts:
        all_v[concept] = {
            "count": concept_counts[concept],
            "indiv_count": concept_indiv_counts[concept],
            "avg_val": concept_avg_vals[concept] / concept_indiv_counts[concept],
            "avg_logval": concept_avg_logvals[concept] / concept_indiv_counts[concept],
        }
    all_v = pd.DataFrame(all_v).T
    all_v.avg_val *= const_mult
    all_v.sort_values(by="avg_val", inplace=True, ascending=False)

    return all_v


def explain_batch(
    batch,
    attribution_generator,
    class_index=-1,
    method="transformer_attribution",
    normalized=False,
    remove_fillup=True,
):
    # data = batch['data']
    event = batch["event"]
    concept_names = np.array(batch["concept_names"]).T

    res = []

    for i in tqdm(range(len(event))):
        data = {
            "data": batch["data"][i : i + 1],
            "ages": batch["ages"][i : i + 1],
            "values": batch["values"][i : i + 1],
            "event": batch["event"][i : i + 1],
        }
        if remove_fillup:
            # print(data["data"].shape)
            # print(data['data'][0], type(data['data'][0]))
            # print(concept_names[i], type(concept_names[i]))
            data["data"] = torch.stack(
                [
                    x
                    for x, concept_name in zip(data["data"][0], concept_names[i])
                    if concept_name != "FILL-UP-CONCEPTS"
                ]
            ).unsqueeze(0)
            data["ages"] = torch.stack(
                [
                    x
                    for x, concept_name in zip(data["ages"][0], concept_names[i])
                    if concept_name != "FILL-UP-CONCEPTS"
                ]
            ).unsqueeze(0)
            data["values"] = torch.stack(
                [
                    x
                    for x, concept_name in zip(data["values"][0], concept_names[i])
                    if concept_name != "FILL-UP-CONCEPTS"
                ]
            ).unsqueeze(0)

        exp = generate_explanation(
            data,
            attribution_generator,
            method=method,
            class_index=class_index,
            normalized=normalized,
        ).numpy()
        # print(exp)
        # print(exp.shape)
        if method in ["transformer_attribution", "rollout"]:
            exp = exp[0]
        res.append(
            {
                # dict only if all equal
                # 'exp': dict(list(zip(exp, concept_names[i]))),
                # 'exp': dict(list(zip(concept_names[i], exp*1e6))),
                "exp": exp,
                "concept_names": concept_names[i],
                # 'exp': list(zip(exp, concept_names[i])),
                "event": event[i, class_index].item(),
            }
        )
    return res


def generate_explanation(
    input,
    attribution_generator,
    method="transformer_attribution",
    class_index=None,
    normalized=True,
):
    """
    methods:
    - transformer_attribution
    - full --> not working, since using pos embed?
    - rollout
    - last_layer
    - last_layer_attn
    - second_layer
    """
    transformer_attribution = attribution_generator.generate_LRP(
        input, method=method, index=class_index
    ).detach()
    if normalized:
        transformer_attribution = (
            transformer_attribution - transformer_attribution.min()
        ) / (transformer_attribution.max() - transformer_attribution.min())

    return transformer_attribution
