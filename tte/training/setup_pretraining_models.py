import torch
import h5py

from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from time import time

# from custom_attention_model import get_simple_transformer, setup_set_dataloader
from tte.training.pretrain_surv_models import (
    train_custom_model,
    eval_survival_model,
    # setup_df_dataloader,
    setup_h5_dataloader,
)
from tte.models.base_networks import setup_network
from tte.data.set_data import setup_set_dataloader
from tte.models.transformers import get_simple_transformer


def run_custom_survival_model(
    train,
    val,
    test,
    h5fn,
    # survival,
    warmup_epochs=1,
    min_epochs=None,
    epochs=10,
    early_stopping_patience=10,
    batch_size=512,
    norm="layer",
    hidden_layers=3,
    hidden_size=4096,
    wandb_project="survival",
    wandb_name=None,
    seed=42,
    use_gpu=None,
    **kwargs,
):
    pl.seed_everything(seed, workers=True)
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    h5f = h5py.File(h5fn, "r")
    tl = setup_h5_dataloader(h5f, train, batch_size=batch_size, shuffle=True)
    vl = setup_h5_dataloader(h5f, val, batch_size=batch_size, shuffle=False)
    ttl = setup_h5_dataloader(h5f, test, batch_size=batch_size, shuffle=False)

    # tl = setup_dataloader(train, survival, batch_size=batch_size, shuffle=True)
    # vl = setup_dataloader(val, survival, batch_size=batch_size, shuffle=False)
    # ttl = setup_dataloader(test, survival, batch_size=batch_size, shuffle=False)

    input_size = tl.dataset.dim
    output_size = train.shape[1] // 2
    net = setup_network(
        input_size,
        output_size,
        norm=norm,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        final_bias=False,
    )
    net = train_custom_model(
        net,
        tl,
        vl,
        warmup_epochs=warmup_epochs,
        min_epochs=min_epochs,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        use_gpu=use_gpu,
        **kwargs,
    )

    # _, train_cindices = eval_survival_model(net, tl)
    try:
        val_loss, val_cindices = eval_survival_model(
            net, vl, loss_type="cox", use_gpu=use_gpu
        )
    except ValueError:
        print("val loss failed, skipping")
        val_loss = np.nan
        val_cindices = {target: np.nan for target in vl.dataset.target_names}
    try:
        test_loss, test_cindices, test_preds = eval_survival_model(
            net, ttl, loss_type="cox", return_preds=True, use_gpu=use_gpu
        )
    except ValueError:
        print("test loss failed, skipping")
        test_loss = np.nan
        test_cindices = {target: np.nan for target in vl.dataset.target_names}
        test_preds = pd.DataFrame(
            np.nan,
            index=ttl.dataset.get_durations().index,
            columns=ttl.dataset.target_names,
        )

    # if hidden_layers == 0:
    #     # linear model, can actually interpret model weights
    #     weights = pd.DataFrame(
    #         net.weight.detach().numpy(),
    #         columns=train.columns,
    #         index=tl.dataset.target_names,
    #     ).T
    #     # print(weights.head(15))
    #     # print(weights.tail(15))
    #     net = (net, weights)

    print("concordance indices")
    print("{:<20} {:<10} {:<10}".format("Target", "Val", "Test"))
    # print("{:<20} {:<10} {:<10} {:<10}".format("Target", "Train", "Val", "Test"))
    for target in val_cindices:
        print(
            # "{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            "{:<20} {:<10.4f} {:<10.4f}".format(
                target,
                # train_cindices[target],
                val_cindices[target],
                test_cindices[target],
            )
        )
    return (
        net,
        tl.dataset.input_names,
        val_loss,
        test_loss,
        val_cindices,
        test_cindices,
        (ttl.dataset.get_durations(), ttl.dataset.get_events(), test_preds),
    )


def run_attention_survival_model(
    train,
    val,
    test,
    records_h5fns,
    embeddings_h5fns,
    warmup_epochs=1,
    min_epochs=10,
    epochs=10,
    early_stopping_patience=10,
    batch_size=512,
    n_token_fill_or_sample=15,
    fill_value=0,
    num_workers=8,
    num_heads=4,
    depth=4,
    embed_dim=64,
    wandb_project="survival",
    wandb_name=None,
    seed=42,
    use_gpu=None,
    binarize=False,
    trunc_pad_eval=False,
    empty_concepts=dict(),
    age_embed_config=None,
    val_embed_config=None,
    positional_constant=False,
    positional_embeddings_normalized=False,
    learnable_positional_factor=False,
    input_dim=None,
    loss_type="cox",
    aft_sigma=1.2,
    aux_endpoints=None,
    aux_weight=0.0,
    **kwargs,
):
    if aux_weight == 0:
    # if aux_endpoints is None:
        aux_endpoints = {'train': None, 'val': None, 'test': None}
        use_aux = False
    else:
        use_aux = True
    if batch_size > 512:
        mult = batch_size / 512
        print(f'batch size {batch_size} is too large for num_workers {num_workers}, adjusting')
        num_workers = int(num_workers / mult)
        print(f'new num_workers: {num_workers}')
    if num_workers == 1:
        print(f'num_workers is 1, setting to 0')
        num_workers = 0
        
    pl.seed_everything(seed, workers=True)
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    print("setting up dataloaders...")
    tl = setup_set_dataloader(
        records_h5fns=records_h5fns,
        embeddings_h5fns=embeddings_h5fns,
        survival=train,
        n_token_fill_or_sample=n_token_fill_or_sample,
        fill_value=fill_value,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        binarize=binarize,
        empty_concepts=empty_concepts,
        input_dim=input_dim,
        aux_survival=aux_endpoints['train'],
    )
    vl = setup_set_dataloader(
        records_h5fns=records_h5fns,
        embeddings_h5fns=embeddings_h5fns,
        survival=val,
        n_token_fill_or_sample=n_token_fill_or_sample,
        fill_value=fill_value,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        binarize=binarize,
        empty_concepts=empty_concepts,
        input_dim=input_dim,
        aux_survival=aux_endpoints['val'],
    )
    ttl = setup_set_dataloader(
        records_h5fns=records_h5fns,
        embeddings_h5fns=embeddings_h5fns,
        survival=test,
        n_token_fill_or_sample=n_token_fill_or_sample if trunc_pad_eval else None,
        fill_value=fill_value,
        batch_size=batch_size if trunc_pad_eval else 1,
        shuffle=False,
        num_workers=num_workers if trunc_pad_eval else 0,
        binarize=binarize,
        empty_concepts=empty_concepts,
        input_dim=input_dim,
        aux_survival=   aux_endpoints['test'],
    )

    print("setting up network...")
    output_size = tl.dataset.n_targets
    if loss_type == "aft":
        output_size = 2 * output_size
    if use_aux and loss_type == 'cox':
        output_size = (output_size, aux_endpoints['train'].shape[1]//2)
    elif use_aux and loss_type == 'aft':
        raise NotImplementedError('auxiliary endpoints not implemented for AFT loss')
    net = get_simple_transformer(
        input_dim=tl.dataset.token_size,
        output_size=output_size,
        num_heads=num_heads,
        depth=depth,
        embed_dim=embed_dim,
        age_embed_config=age_embed_config,
        val_embed_config=val_embed_config,
        positional_constant=positional_constant,
        learnable_positional_factor=learnable_positional_factor,
        positional_embeddings_normalized=positional_embeddings_normalized,
    )
    # print('\n\n\n')
    # print('*'*100)
    # test_loss, test_cindices, test_preds = eval_survival_model(
    #     net, ttl, return_preds=True, use_gpu=use_gpu
    # )
    # print('*'*100)
    # print('\n\n\n')

    start_time = time()
    net = train_custom_model(
        net,
        tl,
        vl,
        warmup_epochs=warmup_epochs,
        min_epochs=min_epochs,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        use_gpu=use_gpu,
        loss_type=loss_type,
        aft_sigma=aft_sigma,
        aux_weight=aux_weight,
        **kwargs,
    )
    end_time = time()
    full_time = end_time - start_time
    print(f"finished training; took {full_time:1f}s, evaluating...")

    # _, train_cindices = eval_survival_model(net, tl)
    try:
        val_loss, val_cindices = eval_survival_model(
            net, vl, loss_type=loss_type, use_gpu=use_gpu, use_aux=use_aux
        )
    except ValueError:
        print("val loss failed, skipping")
        val_loss = np.inf
        val_cindices = {target: 0.0 for target in vl.dataset.target_names}
    try:
        test_loss, test_cindices, test_preds = eval_survival_model(
            net, ttl, loss_type=loss_type, return_preds=True, use_gpu=use_gpu, use_aux=use_aux
        )
    except ValueError:
        print("test loss failed, skipping")
        test_loss = np.inf
        test_cindices = {target: 0.0 for target in vl.dataset.target_names}
        test_preds = pd.DataFrame(
            np.nan,
            index=ttl.dataset.get_durations().index,
            columns=ttl.dataset.target_names,
        )

    print("concordance indices")
    print("{:<20} {:<10} {:<10}".format("Target", "Val", "Test"))
    # print("{:<20} {:<10} {:<10} {:<10}".format("Target", "Train", "Val", "Test"))
    for target in val_cindices:
        print(
            # "{:<20} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            "{:<20} {:<10.4f} {:<10.4f}".format(
                target,
                # train_cindices[target],
                val_cindices[target],
                test_cindices[target],
            )
        )
    return (
        net,
        val_loss,
        test_loss,
        val_cindices,
        test_cindices,
        (ttl.dataset.get_durations(), ttl.dataset.get_events(), test_preds),
        full_time,
    )
