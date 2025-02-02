import pandas as pd


def quick_default_run():
    sweep_params = pd.DataFrame(
        [
            {
                "warmup_epochs": 0,
                "min_epochs": 1,
                "epochs": 1,
                "early_stopping_patience": -1,
                "batch_size": 512,
                "lr": 0.001,
                "weight_decay": 0.0001,
                "n_token_fill_or_sample": 12,
                "fill_value": 0,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 32,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?ingredient_exposure",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": "cox",
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
        ]
    )
    embed_params = pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "none",
                "input_type": "none",
                "pca_dim": 0,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
        ]
    )
    return sweep_params, embed_params


def small_opt():
    LRS = [1e-5, 1e-4, 1e-3]
    WDS = [1e-4]
    # BSS = [256, 512, 1024, 2048]
    BSS = [512]
    OTHER_SETTINGS = [
        {
            "warmup_epochs": 2,
            "min_epochs": 30,
            "epochs": 30,
            "early_stopping_patience": -1,
            "input_dim": None,
            "loss_type": "cox",
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
        },
    ]
    return pd.DataFrame(
        [
            {
                **other,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
            }
            for other in OTHER_SETTINGS
            for lr in LRS
            for wd in WDS
            for bs in BSS
        ]
    )


def optimization():
    LRS = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    WDS = [1e-4]
    # BSS = [256, 512, 1024, 2048]
    BSS = [512]
    OTHER_SETTINGS = [
        {
            "warmup_epochs": 2,
            "min_epochs": 30,
            "epochs": 30,
            "early_stopping_patience": -1,
        },
        {
            "warmup_epochs": 2,
            "min_epochs": 5,
            "epochs": 30,
            "early_stopping_patience": 3,
        },
    ]
    return pd.DataFrame(
        [
            {
                **other,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
            }
            for other in OTHER_SETTINGS
            for lr in LRS
            for wd in WDS
            for bs in BSS
        ]
    )


def default_architecture():
    return pd.DataFrame(
        [
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 64,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
        ]
    )


def ext_architecture():
    return pd.DataFrame(
        [
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 64,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 16,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 256,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 64,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 6,
                "embed_dim": 16,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 6,
                "embed_dim": 256,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
        ]
    )


def default_embed_params():
    return pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "none",
                "input_type": "none",
                "pca_dim": 0,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
        ]
    )


def default_arch_params():
    return pd.DataFrame(
        [
            {
                "n_token_fill_or_sample": 64,
                "num_heads": 8,
                "depth": 4,
                "embed_dim": 64,
                "fill_value": 0,
                "binarize": False,
                "trunc_pad_eval": True,
                "sources": "condition_occurrence?procedure_occurrence",
                "positional_constant": False,
                "use_age_embeddings": False,
                "use_val_embeddings": False,
                "positional_embeddings_normalized": False,
                "learnable_positional_factor": False,
                "input_dim": None,
                "loss_type": 'cox',
                "optimizer": "adam",
                "low_bit": False,
                "num_workers": 8,
            },
        ]
    )


def extensive_embed_params():
    return pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "none",
                "input_type": "none",
                "pca_dim": 0,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
            {
                "provider": "openai",
                "model": "none",
                "input_type": "none",
                "pca_dim": 64,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
            {
                "provider": "cohere",
                "model": "embed-english-v3.0",
                "input_type": "classification",
                "pca_dim": 0,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
            {
                "provider": "cohere",
                "model": "embed-english-v3.0",
                "input_type": "classification",
                "pca_dim": 64,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
            {
                "provider": "cohere",
                "model": "embed-english-light-v3.0",
                "input_type": "classification",
                "pca_dim": 0,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
            {
                "provider": "cohere",
                "model": "embed-english-light-v3.0",
                "input_type": "classification",
                "pca_dim": 64,
                # 'sources': 'condition_occurrence?procedure_occurrence',
            },
        ]
    )


def get_embed_params_sweep():
    A = default_architecture()
    O = optimization()
    E = extensive_embed_params()
    sweep_params = A.merge(O, how="cross")
    return sweep_params, E


def get_arch_sweep():
    A = ext_architecture()
    E = default_embed_params()
    O = optimization()
    sweep_params = A.merge(O, how="cross")
    return sweep_params, E


def get_opt_sweep():
    A = default_architecture()
    E = default_embed_params()
    O = optimization()
    sweep_params = A.merge(O, how="cross")
    return sweep_params, E


def standard_sweep():
    E = default_embed_params()
    O = small_opt()
    A = default_architecture()
    sweep_params = A.merge(O, how="cross")
    return sweep_params, E


# sweep_params, embed_params = get_embed_params_sweep()
# sweep_params, embed_params = get_arch_sweep()
# sweep_params, embed_params = get_opt_sweep()

sweep_params, embed_params = quick_default_run()
# sweep_params, embed_params = standard_sweep()

sweep_params = sweep_params[sorted(sweep_params.columns)]
embed_params = embed_params[sorted(embed_params.columns)]
