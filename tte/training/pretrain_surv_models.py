import numpy as np
import socket
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from box import Box
from lifelines.utils import concordance_index

# ugly hack to prevent bnb & torch cuda warnings that pollute snakemake outputs
import warnings

warnings.filterwarnings("ignore")
if torch.cuda.is_available():
    import bitsandbytes as bnb

from pycox.models.loss import cox_ph_loss
from pytorch_lightning.callbacks import EarlyStopping, ModelSummary, LearningRateMonitor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import WandbLogger, CSVLogger


DEBUG = False
DEBUG_DATA_LOADING = False
# DEBUG_DATA_LOADING = True
# DEBUG = True
# LOW_BIT = True
# LOW_BIT = False
LOG_STEP = 50


def setup_h5_dataloader(h5f, survival, batch_size=512, shuffle=True):
    sources = sorted(h5f.keys())
    h5iids = [
        int(x) for x in h5f[sources[0]].keys() if not x in ["columns", "description"]
    ]
    ind = np.intersect1d(h5iids, survival.index)
    E = survival.loc[ind, [col for col in survival.columns if col.endswith("_event")]]
    D = survival.loc[
        ind, [col for col in survival.columns if col.endswith("_duration")]
    ]
    ds = SurvivalH5Data(
        h5f=h5f,
        duration=D.values,
        event=E.values,
        target_names=[col[:-6] for col in E.columns],
        index_names=ind,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dl


def setup_df_dataloader(df, survival, batch_size=512, shuffle=True):
    ind = df.index.intersection(survival.index)
    E = survival.loc[ind, [col for col in survival.columns if col.endswith("_event")]]
    D = survival.loc[
        ind, [col for col in survival.columns if col.endswith("_duration")]
    ]
    X = df.loc[ind]
    ds = SurvivalData(
        X.values,
        D.values,
        E.values,
        target_names=[col[:-6] for col in E.columns],
        index_names=X.index,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dl


class SurvivalH5Data(Dataset):
    def __init__(self, h5f, duration, event, target_names, index_names):
        self.sources = sorted(h5f.keys())
        self.h5f = h5f
        self.concept_columns = {
            source: h5f[source]["columns"][()] for source in self.sources
        }
        self.source_shapes = {
            source: len(cols) for source, cols in self.concept_columns.items()
        }
        self.source_types = {
            source: h5f[source]["description"][()].decode() for source in self.sources
        }
        self.duration = torch.from_numpy(duration).float()
        self.event = torch.from_numpy(event).float()
        self.index_names = index_names
        self.target_names = target_names
        self.dim = sum(self.source_shapes.values())
        self.input_names = np.concatenate(
            [
                [x.decode() for x in self.concept_columns[source]]
                for source in self.sources
            ]
        )

    def __len__(self):
        return len(self.event)

    def load_omop_item(self, iid):
        tensors = []
        for source in self.sources:
            if self.source_types[source] == "direct":
                tensor = torch.from_numpy(self.h5f[source][str(iid)][()]).to(
                    torch.float
                )
            elif self.source_types[source] == "binary_index":
                local_concept_idx = self.h5f[source][str(iid)][()]
                tensor = torch.zeros(self.source_shapes[source], dtype=torch.float)
                tensor[local_concept_idx] = 1.0
            else:
                raise Exception("unknown source type")
            tensors.append(tensor)
        tensor = torch.cat(tensors)
        return tensor

    def __getitem__(self, idx):
        iid = self.index_names[idx]
        tensor = self.load_omop_item(iid)

        return {
            "data": tensor,
            "duration": self.duration[idx],
            "event": self.event[idx],
        }

    def get_events(self):
        return pd.DataFrame(
            index=self.index_names,
            columns=self.target_names,
            data=self.event.numpy(),
        )

    def get_durations(self):
        return pd.DataFrame(
            index=self.index_names,
            columns=self.target_names,
            data=self.duration.numpy(),
        )


class SurvivalData(Dataset):
    def __init__(self, x, duration, event, target_names=None, index_names=None):
        self.x = torch.from_numpy(x).float()
        self.duration = torch.from_numpy(duration).float()
        self.event = torch.from_numpy(event).float()
        if index_names is None:
            self.index_names = [str(i) for i in range(x.shape[0])]
        else:
            self.index_names = index_names
        if target_names is None:
            self.target_names = [str(i) for i in range(event.shape[1])]
        else:
            self.target_names = target_names

    def get_events(self):
        return pd.DataFrame(
            index=self.index_names,
            columns=self.target_names,
            data=self.event.numpy(),
        )

    def get_durations(self):
        return pd.DataFrame(
            index=self.index_names,
            columns=self.target_names,
            data=self.duration.numpy(),
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "data": self.x[idx],
            "duration": self.duration[idx],
            "event": self.event[idx],
        }


def train_custom_model(
    net,
    tl,
    vl,
    min_epochs=None,
    epochs=10,
    early_stopping_patience=10,
    wandb_project="omop_survival",
    wandb_name=None,
    use_gpu=None,
    log_params=dict(),
    loss_type="cox",
    # aft_sigma=1.2,
    aux_weight=0.0,
    **kwargs,
):
    model = CustomCoxPH(
        net,
        epochs=epochs,
        num_dl_batches=len(tl),
        target_names=tl.dataset.target_names,
        loss_type=loss_type,
        aux_weight=aux_weight,
        # aft_sigma=aft_sigma,
        **kwargs,
    )
    callbacks = [
        ModelSummary(max_depth=3),
        # LearningRateMonitor(
        #     log_momentum=True,
        # ),
    ]
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor="full_eval_loss", patience=early_stopping_patience)
        )
    # log_params['host'] = socket.gethostname()
    # if isinstance(log_params, Box):
    #     log_params = log_params.to_dict()
    # logger = CSVLogger(save_dir="_training_logs_manual/")
    # logger.log_hyperparams({**log_params})
    # if DEBUG:
    #     logger = None
    # else:
    #     logger = CSVLogger(save_dir="_training_logs_manual/")
    #     # logger = WandbLogger(project=wandb_project, name=wandb_name, save_dir="/tmp")
    #     logger.log_hyperparams({**log_params})
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
        min_epochs=min_epochs,
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else "auto",
        # log_every_n_steps=LOG_STEP,
        max_steps=25 if DEBUG else -1,
        # detect_anomaly=True,
        # gradient_clip_val=1.0,
        # logger=logger,
        logger=False,
        enable_checkpointing=False,
        # precision='16' if LOW_BIT else None,
    )
    trainer.fit(model, tl, vl)
    return net.eval()


@torch.no_grad()
def eval_survival_model(
    model,
    dataloader,
    loss_type,
    return_preds=False,
    use_gpu=False,
    use_aux=False,
):
    assert loss_type in ["cox", "aft"]
    output_multiplier = -1 if loss_type == "cox" else 1
    device = "cuda" if use_gpu else "cpu"
    model.eval().to(device)
    log_partial_hazards = []
    E = []
    T = []
    if use_aux:
        aux_log_partial_hazards = []
        aux_E = []
        aux_T = []
    count = 0
    for batch in tqdm(dataloader):
        if "ages" in batch.keys() and "values" in batch.keys():
            x = {
                "data": batch["data"].to(device),
                "ages": batch["ages"].to(device),
                "values": batch["values"].to(device),
            }
        else:
            x = batch["data"].to(device)
        output = model(x)
        if use_aux:
            output, aux_output = output
            aux_log_partial_hazards.append(aux_output.to("cpu"))
            aux_E.append(batch["aux_event"])
            aux_T.append(batch["aux_duration"])
        log_partial_hazards.append(output.to("cpu"))
        E.append(batch["event"])
        T.append(batch["duration"])
        count += 1
        if DEBUG and count >= 10:
            break
    log_partial_hazards = torch.cat(log_partial_hazards)
    T = torch.cat(T)
    E = torch.cat(E)
    if use_aux:
        aux_log_partial_hazards = torch.cat(aux_log_partial_hazards)
        aux_T = torch.cat(aux_T)
        aux_E = torch.cat(aux_E)
    model.to("cpu")

    if loss_type == "cox":
        total_loss, _ = multi_cox_ph_loss(log_partial_hazards, T, E)
        if use_aux:
            aux_loss, _ = multi_cox_ph_loss(aux_log_partial_hazards, aux_T, aux_E)
    else:
        k = log_partial_hazards.shape[1] // 2
        mu, sigma = log_partial_hazards[:, :k], log_partial_hazards[:, k:].exp()
        total_loss, _ = multi_gaussian_aft_loss(mu, sigma, T, E)
        log_partial_hazards = mu
        if use_aux:
            raise NotImplementedError("aux loss not implemented for AFT")

    c_indices = dict()

    for i in range(E.shape[1]):
        name = dataloader.dataset.target_names[i]
        try:
            c = concordance_index(
                T[:, i], output_multiplier * log_partial_hazards[:, i], E[:, i]
            )
        except ZeroDivisionError:
            c = np.nan
        c_indices[name] = c
    if use_aux:
        for i in range(aux_E.shape[1]):
            name = dataloader.dataset.aux_target_names[i]
            try:
                c = concordance_index(
                    aux_T[:, i],
                    output_multiplier * aux_log_partial_hazards[:, i],
                    aux_E[:, i],
                )
            except ZeroDivisionError:
                c = np.nan
            c_indices[f"aux_{name}"] = c
    if return_preds:
        log_partial_hazards = pd.DataFrame(
            index=dataloader.dataset.index_names,
            columns=dataloader.dataset.target_names,
            data=log_partial_hazards.numpy(),
        )
        return total_loss, c_indices, log_partial_hazards
    else:
        return total_loss, c_indices


def multi_cox_ph_loss(log_partial_hazards, duration, event):
    losses = []
    if event.ndim == 1:
        log_partial_hazards = log_partial_hazards.view(-1, 1)
        duration = duration.view(-1, 1)
        event = event.view(-1, 1)
    for i in range(log_partial_hazards.shape[1]):
        n_events = event[:, i].sum()
        if n_events == 0:
            loss = torch.tensor(
                0.0, device=log_partial_hazards.device, dtype=log_partial_hazards.dtype
            )
        else:
            loss = cox_ph_loss(log_partial_hazards[:, i], duration[:, i], event[:, i])
        losses.append(loss)

    total_loss = torch.stack(losses).sum()
    return total_loss, losses


# def multi_gaussian_aft_loss(y_pred, y_true, censored, sigma, eps=1e-6):
# TODO: make sigma more flexible?
def multi_gaussian_aft_loss(mu, sigma, duration, event, eps=1e-6):
    losses = []
    if event.ndim == 1:
        mu = mu.view(-1, 1)
        sigma = sigma.view(-1, 1)
        duration = duration.view(-1, 1)
        event = event.view(-1, 1)
    for i in range(mu.shape[1]):
        loss = single_gaussian_aft_loss(
            mu=mu[:, i],
            sigma=sigma[:, i],
            y_true=duration[:, i],
            censored=1 - event[:, i],
            eps=eps,
        )
        losses.append(loss)
    total_loss = torch.stack(losses).sum()
    return total_loss, losses


# def multi_gaussian_aft_loss(y_pred, duration, event, sigma, eps=1e-6):
#     losses = []
#     if event.ndim == 1:
#         y_pred = y_pred.view(-1, 1)
#         duration = duration.view(-1, 1)
#         event = event.view(-1, 1)
#     for i in range(y_pred.shape[1]):
#         loss = single_gaussian_aft_loss(
#             y_pred[:, i], duration[:, i], 1 - event[:, i], sigma=sigma, eps=eps
#         )
#         losses.append(loss)
#     total_loss = torch.stack(losses).sum()
#     return total_loss, losses


# def single_gaussian_aft_loss(y_pred, y_true, censored, sigma, eps=1e-6):
#     # y_pred: predicted log(T)
#     # y_true: actual log(T)
#     # censored: binary indicator (0 if uncensored, 1 if censored)
#     # sigma: error distribution scale

#     mu = y_pred
#     if mu.isnan().any():
#         return torch.tensor(torch.nan, device=mu.device, dtype=mu.dtype)
#     if isinstance(sigma, float):
#         sigma = torch.tensor(sigma, device=mu.device, dtype=mu.dtype)
#     sigma = torch.clamp(sigma, min=eps)

#     # Creating a normal distribution to use its log_prob method
#     normal_dist_uncensored = torch.distributions.Normal(mu[censored == 0], sigma)
#     normal_dist_censored = torch.distributions.Normal(mu[censored == 1], sigma)

#     # uncensored:
#     y_true_uncensored = y_true[censored == 0]
#     # Log PDF for uncensored data
#     log_pdf = normal_dist_uncensored.log_prob(y_true_uncensored)

#     # censored:
#     y_true_censored = y_true[censored == 1]
#     # Log Survival function (log SF) for censored data can be computed as the log complementary CDF
#     log_sf = torch.log(
#         1 - normal_dist_censored.cdf(y_true_censored) + eps
#     )  # Adding a small constant to avoid log(0)

#     # Apply masks for censored and uncensored data
#     # uncensored_loss = log_pdf * (1 - censored)
#     # censored_loss = log_sf * censored

#     # Sum and normalize the losses
#     # total_loss = -torch.mean(uncensored_loss + censored_loss)  # Negative sign for maximization

#     total_loss = -(log_pdf.sum() + log_sf.sum()) / len(y_true)


#     return total_loss
def single_gaussian_aft_loss(mu, sigma, y_true, censored, eps=1e-6):
    if mu.isnan().any() or sigma.isnan().any():
        return torch.tensor(torch.nan, device=mu.device, dtype=mu.dtype)
    # if isinstance(sigma, float):
    #     sigma = torch.tensor(sigma, device=mu.device, dtype=mu.dtype)
    sigma = torch.clamp(sigma, min=eps)

    normal_dist_uncensored = torch.distributions.Normal(
        mu[censored == 0],
        sigma[censored == 0],
    )
    normal_dist_censored = torch.distributions.Normal(
        mu[censored == 1],
        sigma[censored == 1],
    )

    y_true_uncensored = y_true[censored == 0]
    log_pdf = normal_dist_uncensored.log_prob(y_true_uncensored)

    y_true_censored = y_true[censored == 1]
    log_sf = torch.log(1 - normal_dist_censored.cdf(y_true_censored) + eps)
    total_loss = -(log_pdf.sum() + log_sf.sum()) / len(y_true)

    return total_loss


class CustomCoxPH(pl.LightningModule):
    def __init__(
        self,
        net,
        target_names=None,
        loss_type="cox",
        # aft_sigma=1.2,
        # aft_sigma_trainable=False,
        lr=0.001,
        weight_decay=0.00,
        optimizer="adam",
        warmup_epochs=0,
        momentum=0.9,
        low_bit=False,
        aux_weight=0.0,
        **kwargs,
    ):
        super().__init__()
        self.net = net
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.target_names = target_names
        self.loss_type = loss_type
        self.aux_weight = aux_weight
        assert self.loss_type in ["cox", "aft"]
        # assert aft_sigma > 0
        # if aft_sigma_trainable:
        #     self.aft_log_sigma = nn.Parameter(
        #         torch.tensor(aft_sigma, device=self.device).log()
        #     )
        # else:
        #     self.register_buffer(
        #         "aft_log_sigma", torch.tensor(aft_sigma, device=self.device).log()
        #     )

    @property
    def device(self):
        return next(self.net.parameters()).device

    # def log(self, name, value, **kwargs):
    #     super().log(name, value, **kwargs)
    #     logger = self.trainer.logger
    #     if isinstance(logger, pl.loggers.CSVLogger):
    #         csv_file = f"{logger.log_dir}/metrics.csv"
    #         out_dir = logger.log_dir
    #         if os.path.exists(csv_file) and self.trainer.global_step % LOG_STEP == 0:
    #             self.manual_plotting(csv_file, out_dir)

    def manual_plotting(self, inp, out_dir):
        try:
            train_step = pd.read_csv(inp, usecols=["step", "train_loss_step"]).dropna()
            plt.plot(
                train_step["step"],
                train_step["train_loss_step"],
                label="train_loss_step",
            )
            plt.savefig(f"{out_dir}/train_loss_step.png")
            plt.close()
        except Exception as e:
            print(f"Error in plotting: {e}")
        try:
            train_epoch = pd.read_csv(
                inp, usecols=["epoch", "train_loss_epoch"]
            ).dropna()
            plt.plot(
                train_epoch["epoch"],
                train_epoch["train_loss_epoch"],
                label="train_loss_epoch",
            )
            plt.savefig(f"{out_dir}/train_loss_epoch.png")
            plt.close()
        except Exception as e:
            print(f"Error in plotting: {e}")
        try:
            val_epoch = pd.read_csv(inp, usecols=["epoch", "eval_loss"]).dropna()
            plt.plot(val_epoch["epoch"], val_epoch["eval_loss"], label="eval loss")
            plt.savefig(f"{out_dir}/val_loss_epoch.png")
            plt.close()
        except Exception as e:
            print(f"Error in plotting: {e}")
        try:
            lr_step = pd.read_csv(inp, usecols=["step", "lr-Adam"]).dropna()
            plt.plot(lr_step["step"], lr_step["lr-Adam"], label="lr-Adam")
            plt.savefig(f"{out_dir}/lr-Adam.png")
            plt.close()
        except Exception as e:
            print(f"Error in plotting: {e}")
        try:
            lr_step = pd.read_csv(inp, usecols=["step", "lr-AdamW"]).dropna()
            plt.plot(lr_step["step"], lr_step["lr-AdamW"], label="lr-AdamW")
            plt.savefig(f"{out_dir}/lr-AdamW.png")
            plt.close()
        except Exception as e:
            print(f"Error in plotting: {e}")

    def forward(self, x):
        return self.net(x)

    def loss(self, y_pred, y):
        if self.loss_type == "cox":
            loss, individual_losses = multi_cox_ph_loss(
                y_pred, y["duration"], y["event"]
            )
        elif self.loss_type == "aft":
            k = y_pred.shape[1] // 2
            mu, sigma = y_pred[:, :k], y_pred[:, k:].exp()
            # sigma = self.aft_log_sigma.exp()
            # print(f"current sigma: {sigma}")
            loss, individual_losses = multi_gaussian_aft_loss(
                mu=mu,
                sigma=sigma,
                # y_pred,
                duration=y["duration"],
                event=y["event"],
                # sigma,
            )
        return loss, individual_losses

    def validation_step(self, batch, batch_idx):
        if DEBUG_DATA_LOADING:
            return None
        if "ages" in batch.keys() and "values" in batch.keys():
            x = {
                "data": batch["data"],
                "ages": batch["ages"],
                "values": batch["values"],
            }
        else:
            x = batch["data"]
        log_partial_hazards = self(x)
        if self.loss_type == "cox" and batch["event"].sum() == 0:
            return
        if self.aux_weight > 0:
            log_partial_hazards, aux_log_partial_hazards = log_partial_hazards
            aux_loss, _ = self.loss(
                aux_log_partial_hazards,
                {"duration": batch["aux_duration"], "event": batch["aux_event"]},
            )
            self.log(
                "aux_eval_loss", aux_loss, prog_bar=True, on_epoch=True, on_step=True
            )
        else:
            aux_loss = 0.0
        self.validation_step_outputs.append(
            {
                "log_partial_hazard": log_partial_hazards.cpu(),
                "duration": batch["duration"].cpu(),
                "event": batch["event"].cpu(),
            }
        )
        loss, individual_losses = self.loss(log_partial_hazards, batch)
        self.log("eval_loss_wo_aux", loss, prog_bar=True, on_epoch=True, on_step=True)
        loss += self.aux_weight * aux_loss
        self.log("eval_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for i, l in enumerate(individual_losses):
            name = self.target_names[i] if self.target_names is not None else i
            self.log(
                f"eval_loss_{name}",
                l,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def on_validation_epoch_end(self):
        if DEBUG_DATA_LOADING:
            return None
        output_multiplier = -1 if self.loss_type == "cox" else 1
        P = []
        E = []
        T = []
        for output in self.validation_step_outputs:
            P.append(output["log_partial_hazard"])
            E.append(output["event"])
            T.append(output["duration"])
        P = torch.cat(P).numpy()
        if self.loss_type == "aft":
            k = P.shape[1] // 2
            P = P[:, :k]
        T = torch.cat(T).numpy()
        E = torch.cat(E).numpy()

        for i in range(P.shape[1]):
            name = self.target_names[i] if self.target_names is not None else i

            non_nan = ~np.isnan(P[:, i])
            try:
                c = concordance_index(
                    T[non_nan, i], output_multiplier * P[non_nan, i], E[non_nan, i]
                )
            except ZeroDivisionError:
                c = np.nan

            self.log(f"eval_cindex_{name}", c)

        loss, individual_losses = self.loss(
            torch.from_numpy(P).to(self.device),
            {
                "duration": torch.from_numpy(T).to(self.device),
                "event": torch.from_numpy(E).to(self.device),
            },
        )
        self.log("full_eval_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for i, l in enumerate(individual_losses):
            name = self.target_names[i] if self.target_names is not None else i
            self.log(
                f"full_eval_loss_{name}",
                l,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        if DEBUG_DATA_LOADING:
            return None

        if "ages" in batch.keys() and "values" in batch.keys():
            x = {
                "data": batch["data"],
                "ages": batch["ages"],
                "values": batch["values"],
            }
        else:
            x = batch["data"]
        if self.loss_type == "cox" and batch["event"].sum() == 0:
            return
        log_partial_hazards = self(x)
        if self.aux_weight > 0:
            log_partial_hazards, aux_log_partial_hazards = log_partial_hazards
            aux_loss, _ = self.loss(
                aux_log_partial_hazards,
                {"duration": batch["aux_duration"], "event": batch["aux_event"]},
            )
            self.log(
                "aux_train_loss", aux_loss, prog_bar=True, on_epoch=True, on_step=True
            )
        else:
            aux_loss = 0.0
        loss, _ = self.loss(log_partial_hazards, batch)
        self.log("train_loss_wo_aux", loss, prog_bar=True, on_epoch=True, on_step=True)
        loss += self.aux_weight * aux_loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)

        if torch.isnan(loss):
            return None
        else:
            return loss

    def configure_optimizers(self):
        print("\n\n\n")
        print("***" * 5)
        print(f"optimizer: {self.hparams.optimizer}")
        params = self.parameters()
        assert self.hparams.optimizer in ["sgd", "adam", "adamw"]
        low_bit = self.hparams.low_bit
        if self.hparams.optimizer == "sgd":
            optim = torch.optim.SGD(
                params,
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adam":
            opt = bnb.optim.Adam8bit if low_bit else torch.optim.Adam
            optim = opt(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            opt = bnb.optim.AdamW8bit if low_bit else torch.optim.AdamW
            optim = opt(
                params,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        if self.hparams.warmup_epochs > 0:
            num_warmup_steps = self.hparams.warmup_epochs * self.hparams.num_dl_batches
            num_main_steps = (
                self.hparams.epochs - self.hparams.warmup_epochs
            ) * self.hparams.num_dl_batches + 1
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=0.01, end_factor=1, total_iters=num_warmup_steps
            )
            scheduler2 = CosineAnnealingLR(optim, T_max=num_main_steps)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optim, [scheduler1, scheduler2], milestones=[num_warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optim, self.hparams.epochs * self.hparams.num_dl_batches
            )
        print(f"optimizer object: {optim}")
        print("***" * 5)
        print("\n\n\n")
        return [optim], [{"scheduler": scheduler, "interval": "step"}]
