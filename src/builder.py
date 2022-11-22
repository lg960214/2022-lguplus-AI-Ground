import numpy as np
from scipy import sparse
import torch
import pytorch_lightning as pl
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch


class Engine(pl.LightningModule):
    def __init__(self, model, args, te_path):
        super().__init__()
        self.model = model
        self.save_hyperparameters(args)
        self.te_dataset = sparse.load_npz(te_path)

        self.anneal = self.hparams.anneal_cap

        # Intrinsic Hparams
        self.update_count = 0.0
        self.start_idx = 0

    def step(self, batch, batch_idx, state):
        recon_batch, mu, logvar = self.model(batch)

        if state == "train":
            if self.hparams.total_anneal_steps > 0:
                self.anneal = min(
                    self.hparams.anneal_cap,
                    1.0 * self.update_count / self.hparams.total_anneal_steps,
                )

            self.update_count += 1

            loss = self.model.loss_function(recon_batch, batch, mu, logvar, self.anneal)
            return {"loss": loss}

        else:
            loss = self.model.loss_function(recon_batch, batch, mu, logvar, self.anneal)
            recon_batch = recon_batch.cpu().numpy()
            # recon_batch[batch.cpu().numpy().nonzero()] = -np.inf

            end_idx = min(self.start_idx + batch.shape[0], self.te_dataset.shape[0])
            heldout_data = self.te_dataset[self.start_idx : end_idx]

            ndcg = NDCG_binary_at_k_batch(recon_batch, heldout_data, 25)
            recall = Recall_at_k_batch(recon_batch, heldout_data, 25)

            self.start_idx += batch.shape[0]

            return {"loss": loss, "ndcg": ndcg, "recall": recall}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, state="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, state="valid")

    def epoch_end(self, outputs, state="train"):
        loss = 0.0
        ndcg_list = []
        recall_list = []

        for i in outputs:
            loss += i["loss"].detach().cpu()
            ndcg_list.append(i["ndcg"])
            recall_list.append(i["recall"])

        assert len(ndcg_list) == len(recall_list)

        loss = loss / len(outputs)
        ndcg = np.mean(np.hstack(ndcg_list))
        recall = np.mean(np.hstack(recall_list))
        score = 0.75 * ndcg + 0.25 * recall

        self.log(state + "_loss", float(loss), on_epoch=True, prog_bar=True)
        self.log(
            state + "_ndcg@25",
            ndcg,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            state + "_recall@25",
            recall,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            state + "_score",
            score,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "anneal",
            self.anneal,
            on_epoch=True,
            prog_bar=True,
        )
        self.start_idx = 0

    def train_epoch_end(self, outputs):
        return self.epoch_end(outputs, state="train")

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state="valid")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        return [optimizer], [scheduler]
