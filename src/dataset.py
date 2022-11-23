import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def filter_triplets(df, min_uc=5, min_sc=5):
    """
    Args:
       min_uc (int, optional): [users who clicked on at least min_uc items]. Defaults to 5.
       min_sc (int, optional): [items which were clicked on by at least min_sc users]. Defaults to 5.
    """
    if min_sc > 0:
        item_count = df.groupby("album_id").size()
        df = df[df["album_id"].isin(item_count.index[item_count >= min_sc])]
    if min_uc > 0:
        user_count = df.groupby("profile_id").size()
        df = df[df["profile_id"].isin(user_count.index[user_count >= min_uc])]

    user_count, item_count = (
        df.groupby("profile_id").size(),
        df.groupby("album_id").size(),
    )

    df = df.reset_index(drop=True)
    return df, user_count, item_count


def filtering_results(raw_data, user_activity, item_popularity):
    sparsity = (
        1.0 * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0])
    )
    print(
        "After filtering, there are %d watching events from %d users and %d albums (sparsity: %.3f%%)"
        % (
            raw_data.shape[0],
            user_activity.shape[0],
            item_popularity.shape[0],
            sparsity * 100,
        )
    )


def split_users(unique_uid, n_heldout_users=1000):
    """Split train / validation / test users.
    Select 1K users as heldout users, 1K users as validation users, and the rest of the users for training
    Args:
        unique_uid (Array): [randomly permutated User Index]
        n_heldout_users (int)
    """
    n_users = len(unique_uid)
    tr_users = unique_uid[: (n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2) :]
    return tr_users, vd_users


def train_test_split(data, test_prop=0.1):
    data_grouped_by_user = data.groupby("profile_id")
    tr_list, te_list = list(), list()

    for i, (_, group) in enumerate(data_grouped_by_user):
        n_items_u = len(group)  # per user, item count

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype="bool")
            samples = int((1 - test_prop) * n_items_u)
            idx[samples:] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        else:
            tr_list.append(group)

    sys.stdout.flush()

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    print(f"Dataset Shape: tr:{data_tr.shape} \t te:{data_te.shape}")
    return data_tr, data_te


def numerize(df, profile2id, album2id):
    uid = df["profile_id"].map(lambda x: profile2id[x])
    pid = df["album_id"].map(lambda x: album2id[x])
    ss_watch_cnt = df["ss_watch_cnt"]

    return pd.DataFrame(
        data={"profile_id": uid, "album_id": pid, "ss_watch_cnt": ss_watch_cnt},
    ).reset_index(drop=True)


class Uplus_DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

    def prepare_data(self):
        self.df_history = pd.read_csv(
            os.path.join(self.args.raw_data_dir, "history_data.csv")
        )
        self.df_watch = pd.read_csv(
            os.path.join(self.args.raw_data_dir, "watch_e_data.csv")
        )
        self.df_profile = pd.read_csv(
            os.path.join(self.args.raw_data_dir, "profile_data.csv")
        )
        self.df_meta = pd.read_csv(
            os.path.join(self.args.raw_data_dir, "meta_data.csv")
        )
        self.n_items = self.df_meta["album_id"].nunique()

    def pre_processing(self):
        # history dataset
        self.df_history["continuous_play"] = np.where(
            self.df_history["continuous_play"] == "Y", 1, 0
        )
        self.df_history["short_trailer"] = np.where(
            self.df_history["short_trailer"] == "Y", 1, 0
        )
        self.df_history["ss_continuous_cnt"] = self.df_history.groupby(
            ["profile_id", "ss_id"]
        )["continuous_play"].transform("cumsum")

        # watch dataset
        self.df_watch["watch_ratio"] = (
            self.df_watch["watch_time"] / self.df_watch["total_time"]
        )
        self.df_watch["ss_watch_cnt"] = (
            self.df_watch.groupby(["profile_id", "ss_id", "album_id"])[
                "act_target_dtl"
            ].transform("cumcount")
            + 1
        )
        self.df_watch = self.df_watch.query("watch_ratio>=0.1").reset_index(drop=True)

    def merge_dataframes(self):
        df_seen = pd.merge(
            self.df_history.drop("log_time", axis=1),
            self.df_watch.drop("log_time", axis=1),
            how="inner",
            on=["profile_id", "ss_id", "album_id"],
        )

        df_seen = df_seen.drop_duplicates(
            subset=["profile_id", "ss_id", "album_id"], keep="last"
        ).reset_index(drop=True)

        df_seen = df_seen.merge(
            self.df_meta[["album_id", "genre_large", "genre_mid"]].drop_duplicates(),
            how="left",
            on="album_id",
        )

        df_seen = df_seen.merge(
            self.df_profile[["profile_id", "sex", "age"]], how="left", on=["profile_id"]
        )
        return df_seen

    def save_data(self, data, name):
        if not os.path.exists(self.args.processed_data_dir):
            os.makedirs(self.args.processed_data_dir, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(
                os.path.join(self.args.processed_data_dir, f"{name}.csv"), index=False
            )
        else:
            sparse.save_npz(
                file=os.path.join(self.args.processed_data_dir, f"{name}.npz"),
                matrix=data,
            )

    def to_csr_matrix(self, df):
        n_users = df["profile_id"].max() + 1
        rows, cols = df["profile_id"], df["album_id"]
        data = df["ss_watch_cnt"]
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), dtype="float64", shape=(n_users, self.n_items)
        )
        return matrix

    def setup(self, stage=None):
        self.prepare_data()
        self.pre_processing()

        # Merge dataset and check sparsity
        df_seen = self.merge_dataframes()
        df_clean, user_activity, item_popularity = filter_triplets(df_seen)
        filtering_results(df_clean, user_activity, item_popularity)

        # Get unique ids
        unique_uid = user_activity.index
        unique_pid = item_popularity.index

        # Shuffle User Index
        idx_perm = np.random.permutation(len(unique_uid))
        unique_uid = unique_uid[idx_perm]

        # Split Users
        tr_users, vd_users = split_users(unique_uid, self.args.heldout_users)

        train_plays = df_clean.loc[df_clean["profile_id"].isin(tr_users)]
        valid_plays = df_clean.loc[df_clean["profile_id"].isin(vd_users)]
        test_plays = df_clean.loc[df_clean["profile_id"].isin(unique_uid)]

        # Filtering album_id
        # unique_pid = pd.unique(train_plays["album_id"])

        valid_plays = valid_plays.loc[valid_plays["album_id"].isin(unique_pid)]
        test_plays = test_plays.loc[test_plays["album_id"].isin(unique_pid)]

        # Split Dataset
        valid_tr, valid_te = train_test_split(valid_plays)
        # test_tr, test_te = train_test_split(test)

        # Dictionarize the Unique profile_id and album_id
        profile2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
        album2id = dict((pid, i) for (i, pid) in enumerate(unique_pid))

        # Save Dataset
        train_data = numerize(train_plays, profile2id, album2id)
        self.save_data(train_data, "train")

        valid_data_tr = numerize(valid_tr, profile2id, album2id)
        self.save_data(valid_data_tr, "validation_tr")

        valid_data_te = numerize(valid_te, profile2id, album2id)
        self.save_data(valid_data_te, "validation_te")

        test_data = numerize(test_plays, profile2id, album2id)
        self.save_data(test_data, "test")

        # convert to csr_matrix
        train_mat = self.to_csr_matrix(train_data)
        self.save_data(train_mat, "train")

        # valid is need a processing because matrix indexing
        start_idx = min(
            valid_data_tr["profile_id"].min(), valid_data_te["profile_id"].min()
        )
        end_idx = max(
            valid_data_tr["profile_id"].max(), valid_data_te["profile_id"].max()
        )

        rows_tr, cols_tr = (
            valid_data_tr["profile_id"] - start_idx,
            valid_data_tr["album_id"],
        )
        rows_te, cols_te = (
            valid_data_te["profile_id"] - start_idx,
            valid_data_te["album_id"],
        )

        data_tr = sparse.csr_matrix(
            (valid_data_tr["ss_watch_cnt"], (rows_tr, cols_tr)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )
        data_te = sparse.csr_matrix(
            (valid_data_te["ss_watch_cnt"], (rows_te, cols_te)),
            dtype="float64",
            shape=(end_idx - start_idx + 1, self.n_items),
        )

        self.save_data(data_tr, "validation_tr")
        self.save_data(data_te, "validation_te")

        test_mat = self.to_csr_matrix(test_data)
        self.save_data(test_mat, "test")

    def dataloader(self, path, shuffle=True):
        mat = sparse.load_npz(path)
        dataset = torch.FloatTensor(mat.toarray())

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.dataloader(path=self.args.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(path=self.args.val_tr_path, shuffle=False)

    def test_dataloader(self):
        return self.dataloader(path=self.args.test_data_path, shuffle=False)
