"""Dataset class built on SeqLikes"""

import torch
from torch.utils.data import Dataset


class SeqLikeDataset(Dataset):
    """A Dataset subclass for working with series of SeqLikes"""

    def __init__(self, seq_series, target_series_or_df=None, mode="onehot", target_mode=None):
        """Dataset for SeqLikes.

        This class expects  a Pandas Series of SeqLikes  as the input,
        and  a Pandas  Series  or DataFrame  containing  any type  for
        targets.   If target_mode  is not None, then the target is
        passed as is, otherwise it is expected to be a series of SeqLikes.

        This class takes advantage of the custom SeqLikeAccessor namespace to do
        conversion to onehot or index forms as needed.

        :param seq_series: a Pandas Series of SeqLike objects
        :param target_series_or_df: Pandas Series or DataFrame associated with each sequence, defaults to None
        :param mode: encoding mode for sequences.  can be "onehot" or "index", defaults to "onehot"
        :param target_mode: encoding mode for targets, defaults to None
        """
        self.seq_series = seq_series
        self.target_series_or_df = target_series_or_df

        # convert input sequences
        self.mode = mode
        assert self.mode in [
            "onehot",
            "index",
        ], 'Only supported modes are "onehot" and "index"!'

        if self.mode == "onehot":
            # we unsqueeze here for 2d convs on seq data
            self.data = torch.tensor(seq_series.seq.to_onehot()).float().unsqueeze(1)
        else:
            self.data = torch.tensor(seq_series.seq.to_index()).long()

        # convert target, if it is a sequence
        self.target_mode = target_mode
        assert self.target_mode in [
            "onehot",
            "index",
            None,
        ], 'Only supported modes are "onehot", "index" or None!'

        if target_series_or_df is not None:
            if self.target_mode == "onehot":
                # we unsqueeze here for 2d convs on seq data
                self.targets = torch.tensor(target_series_or_df.seq.to_onehot()).float().unsqueeze(1)

            elif self.target_mode == "index":
                self.targets = torch.tensor(target_series_or_df.seq.to_index()).long()

            elif self.target_mode is None:
                self.targets = torch.tensor(target_series_or_df.values.to_list())

        else:
            self.targets = torch.arange(start=0, end=len(self.seq_series))

    def __len__(self):
        """Return the length of the DataSet.
        :return: length of dataset"""
        return len(self.seq_series)

    def __getitem__(self, index):
        """Return an (input, target) tuple from the DataSet.

        :param index: an integer index into the dataset
        :return: a tuple of (sequence, target)
        """
        return self.data[index, :], self.targets[index]


class WeightedSeqLikeDataSet(SeqLikeDataset):
    """This is a subclass of SeqLikeDataset, and returns the input, target, and weight."""

    def __init__(self, seq_series, target_series_or_df=None, weights=None, mode=None, target_mode=None):
        """Weighted SeqLikeDataSet.

        :param seq_series: a Pandas Series of SeqLike objects
        :param target_series_or_df: Pandas Series or DataFrame associated with each sequence, defaults to None
        :param weights: a set relative or absolute weights, defaults to None
        :param mode: encoding mode for sequences.  can be "onehot" or "index", defaults to "onehot"
        :param target_mode: encoding mode for targets, defaults to None"""
        super().__init__(
            seq_series=seq_series,
            target_series=target_series_or_df,
            mode=mode,
            target_mode=target_mode,
        )

        if weights is None:
            self.weights = torch.ones(len(self.one_hot))
        else:
            self.weights = torch.tensor(weights.tolist())

    def __getitem__(self, index):
        """Return an (input, target, weight) tuple from the DataSet.

        :param index: an integer index into the dataset
        :return: a tuple of (sequence, target, weight)
        """
        return self.data[index, :], self.targets[index], self.weights[index]


class AutoEncoderSeqLikeDataSet(SeqLikeDataset):
    """This is a subclass of SeqLikeDataset for autoencoders, and returns the input and target."""

    def __init__(self, seq_series, mode=None):
        """Subclass of SeqLikeDataset for Autoencoders.

        :param seq_series: a Pandas Series of SeqLike objects
        :param mode: encoding mode for sequences.  can be "onehot" or "index", defaults to "onehot"
        """
        super().__init__(seq_series=seq_series, target_series=seq_series, mode=mode, target_mode=mode)


class WeightedAutoEncoderSeqLikeDataSet(SeqLikeDataset):
    """This is a subclass of SeqLikeDataset for autoencoders, and returns the input, target, and weight."""

    def __init__(self, seq_series, weights, mode=None, target_mode=None):
        """Weighted SeqLikeDataset for Autoencoders.

        :param seq_series: a Pandas Series of SeqLike objects
        :param weights: a set relative or absolute weights, defaults to None
        :param mode: encoding mode for sequences.  can be "onehot" or "index", defaults to "onehot"
        :param target_mode: encoding mode for targets, defaults to None
        """
        super().__init__(seq_series=seq_series, weights=weights, mode=mode)

        if weights is None:
            self.weights = torch.ones(len(self.one_hot))
        else:
            self.weights = torch.tensor(weights.tolist())

    def __getitem__(self, index):
        """Return an (input, target, weight) tuple from the DataSet.

        :param index: an integer index into the dataset
        :return: a tuple of (sequence, target, weight)
        """
        return self.data[index, :], self.data[index, :], self.weights[index]
