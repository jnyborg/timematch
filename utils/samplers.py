import torch


class VariableSequenceLengthBatchSampler(torch.utils.data.Sampler):
    """
    Outputs batches of patch indices where all patches have the same sequence length, so that
    they can be torch.stack'ed.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        n_examples_per_seq_length = []
        for zarr_file_idx in range(len(data_source.zarr_files)):
            n_examples_per_seq_length.append(len([1 for idx, _, _, _ in data_source.images if idx == zarr_file_idx]))

        self.batches = []
        for offset_idx, n_examples in enumerate(n_examples_per_seq_length):
            offset = sum(x for x in n_examples_per_seq_length[:offset_idx])
            indices = list(range(offset, offset + n_examples))
            self.batches += [indices[a:a+batch_size] for a in range(0, n_examples, batch_size)]

        # Sanity checks: all batches should be at most the batch size, and all patches should be included.
        assert all([len(batch) <= batch_size for batch in self.batches])
        assert sum([len(batch) for batch in self.batches]) == len(data_source)
        assert len({idx for batch in self.batches for idx in batch}) == len(data_source)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
