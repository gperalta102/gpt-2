import tqdm
import tensorflow as tf
import os
import numpy as np
import glob

def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi



def load_dataset(enc, path, combine, encoding=None):
    
    paths = []
    if os.path.isfile(path):
        paths.append(path)
    elif os.path.isdir(path):
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        paths = glob.glob(path)


    tokenChunks = []
    rawText = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    tokenChunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r', encoding=encoding) as fp:
                rawText += fp.read()
            if len(rawText) >= combine:
                tokens = np.stack(enc.encode(rawText))
                tokenChunks.append(tokens)
                rawText = ''
            else:
                rawText += '<|endoftext|>'
    if rawText:
        tokens = np.stack(enc.encode(rawText))
        rawText.append(tokens)
    return tokenChunks


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.
    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

   