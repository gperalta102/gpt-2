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
   