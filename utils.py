import numpy as np


def pad_sents(sents, pad_idx=0):
    if isinstance(sents[0], np.ndarray):
        return sents
    padded_sents = []
    maxlen = max([len(sent) for sent in sents])
    for sent in sents:
        padded_sent = sent.copy()
        padded_sent.extend([pad_idx] * (maxlen - len(sent)))
        padded_sents.append(padded_sent)
    return padded_sents


def data_iterator(*args, batch_size=100, random=True):
    npargs = []
    for arg in args:
        assert (len(arg) == len(args[0]))
        npargs.append(np.array(arg))

    ixs = np.arange(len(args[0]))
    if random:
        np.random.shuffle(ixs)
    for i in range(len(args[0]) // batch_size):
        batch_ixs = ixs[i * batch_size:(i + 1) * batch_size]
        batches = []
        for arg in npargs:
            batches.append(arg[batch_ixs])
        yield batches
