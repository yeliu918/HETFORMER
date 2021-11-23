import gc
import glob
import random
import numpy as np
import torch
from itertools import permutations
from others.logging import logger



class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def entity_1d_to_2d(self, entity_1d_id_batch):
        entity_2d_batch = []
        for entity_1d_id in entity_1d_id_batch:
            entity_entity_mask = np.zeros((len(entity_1d_id), len(entity_1d_id)))
            entity_1d_id = np.array(entity_1d_id)
            for cluster_num in range(1, max(entity_1d_id) + 1):
                num_idxs = np.where(entity_1d_id == cluster_num)
                one_idxs = permutations(list(num_idxs[0]), 2)
                for x_y in one_idxs:
                    x, y = x_y
                    entity_entity_mask[x][y] = 1
            entity_2d_batch.append(entity_entity_mask)
        return np.stack(entity_2d_batch)

    def __init__(self, data=None, device=None,  is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_entity_1d = [x[4] for x in data]
            entity_1d_mask = torch.tensor(pre_entity_1d)

            src = torch.tensor(self._pad(pre_src, 1))

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segs, 0))
            mask = ~ (src == 1) ## mask the pad

            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~ (clss == -1)
            clss[clss == -1] = 0
            if labels.size(1) < mask_cls.size(1):
                new_labels = torch.zeros(mask_cls.shape).type_as(labels)
                new_labels[:,:labels.size(1)] = labels
            elif labels.size(1) == mask_cls.size(1):
                new_labels = labels
            else:
                print("check")

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', new_labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))
            setattr(self, "entity_mask", entity_1d_mask.to(device))

            assert new_labels.size(1) == mask_cls.size(1)
            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def batch(data, batch_size):
    """Yield elements from data in chunks of batch_size."""
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + "." + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size,  device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs


    def preprocess(self, ex, is_test):
        src = ex['src']
        if('labels' in ex):
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        entity_1d_mask = ex["entity_1d_mask"]

        if(is_test):
            src_txt = ex['src_txt']
            tgt_txt = ex['tgt_txt']
            return src,labels,segs, clss, entity_1d_mask, src_txt, tgt_txt
        else:
            return src,labels,segs, clss, entity_1d_mask

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 50):

            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
