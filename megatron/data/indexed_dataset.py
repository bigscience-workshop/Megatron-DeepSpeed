# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# copied from fairseq/fairseq/data/indexed_dataset.py
# Removed IndexedRawTextDataset since it relied on Fairseq dictionary
# other slight modifications to remove fairseq dependencies
# Added document index to index file and made it accessible.
#    An empty sentence no longer separates documents.

from functools import lru_cache
import os
import shutil
import struct
from itertools import accumulate

import numpy as np
import torch

from megatron import print_rank_0


def best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def infer_dataset_impl(path):
    if IndexedDataset.exists(path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            if magic == IndexedDataset._HDR_MAGIC:
                return 'cached'
            elif magic == MMapIndexedDataset.Index._HDR_MAGIC[:8]:
                return 'mmap'
            else:
                return None
    else:
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None


def make_builder(out_file, impl, dtype=None):
    if impl == 'mmap':
        assert dtype is not None
        return MMapIndexedDatasetBuilder(out_file, dtype=dtype)
    else:
        assert dtype is None
        return IndexedDatasetBuilder(out_file)


def make_dataset(path, impl, skip_warmup=False):
    if not IndexedDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .idx and .bin can be appended to get full filenames.")
        return None
    if impl == 'infer':
        impl = infer_dataset_impl(path)
    if impl == 'lazy' and IndexedDataset.exists(path):
        return IndexedDataset(path)
    elif impl == 'cached' and IndexedDataset.exists(path):
        return IndexedCachedDataset(path)
    elif impl == 'mmap' and MMapIndexedDataset.exists(path):
        return MMapIndexedDataset(path, skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None


def dataset_exists(path, impl):
    if impl == 'mmap':
        return MMapIndexedDataset.exists(path)
    else:
        return IndexedDataset.exists(path)


def read_longs(f, n):
    a = np.empty(n, dtype=np.int64)
    f.readinto(a)
    return a


def write_longs(f, a):
    f.write(np.array(a, dtype=np.int64))


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16
}


def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return prefix_path + '.idx'


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


class IndexedDataset(torch.utils.data.Dataset):
    """Loader for IndexedDataset"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.read_index(path)

    def read_index(self, path):
        with open(index_file_path(path), 'rb') as f:
            magic = f.read(8)
            assert magic == self._HDR_MAGIC, (
                'Index file doesn\'t match expected format. '
                'Make sure that --dataset-impl is configured properly.'
            )
            version = f.read(8)
            assert struct.unpack('<Q', version) == (1,)
            code, self.element_size = struct.unpack('<QQ', f.read(16))
            self.dtype = dtypes[code]
            self._len, self.s = struct.unpack('<QQ', f.read(16))
            self.doc_count = struct.unpack('<Q', f.read(8))
            self.dim_offsets = read_longs(f, self._len + 1)
            self.data_offsets = read_longs(f, self._len + 1)
            self.sizes = read_longs(f, self.s)
            self.doc_idx = read_longs(f, self.doc_count)

    def read_data(self, path):
        self.data_file = open(data_file_path(path), 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if not self.data_file:
            self.read_data(self.path)
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            return a
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sizes = self.sizes[self.dim_offsets[start]:self.dim_offsets[stop]]
            size = sum(sizes)
            a = np.empty(size, dtype=self.dtype)
            self.data_file.seek(self.data_offsets[start] * self.element_size)
            self.data_file.readinto(a)
            offsets = list(accumulate(sizes))
            sents = np.split(a, offsets[:-1])
            return sents

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def supports_prefetch(self):
        return False  # avoid prefetching to save memory


class IndexedCachedDataset(IndexedDataset):

    def __init__(self, path):
        super().__init__(path)
        self.cache = None
        self.cache_index = {}

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        if all(i in self.cache_index for i in indices):
            return
        if not self.data_file:
            self.read_data(self.path)
        indices = sorted(set(indices))
        total_size = 0
        for i in indices:
            total_size += self.data_offsets[i + 1] - self.data_offsets[i]
        self.cache = np.empty(total_size, dtype=self.dtype)
        ptx = 0
        self.cache_index.clear()
        for i in indices:
            self.cache_index[i] = ptx
            size = self.data_offsets[i + 1] - self.data_offsets[i]
            a = self.cache[ptx: ptx + size]
            self.data_file.seek(self.data_offsets[i] * self.element_size)
            self.data_file.readinto(a)
            ptx += size
        if self.data_file:
            # close and delete data file after prefetch so we can pickle
            self.data_file.close()
            self.data_file = None

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            i = idx
            self.check_index(i)
            tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
            a = np.empty(tensor_size, dtype=self.dtype)
            ptx = self.cache_index[i]
            np.copyto(a, self.cache[ptx: ptx + a.size])
            return a
        elif isinstance(idx, slice):
            # Hack just to make this work, can optimizer later if necessary
            sents = []
            for i in range(*idx.indices(len(self))):
                sents.append(self[i])
            return sents


class IndexedDatasetBuilder(object):
    element_sizes = {
        np.uint8: 1,
        np.int8: 1,
        np.int16: 2,
        np.int32: 4,
        np.int64: 8,
        np.float: 4,
        np.double: 8
    }

    def __init__(self, out_file, dtype=np.int32):
        self.out_file = open(out_file, 'wb')
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []
        self.element_size = self.element_sizes[self.dtype]
        self.doc_idx = [0]

    def add_item(self, tensor):
        bytes = self.out_file.write(np.array(tensor.numpy(), dtype=self.dtype))
        self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
        for s in tensor.size():
            self.sizes.append(s)
        self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))

    def end_document(self):
        self.doc_idx.append(len(self.sizes))

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)
        assert index.dtype == self.dtype

        offset = len(self.sizes)

        begin = self.data_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.data_offsets.append(begin + offset)
        self.sizes.extend(index.sizes)
        begin = self.dim_offsets[-1]
        for dim_offset in index.dim_offsets[1:]:
            self.dim_offsets.append(begin + dim_offset)
        self.doc_idx.extend( (offset + index.doc_idx)[1:] )

        with open(data_file_path(another_file), 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break

    def finalize(self, index_file):
        self.out_file.close()
        index = open(index_file, 'wb')
        index.write(b'TNTIDX\x00\x00')
        index.write(struct.pack('<Q', 1))
        index.write(struct.pack('<QQ', code(self.dtype), self.element_size))
        index.write(struct.pack('<QQ', len(self.data_offsets) - 1, len(self.sizes)))
        index.write(struct.pack('<Q', len(self.doc_idx)))
        write_longs(index, self.dim_offsets)
        write_longs(index, self.data_offsets)
        write_longs(index, self.sizes)
        write_longs(index, self.doc_idx)
        index.close()


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes, npdtype):
                    """Return a numpy array of byte offsets given a list of sizes.

                    Multiplies values in the sizes array by dtype size (bytes),
                    and then computes a zero-based prefix scan.
                    """

                    # create numpy array of desired numpy datatype
                    pointers = np.array(sizes, dtype=npdtype)

                    if len(sizes) > 0:
                        # scale each element by its dtype size
                        dtype_size = dtype().itemsize
                        pointers *= dtype_size

                        # in-place prefix scan to compute byte offsets
                        np.cumsum(pointers, axis=0, out=pointers)

                        # zero-base the prefix scan (exclusive scan)
                        pointers -= pointers[0]

                    return pointers

                def write(self, sizes, doc_idx):
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes32 = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes32.tobytes(order='C'))
                    del sizes32

                    pointers = self._get_pointers(sizes, np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
#                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
#            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
#            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
#            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )

    @property
    def dtype(self):
        return self._index.dtype


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

#        total_len = len(index.sizes)+len(self._sizes)
#        print(f"    concat {another_file} size={len(index.sizes)} for a total size of {total_len}")

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend( (offset + index.doc_idx)[1:] )

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)


def mpi_get_sum(val, mpi, comm):
    """Compute sum of val, and return total on all ranks."""
    insize = np.array([val], dtype=np.int64)
    outsize = np.zeros_like(insize)
    comm.Allreduce(insize, outsize, op=mpi.SUM)
    return outsize[0]


def mpi_get_offset(val, mpi, comm):
    """Compute prefix sum (exclusive scan) of val, and return offset of each rank."""
    insize = np.array([val], dtype=np.int64)
    outsize = np.zeros_like(insize)
    comm.Scan(insize, outsize, op=mpi.SUM)
    offset = outsize[0] - insize[0]
    return offset


def mpi_get_min(val, mpi, comm):
    """Return minimum val to all ranks."""
    insize = np.array([val], dtype=np.int64)
    outsize = np.zeros_like(insize)
    comm.Allreduce(insize, outsize, op=mpi.MIN)
    return outsize[0]


def mpi_create_file(filename, mpi, comm):
    """Create, truncate, and open a file shared by all ranks."""
    # Don't truncate file until all ranks reach this point
    comm.barrier()

    # Wait for rank 0 to open (and truncate) file,
    # then have all ranks open file for writing.
    rank = comm.Get_rank()
    if rank == 0:
        f = open(filename, 'wb')
    comm.barrier()
    if rank != 0:
        f = open(filename, 'r+b')

    # TODO: verify that all ranks successfully opened the file
    comm.barrier()

    return f


# To create the binary files given a set of per-rank binary
# files, one simply concatenates the data from the per-rank
# binary files in rank order.  We stat each rank file to determine
# its size, execute a scan to compute the byte offset where
# the calling rank should write its data, seek to proper
# spot, and copy the full file.
def merge_files_mpi_bin(outfile, infile, mpi, comm):
    """Concatenate per-rank binary files into a new file given by outfile"""
    import stat
    import shutil

    # Create shared output file.
    f = mpi_create_file(data_file_path(outfile), mpi, comm)

    # get file size of binary file for this rank
    filesize = os.stat(data_file_path(infile))[stat.ST_SIZE]

    # compute offset this rank should start copying
    # its data into the merged file
    offset = mpi_get_offset(filesize, mpi, comm)

    # seek to appropriate offset and copy data
    f.seek(offset)
    with open(data_file_path(infile), "rb") as fsrc:
        shutil.copyfileobj(fsrc, f)

    f.close()

    # TODO: check that all ranks wrote successfully
    comm.barrier()


def merge_files_mpi_idx_cached(outfile, infile, mpi, comm, dtype):
    rank = comm.Get_rank()
    numranks = comm.Get_size()

    # Create shared output file
    f = mpi_create_file(index_file_path(outfile), mpi, comm)

    # Read the index file for the calling rank
    index = IndexedDataset(infile)
    sizes = index.sizes
    if rank == 0:
        data_offsets = index.data_offsets
        dim_offsets = index.dim_offsets
        docs = index.doc_idx
    if rank != 0:
        data_offsets = index.data_offsets[1:]
        dim_offsets = index.dim_offsets[1:]
        docs = index.doc_idx[1:]

    # Compute total number of size and document index
    # values across all ranks.  Also compute the offset
    # of the calling rank for each value considering
    # the values of sizes/docs for all ranks before the
    # calling rank.
    numdata = len(data_offsets)
    numsize = len(sizes)
    numdim = len(dim_offsets)
    numdoc = len(docs)

    global_data_count = mpi_get_sum(numdata, mpi, comm)
    global_size_count = mpi_get_sum(numsize, mpi, comm)
    global_dim_count = mpi_get_sum(numdim, mpi, comm)
    global_doc_count = mpi_get_sum(numdoc, mpi, comm)

    global_data_offset = mpi_get_offset(numdata, mpi, comm)
    global_size_offset = mpi_get_offset(numsize, mpi, comm)
    global_dim_offset = mpi_get_offset(numdim, mpi, comm)
    global_doc_offset = mpi_get_offset(numdoc, mpi, comm)

    # Have rank 0 write the file header
    pos = 0
    if rank == 0:
        f.write(IndexedDataset._HDR_MAGIC)
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<QQ', code(index.dtype), index.element_size))
        f.write(struct.pack('<QQ', global_data_count - 1, global_size_count))
        f.write(struct.pack('<Q', global_doc_count))
        pos = f.tell()

    # Broadcast value of pos from rank 0,
    # and advance file position past file header on all ranks.
    pos = mpi_get_sum(pos, mpi, comm)

    # The dimension list records the offset within
    # the sizes list for each sentence.  Adjust dimension
    # offset values based on the number of offsets
    # that come before the calling rank.
    dim_offsets64 = np.array(dim_offsets, dtype=np.int64)
    dim_last = dim_offsets[-1] if numdim > 0 else 0
    dim_offset = mpi_get_offset(dim_last, mpi, comm)
    dim_offsets64 += dim_offset

    # Seek to proper offset for this rank and write
    # dom offset values into file, stored as int64.
    f.seek(pos + global_dim_offset * np.int64().itemsize)
    f.write(dim_offsets64.tobytes(order='C'))
    del dim_offsets64

    # Advance past list of dim offset values
    pos += global_dim_count * np.int64().itemsize

    # The data index records the byte offset to the start of each
    # sentence within the binary data file.
    # Adjust our data index values for number of bytes that
    # come before the calling rank.
    data_offsets64 = np.array(data_offsets, dtype=np.int64)
    byte_last = data_offsets[-1] if numdata > 0 else 0
    byte_offset = mpi_get_offset(byte_last, mpi, comm)
    data_offsets64 += byte_offset

    # Seek to proper offset for this rank and write
    # data (byte) offset index into file, stored as int64.
    f.seek(pos + global_data_offset * np.int64().itemsize)
    f.write(data_offsets64.tobytes(order='C'))
    del data_offsets64

    # Advance past list of data (byte) offset values
    pos += global_data_count * np.int64().itemsize

    # Each sentence is stored as a tensor.
    # The tensor for each sentence can be multidimensional.
    # The number of tensor dimensions per sentence is variable,
    # and the size of each dimension of a sentence is arbitrary.
    # The size list records a flattened list of the sizes
    # for each dimension of a sentence.
    # The list of size values from each rank are
    # concatenated and stored as int64.
    f.seek(pos + global_size_offset * np.int64().itemsize)
    sizes64 = np.array(sizes, dtype=np.int64)
    f.write(sizes64.tobytes(order='C'))
    del sizes64

    # Advance past list of size values
    pos += global_size_count * np.int64().itemsize

    # The document index points to the position in the sizes
    # array for the first sentence of the sample.
    docs64 = np.array(docs, dtype=np.int64)
    docs64 += global_size_offset

    # Seek to proper offset for this rank and write
    # document index into file, stored as int64.
    f.seek(pos + global_doc_offset * np.int64().itemsize)
    f.write(docs64.tobytes(order='C'))
    del docs64

    f.close()

    # TODO: check that all ranks wrote successfully
    comm.barrier()


def merge_files_mpi_idx_mmap(outfile, infile, mpi, comm, dtype):
    rank = comm.Get_rank()
    numranks = comm.Get_size()

    # Create shared output file
    f = mpi_create_file(index_file_path(outfile), mpi, comm)

    # Read the index file for the calling rank
    index = MMapIndexedDataset.Index(index_file_path(infile))
    sizes = index.sizes
    if rank == 0:
        docs = index.doc_idx
    if rank != 0:
        docs = index.doc_idx[1:]

    # Compute total number of size and document index
    # values across all ranks.  Also compute the offset
    # of the calling rank for each value considering
    # the values of sizes/docs for all ranks before the
    # calling rank.
    numsizes = len(sizes)
    numdocs = len(docs)
    total_size_count = mpi_get_sum(numsizes, mpi, comm)
    total_docs_count = mpi_get_sum(numdocs, mpi, comm)
    size_offset = mpi_get_offset(numsizes, mpi, comm)
    docs_offset = mpi_get_offset(numdocs, mpi, comm)

    # Have rank 0 write the file header
    pos = 0
    if rank == 0:
        f.write(MMapIndexedDataset.Index._HDR_MAGIC)
        f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<B', code(dtype)))
        f.write(struct.pack('<Q', total_size_count))
        f.write(struct.pack('<Q', total_docs_count))
        pos = f.tell()

    # Broadcast value of pos from rank 0,
    # and advance file position past file header on all ranks.
    pos = mpi_get_sum(pos, mpi, comm)

    # The list of size values from each rank are
    # concatenated and stored as int32.
    f.seek(pos + size_offset * np.int32().itemsize)
    sizes32 = np.array(sizes, dtype=np.int32)
    f.write(sizes32.tobytes(order='C'))
    del sizes32

    # Advance past list of size values
    pos += total_size_count * np.int32().itemsize

    # The pointer values store the byte offset to each sentence.
    # A sentence has a variable number of tokens, given by
    # its corresponding entry in the size array.  Each token
    # of a sentence is stored in units of type dtype, which consumes
    # dtype().itemsize bytes (often a standard type that is just
    # large enough to represent all elements of the vocabulary).

    # First compute byte offsets for each sentence of our
    # local set of sentences.
    pointers = np.array(sizes, dtype=np.int64)
    pointer_last = 0
    if len(sizes) > 0:
        np.cumsum(pointers, axis=0, out=pointers)
        pointers *= dtype().itemsize
        pointer_last = pointers[-1]

    # Then account for bytes for all sentences on ranks
    # before the calling rank.
    pointer_offset = mpi_get_offset(pointer_last, mpi, comm)
    pointers += pointer_offset

    # Finally, zero-base the offset values by subtracting
    # the number of bytes of the first sentence.  To do that
    # we first need to find the rank having the first sentence,
    # then bcast that size to all ranks.
    if total_size_count > 0:
        # There is at least one sentence across all ranks,
        # figure out which rank has the first sentence which
        # is not necessarily rank 0.
        minrank = numranks
        if len(sizes) > 0:
            minrank = rank
        minrank = mpi_get_min(minrank, mpi, comm)

        # Broadcast size of the first sentence from minrank.
        # We "bcast" using all allreduce.
        pointers_shift = 0
        if minrank == rank:
            pointers_shift = pointers[0]
        pointers_shift = mpi_get_sum(pointers_shift, mpi, comm)

        # Zero-base pointers by subtracting size of first
        # sentence from all values.
        pointers -= pointers_shift

    # Seek to proper offset for this rank and write
    # pointer values into file, stored as int64.
    f.seek(pos + size_offset * np.int64().itemsize)
    f.write(pointers.tobytes(order='C'))
    del pointers

    # Advance past list of pointer values
    pos += total_size_count * np.int64().itemsize

    # The document index points to the position in the sizes
    # array for the starting sentence of each document.
    # A variable number of sentences can be in each document.
    # Adjust document index for number of sentences that
    # come before the calling rank.
    doc_idx = np.array(docs, dtype=np.int64)
    doc_idx += size_offset

    # Seek to proper offset for this rank and write
    # document index into file, stored as int64.
    f.seek(pos + docs_offset * np.int64().itemsize)
    f.write(doc_idx.tobytes(order='C'))
    del doc_idx

    f.close()

    # TODO: check that all ranks wrote successfully
    comm.barrier()


def merge_files_mpi(filemain, filerank, mpi, comm, dtype=np.int64):
    # read index file of this rank to determine its type
    indexstr = infer_dataset_impl(filerank)

    # check that all ranks have the same type
    indexstrmap = {"cached": 1, "mmap": 2}
    indextype = indexstrmap[indexstr] if indexstr in indexstrmap else 0
    rank0type = comm.bcast(indextype, root=0)
    #allsame = all_true(indextype == rank0type)
    #if not allsame:
    #    error

    # Concatenate the data files
    merge_files_mpi_bin(filemain, filerank, mpi, comm)

    # Combine index files into a single index file
    if indexstr == "cached":
        merge_files_mpi_idx_cached(filemain, filerank, mpi, comm, dtype)
    elif indexstr == "mmap":
        merge_files_mpi_idx_mmap(filemain, filerank, mpi, comm, dtype)
