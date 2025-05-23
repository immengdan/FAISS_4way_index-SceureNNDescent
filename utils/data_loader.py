import struct
import numpy as np

"""
                  IO Utils
"""

def load_fvecs_fbin(filename, offset=0, count=None, dim=96):
    """Load vectors from .fbin file format"""
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    with open(filename, 'rb') as f:
        f.seek(offset * dim * itemsize)
        if count is not None:
            data = np.fromfile(f, dtype=dtype, count=count * dim)
            data = data.reshape(-1, dim)
        else:
            data = np.fromfile(f, dtype=dtype)
            data = data.reshape(-1, dim)
    return data

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, offset=start_idx * 4 * dim)
    
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:

        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size       

        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, offset=start_idx * 4 * dim)

    return arr.reshape(nvecs, dim)


def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))

        vecs.astype('float32').flatten().tofile(f)


def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))

        vecs.astype('int32').flatten().tofile(f)
