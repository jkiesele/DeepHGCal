import numpy as np
import sparse_hgcal


def sequence_mask(lengths, max_length):
    sequence = np.tile(np.expand_dims(np.arange(max_length), axis=0), [len(lengths), 1])
    array = np.tile(np.expand_dims(lengths, axis=1), [1, max_length])

    return sequence < array


def merge_two_arrays(arrayA, arrayB, idsA, idsB, sizesA, sizesB):
    """
    Merges two showers of calo data

    :param arrayA: data for first shower (BxVxF)
    :param arrayB: data for second shower (BxVxF)
    :param idsA: rechit ids for first shower (BxV)
    :param idsB: rechit ids for second shower (BxV)
    :param sizesA: number of rechits in first shower (B,)
    :param sizesB: number of rechits in second shower (B,)

    :return: Tuple: merged data (Bx(2V)xF), ids (Bx(2V)), sizes (B)
    """

    return _merge_function(arrayA, arrayB, idsA, idsB, sizesA, sizesB, separate=False)


def merge_two_arrays_separate(arrayA, arrayB, idsA, idsB, sizesA, sizesB):
    """
    Merges two showers of calo data

    :param arrayA: data for first shower (BxVxF)
    :param arrayB: data for second shower (BxVxF)
    :param idsA: rechit ids for first shower (BxV)
    :param idsB: rechit ids for second shower (BxV)
    :param sizesA: number of rechits in first shower (B,)
    :param sizesB: number of rechits in second shower (B,)

    :return: Tuple: merged data (Bx(2V)xF), ids (Bx(2V)), sizes (B)
    """

    return _merge_function(arrayA, arrayB, idsA, idsB, sizesA, sizesB, separate=True)



def _merge_function(arrayA, arrayB, idsA, idsB, sizesA, sizesB, separate=False):
    """
    Merges two showers of calo data

    :param arrayA: data for first shower (BxVxF)
    :param arrayB: data for second shower (BxVxF)
    :param idsA: rechit ids for first shower (BxV)
    :param idsB: rechit ids for second shower (BxV)
    :param sizesA: number of rechits in first shower (B,)
    :param sizesB: number of rechits in second shower (B,)

    :return: Tuple: merged data (Bx(2V)xF), ids (Bx(2V)), sizes (B)
    """

    assert arrayA.dtype == np.float32
    assert arrayB.dtype == np.float32
    assert idsA.dtype == np.int32
    assert idsB.dtype == np.int32
    assert sizesA.dtype == np.int32
    assert sizesB.dtype == np.int32

    assert (arrayA.ndim == 3 and arrayB.ndim == 3)
    assert (idsA.ndim == 2 and idsB.ndim == 2)
    assert (sizesA.ndim == 1 and sizesB.ndim == 1)

    batch_size = arrayA.shape[0]
    max_vertices = arrayA.shape[1]
    num_features = arrayA.shape[2]

    assert (batch_size == arrayB.shape[0])
    assert (max_vertices == arrayB.shape[1])
    assert (num_features == arrayB.shape[2])
    assert (batch_size == idsA.shape[0] and batch_size == idsB.shape[0])
    assert (max_vertices == idsA.shape[1] and max_vertices == idsB.shape[1])
    assert (batch_size == sizesA.shape[0] and batch_size == sizesB.shape[0])

    idsA[np.logical_not(sequence_mask(sizesA, max_vertices))] = np.iinfo(np.int32).max
    idsB[np.logical_not(sequence_mask(sizesB, max_vertices))] = np.iinfo(np.int32).max

    indexing_tensor_A = np.concatenate((np.expand_dims(
        np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, max_vertices]), axis=-1),
                                        np.expand_dims(np.argsort(idsA, axis=1), axis=-1)), axis=2)
    indexing_tensor_B = np.concatenate((np.expand_dims(
        np.tile(np.expand_dims(np.arange(batch_size), axis=1), [1, max_vertices]), axis=-1),
                                        np.expand_dims(np.argsort(idsB, axis=1), axis=-1)), axis=2)

    arrayA = arrayA[indexing_tensor_A[:, :, 0], indexing_tensor_A[:, :, 1]]
    arrayB = arrayB[indexing_tensor_B[:, :, 0], indexing_tensor_B[:, :, 1]]

    idsA = idsA[indexing_tensor_A[:, :, 0], indexing_tensor_A[:, :, 1]]
    idsB = idsB[indexing_tensor_B[:, :, 0], indexing_tensor_B[:, :, 1]]

    return sparse_hgcal.merge_two_arrays_separate(arrayA, arrayB, idsA, idsB, sizesA, sizesB) if separate else sparse_hgcal.merge_two_arrays(arrayA, arrayB, idsA, idsB, sizesA, sizesB)
