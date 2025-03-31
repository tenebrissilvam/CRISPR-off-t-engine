import numpy as np


def preprocess_inputs_batch(guide_seqs, off_seqs):
    """
    Processes a batch of sequence pairs.
    Each element in guide_seqs/off_seqs should be a string.
    Returns a list of arrays, each of shape (4, max_length) for the corresponding pair.
    """
    code_dict = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "C": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }

    processed_list = []

    for guide_seq, off_seq in zip(guide_seqs, off_seqs):
        # Pad sequences to equal length.
        if len(guide_seq) != len(off_seq):
            max_len = max(len(guide_seq), len(off_seq))
            guide_seq = guide_seq.ljust(max_len, "N")
            off_seq = off_seq.ljust(max_len, "N")

        pair_code = []
        for g, o in zip(guide_seq, off_seq):
            g_code = np.array(code_dict[g])
            o_code = np.array(code_dict[o])
            combined = np.where((g_code + o_code) > 0, 1, 0).tolist()
            pair_code.append(combined)

        # Transpose so that each output is (4, sequence_length)
        input_array = np.array(pair_code).T
        processed_list.append(input_array)

    return processed_list
