import os
import sys
from itertools import product

import numpy as np
import pandas as pd


def change_nucleotide(char):
    """get the complementary base pairs"""
    if char == "A":
        return "T"
    elif char == "T":
        return "A"
    elif char == "G":
        return "C"
    elif char == "C":
        return "G"
    else:
        return "-"


def reverse_seq(sequence):
    """find the reverse sequences of the off target DNAs"""
    New_seq = []
    for i in sequence:
        my_seq = ""
        for c in range(0, len(i)):
            char = change_nucleotide(i[c])
            my_seq += char
        New_seq.append(my_seq)
    return New_seq


def create_base_dict(letters, combination_length):
    """Combination length specifies single, doublet or triplet
    Creates a base dictionary, ex. AATT, GGCC, ..."""
    my_list = list(
        product(letters, repeat=combination_length)
    )  # create all 4 (or 2 or 6) letter combination list. ex: ('A', 'C', 'G', 'U'), ..
    final_list = [None] * len(my_list)

    for i in range(
        len(my_list)
    ):  # get rid of the inner lists and bring the strings together
        my_str = ""
        temp_list = my_list[i]
        for j in range(len(temp_list)):
            my_str += temp_list[j]
        final_list[i] = my_str
    return final_list


def dense_encoder(target_RNA, reverse_DNA, pairs, encoding="doublet"):
    """5'->target RNA (NGG)->3'
    3'->DNA->5"""

    if encoding == "triplet":  # Take 2 or 3 stacks of nucleotides and stop index
        # specific to the encoding scheme
        end = 2
        jump = 3
    elif encoding == "single":
        end = 0
        jump = 1
    else:  # doublet
        end = 1
        jump = 2

    encode = [[] for _ in range(len(target_RNA))]  # create the 2D encode matrix
    for j in range(len(target_RNA)):  # go through all sequences
        RNA = target_RNA[j]  # assign RNA sequence/target
        DNA = reverse_DNA[j]  # assign DNA/off-target seq
        list_1D = []  # create empty 1D matrix
        for i in range(len(RNA) - end):  # go through stacks
            base_pair = ""
            up = RNA[i : i + jump]
            down = DNA[i : i + jump]
            base_pair = up + down  # create the stack pair, ex. AATT
            my_index = 0
            for (
                pair,
                index,
            ) in pairs.items():  # get the index of the base pair (1 to 256)
                if pair == base_pair:
                    my_index = index
                    break
            list_1D.append(my_index)  # fill in the 1D matrix
        encode[j] = list_1D  # fill in the 2D encode matrix
    encode = np.array(encode, dtype=int)
    return encode


def extract_file_name(name):
    # Find the position of the last '/'
    last_slash_index = name.rfind("/")
    # Find the position of '.csv'
    dot_csv_index = name.rfind(".csv")
    # Extract the part of the string between the last '/' and '.csv'
    name = name[last_slash_index + 1 : dot_csv_index]
    return name


class off_tar_read(object):
    """Read and encode the data"""

    def __init__(self, file_path, pairs):  # passing arguments to class
        super(off_tar_read, self).__init__()
        self.file_path = file_path
        self.read_file = pd.read_csv(file_path)  # assign the file path
        self.target = self.read_file.loc[
            :, "sequence"
        ]  # list of the target RNA seq/on-target sites
        self.off_tar = self.read_file.loc[
            :, "Target sequence"
        ]  # list of the off_target DNA seq
        # target -> the gRNA sequence
        # off-target -> the DNA sequence gRNA binds to

        self.labels = np.asarray(
            self.read_file.loc[:, "class"]
        )  # get the labels and convert to array
        self.pairs = pairs

    def encode(self, encoding="doublet"):  # encoding starts here
        name = extract_file_name(self.file_path)
        if encoding == "doublet":
            print("Doublet encoding...")
            name = name + "_doublet_encoded.txt"

        elif encoding == "triplet":
            print("Triplet encoding...")
            name = name + "_triplet_encoded.txt"

        elif encoding == "single":
            print("Single encoding...")
            name = name + "_single_encoded.txt"

        else:
            print(
                'Invalid encoding nomenclature. Encoding options: "single", "doublet" or "triplet" '
            )
            sys.exit()

        encode_path = "./" + "encoded_data/" + name

        if os.path.isfile(encode_path):  # If the encoded data exists, use that instead
            encode_matrix = np.loadtxt(encode_path)
        else:  # If encoded data does not exist, do it from scratch
            directory = os.path.dirname(encode_path)
            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            reverse_RNA = reverse_seq(self.off_tar)
            encode_matrix = dense_encoder(
                self.target, reverse_RNA, self.pairs, encoding
            )  # dense encoding for doublet encoding
            np.savetxt(encode_path, encode_matrix)

        assert encode_matrix.shape[0] == len(
            self.labels
        )  # sanity check for labels and encode array
        return encode_matrix, self.labels


class base_pair(object):
    """Create the stack index for encoding"""

    def __init__(self):
        super(base_pair, self).__init__()
        self.letters = ["A", "T", "G", "C"]
        self.pairs = {}

    def create_dict(self, encoding="doublet"):
        temp = []
        combination = 0

        if encoding == "single":
            combination = 2
        elif encoding == "doublet":
            combination = 4
        elif encoding == "triplet":
            combination = 6

        number = pow(4, combination)
        for i in range(number):
            temp.append(i)

        list_of_letters = create_base_dict(
            self.letters, combination
        )  # get all possible length X combinations
        self.pairs = dict(
            zip(list_of_letters, temp)
        )  # create the code numbers for encoding
        return self.pairs


class GetSeq(object):
    """
    Convert encoded sequences to sgRNA-DNA pairs
    """

    def __init__(self, input_seq, base_list):
        super(GetSeq, self).__init__()
        self.input_seq = input_seq
        self.base_list = base_list

    def get_reverse(self, encoding="doublet"):
        # Take 1,2 or 3 stacks of nucleotides and stop index
        if encoding == "triplet":
            end = 2
            jump = 3
        elif encoding == "single":
            end = 0
            jump = 1
        else:  # doublet
            end = 1
            jump = 2

        on_target = (
            []
        )  # Get an empty 2D array, [] for _ in range(self.input_seq.shape[0])
        off_target = []

        key_list = list(self.base_list.keys())
        val_list = list(self.base_list.values())
        for i in range(self.input_seq.shape[0]):
            on_seq = ""
            off_seq = ""
            for j in range(self.input_seq.shape[1]):
                key = int(self.input_seq[i][j])
                position = val_list.index(key)
                pair = key_list[position]
                on_temp = pair[:jump]
                off_temp = pair[jump:]
                if j == self.input_seq.shape[1] - end:
                    on_seq += on_temp
                    off_seq += off_temp
                else:
                    on_seq += on_temp[0]
                    off_seq += off_temp[0]
            on_target.append(on_seq)
            off_target.append(off_seq)

        off_target = reverse_seq(off_target)

        return on_target, off_target
