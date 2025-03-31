import numpy as np


def encoder(RNAseq, order=None):
    if order is None:
        order = ["A", "T", "C", "G"]

    lookup_table = {
        order[0]: [1, 0, 0, 0],
        order[1]: [0, 1, 0, 0],
        order[2]: [0, 0, 1, 0],
        order[3]: [0, 0, 0, 1],
    }
    encoded = np.zeros((len(RNAseq), len(order)))

    for i in range(len(RNAseq)):
        nu = RNAseq[i]
        if nu in lookup_table:
            encoded[i] = np.array(lookup_table[nu])
        else:
            print("Exception: Unindentified Nucleotide")

    return encoded


def decoder(encoded, order=None):
    if order is None:
        order = ["A", "T", "C", "G"]
    RNAseq = ""

    for i in range(encoded.shape[0]):
        idx = np.where(encoded[i] == 1)[0][0]  # first occurance only
        RNAseq += order[idx]

    return RNAseq


def superpose(encoded1, encoded2):
    if len(encoded1) != len(encoded2):
        print("Size Mismatch")
        return encoded1

    superposed = np.zeros(encoded1.shape)

    for i in range(len(encoded1)):
        for j in range(len(encoded1[i])):
            if encoded1[i][j] == encoded2[i][j]:
                superposed[i][j] = encoded1[i][j]
            else:
                superposed[i][j] = encoded1[i][j] + encoded2[i][j]
    return superposed


def superposeWithDirection(encoded1, encoded2):
    if len(encoded1) != len(encoded2):
        print("Size Mismatch")
        return encoded1

    superposed = np.zeros((encoded1.shape[0], encoded1.shape[1] + 1))

    for i in range(len(encoded1)):
        for j in range(len(encoded1[i])):
            if encoded1[i][j] == encoded2[i][j]:
                superposed[i][j] = encoded1[i][j]
            else:
                superposed[i][j] = encoded1[i][j] + encoded2[i][j]
                superposed[i][-1] = encoded1[i][j]
    return superposed


def get_encoded_data(df, channel_size=4):
    enc_targets = []
    enc_off_targets = []
    enc_superposed = []
    enc_superposed_with_dir = []
    labels = []

    for i in range(df.shape[0]):
        df_row = df.iloc[i]
        target = encoder(df_row["sequence"])
        off_target = encoder(df_row["Target sequence"])
        superposed = superpose(target, off_target)
        superposed_with_dir = superposeWithDirection(target, off_target)

        enc_targets.append(target)
        enc_off_targets.append(off_target)
        enc_superposed.append(superposed)
        enc_superposed_with_dir.append(superposed_with_dir)
        labels.append(df_row["class"])

        if i % 1000 == 0:
            print(i + 1, "/", df.shape[0], "done")

    print(len(enc_targets))
    print(len(enc_off_targets))
    print(len(enc_superposed))
    print(len(superposed_with_dir))
    print(len(labels))

    if channel_size == 4:
        return enc_superposed, labels
    else:
        return enc_superposed_with_dir, labels
