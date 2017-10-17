import numpy as np

with open("../data/objectInfo150.txt") as f:
    lines = []
    for l in f.readlines()[1:]:
        lines.append(float(l.split("\t")[3].strip()))
    frequencies = np.array(lines)

    inverse_frequencies = np.max(frequencies) / frequencies

    # inverse_frequency_probabilities = inverse_frequency_exp / np.sum(inverse_frequency_exp)
    for a in zip(frequencies, inverse_frequencies):
        print "%.4f," % a[1]
