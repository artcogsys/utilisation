import numpy as np
import matplotlib.pyplot as plt

conf_arr = np.loadtxt("confusion_matrix.txt", dtype=int, delimiter=",")

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        if a == 0:
            tmp_arr.append(0)
        else:
            tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure(figsize=(400, 400))
res = plt.imshow(np.array(norm_conf), aspect='auto', cmap=plt.cm.jet,
                interpolation='nearest')

width, height = conf_arr.shape

cb = plt.colorbar(res)
plt.show()