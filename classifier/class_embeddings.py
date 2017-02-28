from settings import CLASS_IDS, MAX_CLASS_ID
import numpy as np

class_embedding_lookup_table = np.full(MAX_CLASS_ID, len(CLASS_IDS))
for idx, class_id in zip(range(0, len(CLASS_IDS)), CLASS_IDS):
    class_embedding_lookup_table[class_id] = idx

