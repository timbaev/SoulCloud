import numpy as np

def get_vectors_by_category(model, category_id):
    return np.array(model.docvecs[category_id])