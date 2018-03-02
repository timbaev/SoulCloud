import numpy as np

def get_vectors_by_category(model, category_number):
    return np.array(model.docvecs[category_number])