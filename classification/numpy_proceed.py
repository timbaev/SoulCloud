import numpy as np

def get_vectors_by_category(model, category_name):
    return np.array(model.docvecs.most_similar(str(category_name)))