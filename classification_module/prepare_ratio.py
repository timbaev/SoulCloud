from classification_module.numpy_proceed import get_vectors_by_category
from gensim.models import Doc2Vec
import numpy as np

def get_model_ratio(model : Doc2Vec):

    print("Подсчитываем коэффициенты модели...")

    result = 0

    for i in range(0, 1):
        temp = np.reshape(get_vectors_by_category(model, str(i)), (2, -2))
        result += np.sum(temp) / temp.size

    print("Подсчитано")

    return result