from word_tokenization import tokenize
from .model_train import get_model_with_types
from sklearn.preprocessing import scale
from sklearn import svm
import copy
import numpy as np


def get_ratio_for_text():

    # Здесь - получение текста. Ввод не работает, так как обучение блокирует текст
    text = input("Введите текст для проверки алгоритма: ")

    text_clear = tokenize.Tokenization()
    fixed_text = text_clear.tokenize_me(text)

    model_dict = copy.deepcopy(get_model_with_types())
    model = model_dict.get("model")
    tags = model_dict.get("tags")

    vec = np.zeros(len(fixed_text))
    count = 0

    for word in fixed_text:

        try:
            vec += model[word].reshape((1, len(fixed_text)))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count

    vec = scale(vec)

    lin_clf = svm.LinearSVC()
    lin_clf.fit(model.docvecs[tags[0]], model.docvecs[tags[1]])

    print("Коэффициент для жанра ЛЮБОВЬ: " + str(lin_clf.score(vec, model.wv[tags[0]])))
    print("Коэффициент для жанра ДРАМА: " + str(lin_clf.score(vec, model.wv[tags[1]])))
