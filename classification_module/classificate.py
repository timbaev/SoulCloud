from classification_module.prepare_ratio import get_model_ratio
from doc2vec_module.train_model import get_model_for_genre
from doc2vec_module.constants import FileConstants
from gensim.models import Doc2Vec
from doc2vec_module import load
from sklearn import svm
import numpy as np
import math

print("Подгружаем книги пользователя")
print(" ")

user_prop = getattr(FileConstants, "USER_BOOK_TEST")
love_model_prop = getattr(FileConstants, "MODEL_LOVE")
fantasy_model_prop = getattr(FileConstants,"MODEL_FANTASY")

user_documents = load.get_doc(user_prop.fget(FileConstants()))

print("Количество книг пользователя: {0}".format(len(user_documents)))

fantasy_model = Doc2Vec.load(fantasy_model_prop.fget(FileConstants()))
love_model = Doc2Vec.load(love_model_prop.fget(FileConstants()))

print(" ")
print(" Получаем модель для книг пользователя ")
user_model = get_model_for_genre(user_documents)

print(" ")
print("Готовим коэффициенты")
print(" ")
print("Обрабатываем любовную модель")

love_ratio = get_model_ratio(love_model)
print("Обработали")
print(" ")

print("Обрабатываем фэнтезийную модель")
fantasy_ratio = get_model_ratio(fantasy_model)
print("Обработали")
print(" ")

user_train = np.array(user_model.docvecs[str(0)])

love_model_test = np.reshape(np.array(love_model.docvecs[str(1)]), (2, -2))
fantasy_model_test = np.reshape(np.array(fantasy_model.docvecs[str(1)]), (2, -2))
user_model_test = np.reshape(user_train, (2, -2))

lin_clf = svm.LinearSVC()
lin_clf.fit(user_model_test, [int(math.fabs(love_ratio) * 100), int(math.fabs(fantasy_ratio) * 100)])

print("Коэффициент для любовного жанра: " + str(lin_clf.score(user_model_test, [int(math.fabs(love_ratio) * 100), 0])))

print("Коэффициент для фэнтезийного жанра: " + str(lin_clf.score(user_model_test, [int(math.fabs(fantasy_ratio) * 100), 1])))