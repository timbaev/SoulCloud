from sklearn import svm
from pathlib import Path
from classification import train_model
from load import get_doc_names, get_doc_list
from classification.numpy_proceed import get_vectors_by_category
from gensim.models import Doc2Vec
from classification.clear_text import tokenize
import numpy as np

lin_clf = svm.LinearSVC()

base_path = Path(__file__).parents[1]

model_path = Path(str(base_path)).joinpath('models/book_model.doc2vec')
user_book_path = Path(str(base_path)).joinpath('books/user_books')
user_book_1 = Path(str(base_path)).joinpath('books/user_books/user_book_test1.txt')
fantasy_book = Path(str(base_path)).joinpath('books/train_books/fantasy.txt')
love_book = Path(str(base_path)).joinpath('books/train_books/love.txt')

book_train_model = Doc2Vec.load(str(model_path))

doc_list = get_doc_list(str(user_book_path))
doc_name_list = get_doc_names(str(user_book_path))

user_books = tokenize(doc_list)

print('User Data Loading Finished')
print('Files: {0}, type: {1}'.format(len(user_books), type(user_books)))

user_model = train_model.train_model_for_user_books(user_books)

fantasy_train = get_vectors_by_category(book_train_model, int(0.1))
love_train = get_vectors_by_category(book_train_model, int(1.5))
user_test = get_vectors_by_category(user_model, int(0.3))

temp = np.reshape(fantasy_train, (2, -2))
temp1 = np.reshape(love_train, (2, -2))
temp2 = np.reshape(user_test, (2, -2))

coef1 = np.sum(temp) / temp.size
coef2 = np.sum(temp1) / temp1.size

print("==============================================================")

lin_clf.fit(temp2, [int(coef1 * 10), int(coef2 * 10)])

temp_predict = lin_clf.predict(temp)
temp_coef = (np.sum(temp_predict) / np.size(temp_predict))/10

temp1_predict = lin_clf.predict(temp1)
temp1_coef = (np.sum(temp1_predict) / np.size(temp1_predict))/10

print("Коэффициенты для фэнтезийного жанра: " + str(temp_coef))
print("Коэффициенты для любовного жанра: " + str(temp1_coef))
