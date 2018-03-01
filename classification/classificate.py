from sklearn import svm

from classification import train_model
from load import get_doc_names, get_doc_list
from .numpy_proceed import get_vectors_by_category
from gensim.models import Doc2Vec
from classification.tokenization import tokenize

lin_clf = svm.LinearSVC()

book_train_model = Doc2Vec.load('./model/book_model.doc2vec')
user_book_path = './books/user_books'

doc_list = get_doc_list(user_book_path)
doc_name_list = get_doc_names(user_book_path)

user_books = tokenize(doc_list, doc_name_list)

print('User Data Loading Finished')
print('Files: {0}, type: {1}'.format(len(user_books), type(user_books)))

user_model = train_model.train_model_for_user_books(user_books)

fantasy_train = get_vectors_by_category(book_train_model, str('./books/train_books/fantasy.txt'))
love_train = get_vectors_by_category(book_train_model, str('./books/trains_books/love.txt'))
user_test = get_vectors_by_category(user_model, str('./books/user_books/user_book_test1.txt'))

lin_clf.fit(fantasy_train, love_train)

print("Вероятность для жанра любви: ".format(lin_clf.score(love_train, user_test)))
print("Вероятность для жанра фантастики: ".format(lin_clf.score(fantasy_train, user_test)))