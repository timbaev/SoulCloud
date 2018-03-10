from doc2vec_module import load
from doc2vec_module.constants import FileConstants
from doc2vec_module.train_model import get_model_for_genre

love_prop = getattr(FileConstants, "LOVE_BOOK_TEST")
fantasy_prop = getattr(FileConstants, "FANTASY_BOOK_TEST")
love_model_prop = getattr(FileConstants, "MODEL_LOVE")
fantasy_model_prop = getattr(FileConstants,"MODEL_FANTASY")

love_documents = load.get_doc(love_prop.fget(FileConstants()))
fantasy_documents = load.get_doc(fantasy_prop.fget(FileConstants()))
print('Данные для обучения загружены')

print(" ")

print("Количество документов для любви : {0}, тип: {1}".format(len(love_documents), type(love_documents)))
print("Количество документов для фантастики : {0}, тип: {1}".format(len(fantasy_documents), type(fantasy_documents)))
print(" ")

print("Обучаем модель для любовного жанра")
love_model = get_model_for_genre(love_documents)
print("Модель обучена")
print(" ")

print("Обучаем модель для жанра фантастики")
fantasy_model = get_model_for_genre(fantasy_documents)
print("Модель обучена")
print(" ")

love_model.save(love_model_prop.fget(FileConstants()))
fantasy_model.save(fantasy_model_prop.fget(FileConstants()))
print("Модели сохранены")