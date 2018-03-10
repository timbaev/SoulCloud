from pathlib import Path

class FileConstants(object):

    base_path = Path(__file__).parents[1]

    @property
    def MODEL_LOVE(self) -> str:
        return str( Path(str(FileConstants.base_path) ).joinpath('models/love_model.doc2vec'))

    @property
    def MODEL_FANTASY(self) -> str:
        return str( Path(str(FileConstants.base_path) ).joinpath('models/fantasy_model.doc2vec'))

    @property
    def LOVE_BOOK_TEST(self) -> str:
        return str( Path(str(FileConstants.base_path) ).joinpath('books/train_books/love'))

    @property
    def FANTASY_BOOK_TEST(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/train_books/fantasy'))

    @property
    def USER_BOOK_TEST(self) -> str:
        return str(Path(str(FileConstants.base_path)).joinpath('books/user_books'))
