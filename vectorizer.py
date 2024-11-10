import numpy as np

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import Tensor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from annoy import AnnoyIndex

from library import Library, Book


class Vectorizer:
    def __init__(self, lib: Library, n_dim_reduction: int = 100, n_trees: int = 10):
        self.__tfidf = TfidfVectorizer(stop_words='english')
        self.__sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        self.__svd = TruncatedSVD(n_components=n_dim_reduction)

        self.tfidf_matrix = self.__tfidf.fit_transform(lib.get_combined_data())
        self.sbert_embeddings = self.__sbert_vectorize(lib)

        self.__ann_matrix = self.__svd.fit_transform(self.tfidf_matrix)
        self.__ann_indices = self.__get_ann_indices(n_trees)

    def __tfidf_vectorize(self, book: Book) -> csr_matrix:
        return self.__tfidf.transform([book.get_combined_data()])

    def __sbert_vectorize(self, input_data: Library | Book) -> Tensor:
        return self.__sbert.encode(input_data.get_combined_data(), convert_to_tensor=True)

    def __svd_reduce(self, book: Book) -> np.ndarray:
        return self.__svd.transform(self.__tfidf_vectorize(book).toarray())

    def __get_ann_indices(self, n_trees: int) -> AnnoyIndex:
        n_dim = self.__ann_matrix.shape[1]
        ann_indices = AnnoyIndex(n_dim, 'angular')
        for i in range(self.__ann_matrix.shape[0]):
            ann_indices.add_item(i, self.__ann_matrix[i])
        ann_indices.build(n_trees)
        return ann_indices

    def get_tfidf_vector(self, book: Book, lib: Library) -> csr_matrix:
        book_idx = lib.get_book_idx(book)
        if book_idx:
            return self.tfidf_matrix[book_idx]
        return self.__tfidf_vectorize(book)

    def get_sbert_vector(self, book: Book, lib: Library) -> Tensor:
        book_idx = lib.get_book_idx(book)
        if book_idx:
            return self.sbert_embeddings[book_idx]
        return self.__sbert_vectorize(book)

    def get_ann(self, book: Book, lib: Library, n_neighbors) -> list[int]:
        book_idx = lib.get_book_idx(book)
        if book_idx:
            return self.__ann_indices.get_nns_by_item(book_idx, n_neighbors + 1)
        return self.__ann_indices.get_nns_by_vector(self.__svd_reduce(book)[0], n_neighbors + 1)
