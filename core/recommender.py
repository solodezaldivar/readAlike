import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from core.library import Library, Book
from core.vectorizer import Vectorizer
from core.dimensionality_reducer import DimensionalityReducer
from core.ann import Ann


class Recommender:
    def __init__(self, lib: Library, n_dim_reduction: int = 100, n_trees: int = 10):
        self.__lib = lib
        self.__vectorizer = Vectorizer(lib)
        self.__reducer = DimensionalityReducer(self.__vectorizer.tfidf_matrix, n_dim_reduction)
        self.__ann = Ann(self.__reducer.reduced_matrix, n_trees)

    def __get_recommendations_by_indices(self, indices: list[int]) -> list[Book]:
        return [self.__lib.books[idx] for idx in indices]

    def __get_recommendations_by_scores(self, scores: np.ndarray, n_recommendations: int) -> list[Book]:
        sim_scores_indices = np.argsort(scores)[::-1][1:n_recommendations + 1]
        return self.__get_recommendations_by_indices(sim_scores_indices)

    def recommend(self, book: Book, n_recommendations: int = 5) -> tuple[list[Book], list[Book], list[Book]]:
        book_idx = self.__lib.get_book_idx(book)

        # TD-IDF cosine
        tfidf_vector = self.__vectorizer.tfidf_matrix[book_idx] if book_idx else self.__vectorizer.tfidf_vectorize(book)
        tfidf_sim_scores = cosine_similarity(tfidf_vector, self.__vectorizer.tfidf_matrix).flatten()
        tfidf_recommendations = self.__get_recommendations_by_scores(tfidf_sim_scores, n_recommendations)

        # SBERT model
        sbert_vector = self.__vectorizer.sbert_embeddings[book_idx] if book_idx else self.__vectorizer.sbert_vectorize(book)
        sbert_sim_scores = cosine_similarity([sbert_vector], self.__vectorizer.sbert_embeddings).flatten()
        sbert_recommendations = self.__get_recommendations_by_scores(sbert_sim_scores, n_recommendations)

        # ANN
        if book_idx:
            ann_indices = self.__ann.get_nearest_neighbors_by_index(book_idx, n_recommendations + 1)[1:]
        else:
            reduced_vector = self.__reducer.reduce(tfidf_vector)
            ann_indices = self.__ann.get_nearest_neighbors_by_vector(reduced_vector[0], n_recommendations + 1)[1:]
        ann_recommendations = self.__get_recommendations_by_indices(ann_indices)

        return tfidf_recommendations, sbert_recommendations, ann_recommendations

