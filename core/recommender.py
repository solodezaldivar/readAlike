import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from core.library import Library, Book
from core.vectorizer import Vectorizer
from core.dimensionality_reducer import DimensionalityReducer
from core.ann import Ann
import torch


class Recommender:
    def __init__(self, lib: Library, n_dim_reduction: int = 100, n_trees: int = 10):
        self.__lib = lib
        self.__vectorizer = Vectorizer(lib)
        self.__reducer = DimensionalityReducer(self.__vectorizer.tfidf_matrix, n_dim_reduction)
        self.__ann = Ann(self.__reducer.reduced_matrix, n_trees)

    def __get_recommendations_by_indices(self, indices: list[int]) -> list[Book]:
        return [self.__lib.books[idx] for idx in indices]

    def __get_recommendations_by_scores(self, scores: np.ndarray, n_recommendations: int) -> tuple[list[Book], list[float]]:
        sim_scores_indices = np.argsort(scores)[::-1][1:n_recommendations + 1]
        return self.__get_recommendations_by_indices(sim_scores_indices), scores[sim_scores_indices].tolist()

    def recommend(self, book: Book, n_recommendations: int = 5) -> tuple[tuple[list[Book], list[float]], tuple[list[Book], list[float]], list[tuple[list[Book], float]]]:
        book_idx = self.__lib.get_book_idx(book)

        # TD-IDF cosine
        tfidf_vector = self.__vectorizer.tfidf_matrix[book_idx] if book_idx else self.__vectorizer.tfidf_vectorize(book)
        tfidf_sim_scores = cosine_similarity(tfidf_vector, self.__vectorizer.tfidf_matrix).flatten()
        tfidf_recommendations = self.__get_recommendations_by_scores(tfidf_sim_scores, n_recommendations)

        # SBERT model
        sbert_vector = self.__vectorizer.sbert_embeddings[book_idx] if book_idx else self.__vectorizer.sbert_vectorize(book)
        if isinstance(sbert_vector, torch.Tensor):
            sbert_vector = sbert_vector.cpu().numpy()
        sbert_sim_scores = cosine_similarity([sbert_vector], self.__vectorizer.sbert_embeddings).flatten()
        sbert_recommendations = self.__get_recommendations_by_scores(sbert_sim_scores, n_recommendations)

        # ANN
        if book_idx:
            ann_results = self.__ann.get_nearest_neighbors_by_index(book_idx, n_recommendations + 1)
            indices, scores = ann_results
            indices, scores = indices[1:], scores[1:]
        else:
            reduced_vector = self.__reducer.reduce(tfidf_vector)
            ann_results = self.__ann.get_nearest_neighbors_by_vector(reduced_vector[0], n_recommendations + 1)
            indices, scores = ann_results
            indices, scores = indices[1:], scores[1:]
        ann_recommendations = [(self.__get_recommendations_by_indices([index])[0], score) for (index, score) in zip(indices,scores)]


        return tfidf_recommendations, sbert_recommendations, ann_recommendations

