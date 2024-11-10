from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from library import Library, Book
from vectorizer import Vectorizer


class Recommender:
    def __init__(self, lib: Library, vectorizer: Vectorizer):
        self.lib = lib
        self.vectorizer = vectorizer

    def recommend(self, book: Book, n_recommendations: int = 5) -> tuple[list[Book], list[Book], list[Book]]:
        # TD-IDF cosine
        sim_scores_tfidf = cosine_similarity(self.vectorizer.get_tfidf_vector(book, self.lib), self.vectorizer.tfidf_matrix).flatten()
        tfidf_sim_scores_indices = np.argsort(sim_scores_tfidf)[::-1][1:n_recommendations + 1]
        tfidf_recommendations = [self.lib.books[idx] for idx in tfidf_sim_scores_indices]

        # SBERT model
        sim_scores_sbert = cosine_similarity([self.vectorizer.get_sbert_vector(book, self.lib)], self.vectorizer.sbert_embeddings).flatten()
        sbert_sim_scores_indices = np.argsort(sim_scores_sbert)[::-1][1:n_recommendations + 1]
        sbert_recommendations = [self.lib.books[idx] for idx in sbert_sim_scores_indices]

        # ANN
        nearest_neighbors_indices = self.vectorizer.get_ann(book, self.lib, n_recommendations)[1:]
        ann_recommendations = [self.lib.books[idx] for idx in nearest_neighbors_indices]

        return tfidf_recommendations, sbert_recommendations, ann_recommendations

