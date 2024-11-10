import random

from tqdm import tqdm

from preprocessing.preprocessor import Preprocessor
from core.library import Library
from core.recommender import Recommender
from config import DATASET_PATH, COLS, N_REDUCED_DATASET, N_DIM_REDUCTION, N_TREES, N_RECOMMENDATIONS, N_ITERATIONS


def setup_recommender(use_reduced_dataset: bool) -> tuple[Library, Recommender]:
    preprocessor = Preprocessor(DATASET_PATH, COLS)
    preprocessed_data = preprocessor.preprocess_data()
    if use_reduced_dataset:
        preprocessed_data = preprocessed_data.head(N_REDUCED_DATASET)
    library = Library(preprocessed_data)
    recommender = Recommender(library, n_dim_reduction=N_DIM_REDUCTION, n_trees=N_TREES)
    return library, recommender


def run_recommender(library: Library, recommender: Recommender, n_iterations=10):
    tqdm.write("Starting recommender...")
    all_recommendations = []

    for _ in tqdm(range(n_iterations), desc="Generating recommendations"):
        random_book = random.choice(library.books)
        tfidf_recs, sbert_recs, ann_recs = recommender.recommend(random_book, n_recommendations=N_RECOMMENDATIONS)
        recommendations = {
            "book": random_book,
            "tfidf": tfidf_recs,
            "sbert": sbert_recs,
            "ann": ann_recs
        }
        all_recommendations.append(recommendations)

        # TODO: Evaluation for each approach separately, and across approaches
        #  (think about file structure to split up meaningfully)

        # IDEA 1: average cosine similarity
        # def average_cosine_similarity(recommended_indices, cosine_sim_matrix, ref_idx, k):
        #     similarity_scores = [cosine_sim_matrix[ref_idx][idx] for idx in recommended_indices[:k]]
        #     avg_similarity = sum(similarity_scores) / k
        #     return avg_similarity

        # IDEA 2: average precision @ K
        # def average_precision_at_k(ref_idx, recommended_indices, cosine_sim_matrix, relevance_threshold, k):
        #     score = 0.0
        #     num_hits = 0.0
        #     for i, idx in enumerate(recommended_indices[:k]):
        #         if cosine_sim_matrix[ref_idx][idx] > relevance_threshold:
        #             num_hits += 1.0
        #             score += num_hits / (i + 1.0)
        #     return score / min(len(recommended_indices), k)

        # IDEA 3: intra list similarity -> diversity
        # from itertools import combinations
        #
        # def intra_list_similarity(recommended_indices, cosine_sim_matrix):
        #     pairs = list(combinations(recommended_indices, 2))
        #     similarities = [cosine_sim_matrix[i][j] for i, j in pairs]
        #     return sum(similarities) / len(similarities) if similarities else 0

        # IDEA 4: novelty
        # def novelty_score(recommended_indices, movie_popularity):
        #     popularities = [movie_popularity[idx] for idx in recommended_indices]
        #     avg_popularity = sum(popularities) / len(popularities)
        #     return avg_popularity  # Lower score indicates higher novelty


if __name__ == "__main__":
    read_alike_library, read_alike_recommender = setup_recommender(use_reduced_dataset=True)
    run_recommender(read_alike_library, read_alike_recommender, n_iterations=N_ITERATIONS)
