import random
import numpy as np
import csv
from tqdm import tqdm

from plot_service import plot_distribution, plot_intra_scores
from preprocessing.preprocessor import Preprocessor
from core.library import Library
from core.recommender import Recommender
from config import DATASET_PATH, COLS, N_REDUCED_DATASET, N_DIM_REDUCTION, N_TREES, N_RECOMMENDATIONS, N_ITERATIONS
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")

print("Path to dataset files:", path)


def setup_recommender(use_reduced_dataset: bool) -> tuple[Library, Recommender]:
    preprocessor = Preprocessor(path+"/books_data.csv", COLS)
    preprocessed_data = preprocessor.preprocess_data()
    if use_reduced_dataset:
        preprocessed_data = preprocessed_data.head(N_REDUCED_DATASET)
    library = Library(preprocessed_data)
    recommender = Recommender(library, n_dim_reduction=N_DIM_REDUCTION, n_trees=N_TREES)
    return library, recommender


def run_recommender(library: Library, recommender: Recommender, n_iterations=10):
    tqdm.write("Starting recommender...")
    all_recommendations = []

    # select n random books from user library
    for _ in tqdm(range(n_iterations), desc="Generating recommendations"):
        random_book = random.choice(library.books)
        tfidf_recs, sbert_recs, ann_recs = recommender.recommend(random_book, n_recommendations=N_RECOMMENDATIONS)
        ann_books = [book for book, _ in ann_recs]
        ann_scores = [score for _, score in ann_recs]
        
        # Normalize ANN scores between 0 and 1
        min_ann_score = min(ann_scores)
        max_ann_score = max(ann_scores)
        ann_scores_normalized = [(score - min_ann_score) / (max_ann_score - min_ann_score) for score in ann_scores]
        recommendations = {
            "book": random_book,
            "tfidf_recs": tfidf_recs[0],
            "avg_tfidf_score": np.average(tfidf_recs[1]),
            "sbert_recs": sbert_recs[0],
            "avg_sbert_score": np.average(sbert_recs[1]),
            "ann_recs": ann_books,
            "avg_ann_score": np.average(ann_scores_normalized)
        }
        all_recommendations.append(recommendations)
        
    tfidf_scores = [score["avg_tfidf_score"] for score in all_recommendations]
    sbert_scores = [score["avg_sbert_score"] for score in all_recommendations]
    ann_scores = [score["avg_ann_score"] for score in all_recommendations]
    # Plotting
    plot_distribution(all_recommendations)
    
    
    plot_intra_scores(tfidf_scores, "Avg TF-IDF")
    plot_intra_scores(sbert_scores, "Avg SBERTH")
    plot_intra_scores(ann_scores_normalized, "Avg ANN Distance")
    


if __name__ == "__main__":
    read_alike_library, read_alike_recommender = setup_recommender(use_reduced_dataset=False)
    run_recommender(read_alike_library, read_alike_recommender, n_iterations=N_ITERATIONS)
