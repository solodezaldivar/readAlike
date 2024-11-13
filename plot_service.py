import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn

def plot_intra_scores(scores, name):
    plt.figure(figsize=(10,6))
    seaborn.histplot(scores, bins=30, kde=True, color='skyblue', alpha=0.7)
    plt.title(f'Distribution of {name}')
    plt.xlabel(f'{name}')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_distribution(all_recommendations):
    tfidf_scores = [rec["avg_tfidf_score"] for rec in all_recommendations]
    sbert_scores =  [rec["avg_sbert_score"] for rec in all_recommendations]
    ann_scores =  [rec["avg_ann_score"] for rec in all_recommendations]

    score_data = {
        "TF-IDF": tfidf_scores,
        "SBERT": sbert_scores,
        "ANN": ann_scores,
    }

    plot_data = [(key, value) for key, values in score_data.items() for value in values]
    methods, scores = zip(*plot_data)

    # Boxplot
    plt.figure(figsize=(12, 6))
    seaborn.histplot(tfidf_scores, label='TF-IDF', kde=True, bins=20, color='blue', alpha=0.5)
    seaborn.histplot(sbert_scores, label='SBERT', kde=True, bins=20, color='orange', alpha=0.5)
    seaborn.histplot(ann_scores, label='ANN', kde=True, bins=20, color='red', alpha=0.5)
    

    plt.title('Distribution of Average Scores for Each Recommender')
    plt.xlabel('Average Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("distribution_of_scores_boxplot.png")
    plt.show()