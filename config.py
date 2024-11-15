# File Paths
DATASET_PATH = './datasets/amazon_books_data.csv'  # original source: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews

# DataFram Column Definitions
TITLE_COL = 'Title'
DESCRIPTION_COL = 'description'
AUTHORS_COL = 'authors'
CATEGORIES_COL = 'categories'
COLS = [TITLE_COL, DESCRIPTION_COL, AUTHORS_COL, CATEGORIES_COL]

# Dataset Settings
N_REDUCED_DATASET = 2500  # number of samples to use after preprocessing

# Recommender Algorithm Settings
N_DIM_REDUCTION = 400  # number of dimensions for SVD reduction
N_TREES = 100  # number of trees for the ANN index
N_RECOMMENDATIONS = 5  # number of recommendations to return
N_ITERATIONS = 10000  # number of algorithm iterations
