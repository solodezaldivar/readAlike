import preprocess

from library import Library, Book
from vectorizer import Vectorizer
from recommender import Recommender

DATASET_PATH = './datasets/amazon_books_data.csv'  # https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
COLS = ['Title', 'description', 'authors', 'categories']
N_REDUCED_DATASET = 500

readAlike = preprocess.read_data(DATASET_PATH, COLS)
readAlike_preprocessed = preprocess.preprocess_data(readAlike)
readAlike_preprocessed_reduced = readAlike_preprocessed.head(N_REDUCED_DATASET)

readAlike_lib_reduced = Library(readAlike_preprocessed_reduced)
readAlike_vectorizer_reduced = Vectorizer(readAlike_lib_reduced)
readAlike_recommender_reduced = Recommender(readAlike_lib_reduced, readAlike_vectorizer_reduced)

input_book = Book(
    title='Whispers of the Wicked Saints',
    description="Julia Thomas finds her life spinning out of control after the death of her husband, Richard. "
                "Julia turns to her minister for comfort when she finds herself falling for him with a passion that "
                "is forbidden by the church. Heath Sparks is a man of God who is busy taking care of his quadriplegic "
                "wife who was seriously injured in a sever car accident. In an innocent effort to reach out to a "
                "lonely member of his church, Heath finds himself as the man and not the minister as Heath and Julia "
                "surrender their bodies to each other and face the wrath of God. Julia finds herself in over her head "
                "as she faces a deadly disease, the loss of her home and whispers about her wicked affair. Julia "
                "leaves the states offering her body as a living sacrifice in hopes of finding a cure while her heart "
                "remains thousands of miles away hoping to one day reunite with the man who holds it hostage.Whispers "
                "of the Wicked Saints is a once in a lifetime romance that is breath taking, defying all the rules of "
                "romance and bending the laws of love.",
    authors=['Veronica Haddon'],
    categories=['Fiction']
)

tfidf, sbert, ann = readAlike_recommender_reduced.recommend(input_book)

for idx in range(5):
    print(f"Recommendation {idx+1}:")
    print(f"TF-IDF Title: {tfidf[idx].title}")
    print(f"TF-IDF Categories: {tfidf[idx].categories}")
    print(f"SBERT Title: {sbert[idx].title}")
    print(f"SBERT Categories: {sbert[idx].categories}")
    print(f"ANN Title: {ann[idx].title}")
    print(f"ANN Categories: {ann[idx].categories}")
    print("")
