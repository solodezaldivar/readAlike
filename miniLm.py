from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from torch import Tensor


class Book:
    def __init__(self, title, description, genre):
        self.title = title
        self.description = description
        self.genre = genre

    def get_combined_text(self):
        return f"{self.title}. {self.description}. Genre: {self.genre}"


def get_similar_books(book: Book, embeddings: Tensor, books: list[Book], top_n: int = 5):
    input_embedding = model.encode(book.get_combined_text(), convert_to_tensor=True)
    similarities = cosine_similarity([input_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    return [books[i] for i in top_indices]


# Dataset
print("reading dataset...")
readAlikeDataFrame = pd.read_csv('./datasets/BooksDatasetClean.csv',
                                 usecols=['Description', 'Category', 'Title'])
print("reading completed")

# Preprocessing
print("preprocessing dataset...")
readAlikeDataFrame['Description'] = readAlikeDataFrame["Description"].replace(r'', np.nan, regex=True)
readAlikeDataFrame["Category"] = readAlikeDataFrame["Category"].replace(r'', np.nan, regex=True)
readAlikeDataFrame.dropna(subset=["Description"], inplace=True)
readAlikeDataFrame.dropna(subset=["Category"], inplace=True)
readAlikeDataFrame["description_length"] = readAlikeDataFrame["Description"].apply(lambda x: len(x.split()))
readAlikeDataFrame = readAlikeDataFrame[readAlikeDataFrame["description_length"] >= 10]  # drop books with too short description
readAlikeDataFrame.drop_duplicates(subset='Title', keep='first', inplace=True)  # drop duplicate books
readAlikeDataFrame['Genre_and_Description'] = readAlikeDataFrame['Category'] + ' ' + readAlikeDataFrame['Description']
readAlikeDataFrame.reset_index(drop=True, inplace=True)
readAlikeDataFrame_reduced = readAlikeDataFrame.head(500)
book_objects = [Book(row['Title'], row['Description'], row['Category']) for _, row in readAlikeDataFrame_reduced.iterrows()]
print("preprocessing completed")

# Embeddings
print("computing embeddings...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
book_info = [book.get_combined_text() for book in book_objects]
book_embeddings = model.encode(book_info, convert_to_tensor=True)
print("embeddings completed")

# Example
print("running example...")
book_input = Book(
    "Journey Through Heartsongs",
    "Mattie J. T. Stepanek takes us on a Journey Through Heartsongs with more of his moving poems. These poems share the rare wisdom that Mattie has acquired through his struggle with a rare form of muscular dystrophy and the death of his three siblings from the same disease. His life view was one of love and generosity and as a poet and a peacemaker, his desire was to bring his message of peace to as many people as possible.",
    "Poetry , Subjects & Themes , Inspirational & Religious")
similar_books = get_similar_books(book_input, book_embeddings, book_objects, top_n=5)
print("example completed\n")
for idx, book in enumerate(similar_books, start=1):
    print(f"Recommendation {idx}:")
    print(f"Title: {book.title}")
    print(f"Description: {book.description}")
    print(f"Genre: {book.genre}")
    print("")
