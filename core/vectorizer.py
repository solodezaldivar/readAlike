import numpy as np

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from torch import Tensor

from core.library import Library, Book


class Vectorizer:
    def __init__(self, lib: Library):
        tqdm.write("Initializing TF-IDF vectorization...")
        self.__tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.__tfidf.fit_transform(lib.get_combined_data())

        tqdm.write("Initializing SBERT embeddings...")
        self.__sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        self.sbert_embeddings = self.__sbert_encode_with_progress(lib.get_combined_data())

    def __sbert_encode_with_progress(self, books: list[str], chunk_size=100):
        embeddings = []
        for i in tqdm(range(0, len(books), chunk_size), desc="SBERT embeddings"):
            chunk = books[i:i + chunk_size]
            chunk_embeddings = self.__sbert.encode(chunk, convert_to_tensor=True)
            embeddings.append(chunk_embeddings)
        return np.vstack([embedding.cpu().numpy() for embedding in embeddings])

    def tfidf_vectorize(self, book: Book) -> csr_matrix:
        return self.__tfidf.transform([book.get_combined_data()])

    def sbert_vectorize(self, book: Book) -> Tensor:
        return self.__sbert.encode(book.get_combined_data(), convert_to_tensor=True)
