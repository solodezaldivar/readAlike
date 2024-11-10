from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from torch import Tensor

from library import Library, Book


class Vectorizer:
    def __init__(self, lib: Library):
        self.__tfidf = TfidfVectorizer(stop_words='english')
        self.__sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
        self.tfidf_matrix = self.__tfidf.fit_transform(lib.get_combined_data())
        self.sbert_embeddings = self.sbert_vectorize(lib)

    def tfidf_vectorize(self, book: Book) -> csr_matrix:
        return self.__tfidf.transform([book.get_combined_data()])

    def sbert_vectorize(self, input_data: Library | Book) -> Tensor:
        return self.__sbert.encode(input_data.get_combined_data(), convert_to_tensor=True)
