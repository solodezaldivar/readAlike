import numpy as np

from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix


class DimensionalityReducer:
    def __init__(self, tfidf_matrix: csr_matrix, n_dim_reduction: int):
        tqdm.write("Initializing dimensionality reduction (SVD)...")
        self.__svd = TruncatedSVD(n_components=n_dim_reduction)
        self.reduced_matrix = self.__svd.fit_transform(tfidf_matrix)

    def reduce(self, tfidf_vector: csr_matrix) -> np.ndarray:
        return self.__svd.transform(tfidf_vector.toarray())
