import numpy as np

from annoy import AnnoyIndex


class Ann:
    def __init__(self, reduced_matrix: np.ndarray, n_trees: int):
        self.__ann_indices = self.__get_ann_indices(reduced_matrix, n_trees)

    @staticmethod
    def __get_ann_indices(reduced_matrix: np.ndarray, n_trees: int) -> AnnoyIndex:
        n_dim = reduced_matrix.shape[1]
        ann_indices = AnnoyIndex(n_dim, 'angular')
        for i in range(reduced_matrix.shape[0]):
            ann_indices.add_item(i, reduced_matrix[i])
        ann_indices.build(n_trees)
        return ann_indices

    def get_nearest_neighbors_by_index(self, index: int, n_neighbors: int) -> list[int]:
        return self.__ann_indices.get_nns_by_item(index, n_neighbors)

    def get_nearest_neighbors_by_vector(self, vector: np.ndarray, n_neighbors: int) -> list[int]:
        return self.__ann_indices.get_nns_by_vector(vector, n_neighbors)
