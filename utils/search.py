import torch
from utils.utils import distance_matrix
from scipy.spatial.distance import cdist
import numpy as np
from numba import jit
import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F



# @jit(nopython=True)
# def mahalanobis_distance(patches, coreset, inv_cov_matrix):
#     """
#     Calculate the Mahalanobis distance using numba
#     """
#     n_patches = patches.shape[0]
#     n_samples = coreset.shape[0]
#     distances = np.empty(shape=(n_samples, n_patches))#, dtype=np.float16)

#     for l in range(n_patches):
#         for i in range(n_samples):
#             diff = coreset[i] - patches[l]
#             # print(diff.shape)
#             a = np.dot(diff, inv_cov_matrix)
#             # print(a.shape)
#             b = np.dot(a, diff.T)
#             # print(b)
#             distances[i,l] = np.sqrt(b)#np.dot(np.dot(diff, inv_cov_matrix), diff)
#             # distances[i,l] = np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff))
#     return distances

import numpy as np
from numba import jit

# @jit(nopython=True)
def mahalanobis(x, y, inv_cov):
    diff = x - y
    return np.sqrt(diff @ inv_cov @ diff.T)

def mahalanobis_distances(X, y, cov):
    # convert to numpy
    X = X.cpu().numpy().astype(np.double)#.T
    print('X: ', X.shape)
    
    y = y.cpu().numpy().astype(np.double)#.T
    print('y: ', y.shape)
    # cov = cov.cpu().numpy()
    # inv_cov = np.linalg.inv(cov)
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # print()
        a = mahalanobis(X[i], y, cov)
        print(a.shape)
        distances[i] = mahalanobis(X[i], y, cov)
    return distances

@jit(nopython=True)
def mahalanobis_taylor(x, y, inv_cov):
    diff = x - y
    a = np.dot(diff, inv_cov)
    b = np.dot(a, diff.T)
    c = np.dot(a, a.T)
    d = np.dot(c, diff.T)
    return np.sqrt(b) + 0.5 * np.trace(inv_cov @ d) - 0.25 * np.trace(inv_cov @ c @ inv_cov @ b)

def mahalanobis_distances_taylor(X, y, cov):
    inv_cov = np.linalg.inv(cov)
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances[i] = mahalanobis_taylor(X[i], y, inv_cov)
    return distances

class NN():
    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):
    def __init__(self, X=None, Y=None, k=3, p=None, metric='euclidean', inv_cov=None):
        self.k = k
        self.p = p
        # self.metrices = { 
        #             0:'euclidean', # 0.88
        #             1:'minkowski', # nur mit p spannend
        #             2:'cityblock', # manhattan
        #             3:'chebyshev',
        #             4:'cosine',
        #             5:'correlation',
        #             6:'hamming',
        #             7:'jaccard',
        #             8:'braycurtis',
        #             9:'canberra',
        #             10:'jensenshannon',
        #             # 11:'matching', # sysnonym for hamming
        #             11:'dice',
        #             12:'kulczynski1',
        #             13:'rogerstanimoto',
        #             14:'russellrao',
        #             15:'sokalmichener',
        #             16:'sokalsneath',
        #             # 18:'wminkowski',
        #             17:'mahalanobis',
        #             18:'seuclidean',
        #             19:'sqeuclidean',
        #             }
        # self.metrices_dict = {metric: i for i, metric in enumerate(metrices)}
        self.metric = metric
        print(f"\nUsing metric: {self.metric}\n")
        if self.metric == 'mahalanobis':
            assert inv_cov is not None, "Need to provide inverse covariance matrix for mahalanobis distance"
            self.inv_cov = inv_cov
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        # print('self.k: ', self.k)
        if self.p is None and not self.metric == 'mahalanobis':
            dist = torch.from_numpy(cdist(x, self.train_pts, metric=self.metric))
        elif self.metric == 'mahalanobis':
            # dist = torch.from_numpy(cdist(x, self.train_pts, metric=self.metric, VI=self.inv_cov))
            dist = mahalanobis_distances(x, self.train_pts, self.inv_cov)
        else:
            dist = torch.from_numpy(cdist(x, self.train_pts, metric=self.metric, p=self.p))
        knn = dist.topk(self.k, largest=False)
        return knn


### from patchcore original

class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            # print('')
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        # print(filename)
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            print('resetting index')
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)

class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: [dict of np.arrays] List of np.arrays
                 corresponding to the test features generated by
                 some backbone network.
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        # print('query_distances.shape: ', query_distances.shape)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = True, # Edited
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
            
class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


