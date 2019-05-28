from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
from PreTraitement import PreTraitement

class Model:

    def __init__(self, rerun_pre_trait = False):
        self.pre_traitement = PreTraitement("training2", rerun_pre_trait)
        print("pretraitement terminé")
        self.data_matrix = self.pre_traitement.data_matrix
        self.x, self.y = self.get_x_y_from_data(self.data_matrix, self.pre_traitement.header)
        print(self.x)
        print(self.y)
        self.x_svd, self.svd_obj = self.reduction_dimention(50)
        print("class: " + str(self.svd_obj.__class__))
        print("shape: " + str(self.x_svd.shape))

    def get_x_y_from_data(self, data_matrix, header):

        y_index = header.index("volume")
        print("y_index: " + str(y_index))
        print("len matrix: " + str(data_matrix.shape))
        index_to_charge = list(range(len(header)))
        del index_to_charge[y_index]
        print("index to charge: " + str(index_to_charge))
        print(data_matrix[:, index_to_charge].shape)
        print(data_matrix[:, index_to_charge][-1])
        return data_matrix[:, index_to_charge], data_matrix[y_index]

    def reduction_dimention(self, nb_dimentions):
        svd = TruncatedSVD(n_components=nb_dimentions)
        x_svd = svd.fit_transform(self.x)

        return x_svd, svd



if __name__ == "__main__":
    model = Model(True)