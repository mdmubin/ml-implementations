from numpy import sqrt, square
from statistics import mean, mode


class KNN:
    """
    DESCRIPTION
    -----------
    A class to abstract away the KNN algorithm tasks. Contains methods for both
    the KNN Classification and the Regression algorithms.
    """

    def __init__(self, trainX, trainY, k_neighbors=5):
        self.k = k_neighbors
        self.data_x = trainX
        self.data_y = trainY

    @staticmethod
    def dist_func(p1, p2):
        """
        DESCRIPTION
        -----------
        Get the euclidean distance between two points p1 and p2
        """

        data_len = len(p1)
        square_sum = 0
        for i in range(data_len):
            square_sum += square(p1[i] - p2[i])
        return sqrt(square_sum)

    def __knn(self, dataX):
        """
        DESCRIPTION
        -----------
        The main KNN algorithm. This method calculates the distance between
        every training data and the given data, dataX, and then returns the
        k number of nearest values along with the distances between them.
        """

        distances = []
        train_len = len(self.data_y)

        for i in range(train_len):
            distances.append([i, KNN.dist_func(self.data_x[i], dataX)])

        k_nearest = sorted(distances, key=lambda item: item[1])[: self.k]

        return k_nearest

    def predict_class(self, dataX):
        """
        DESCRIPTION
        -----------
        Predict the class of a given variable based on the K nearest values to
        this variable by finding the most common variables within the k values.
        """

        k_nearest = self.__knn(dataX)

        knn_labels = [self.data_y[i] for i, _ in k_nearest]

        return mode(knn_labels)

    def predict_regressed(self, dataX):
        """
        DESCRIPTION
        -----------
        Estimate the value of a given variable based on the K nearest values to
        this variable through regression, by finding the mean of the k values.
        """
        k_nearest = self.__knn(dataX)

        knn_labels = [self.data_y[i] for i, _ in k_nearest]

        return mean(knn_labels)
