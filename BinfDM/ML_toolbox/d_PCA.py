# Written by Xiuxia
# 2018 Spring and 2019 Spring

import numpy as np
from numpy import linalg as LA

class d_PCA:
    """"
    Class for Principal Component Analysis.

    Attributes:
    """

    def __init__(self, num_of_components, corr_logic):
        self.num_of_components = num_of_components
        self.corr_logic = corr_logic

    def fit_transform(self, x):
        columnMean = x.mean(axis=0)
        columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
        xMeanCentered = x - columnMeanAll

        # use mean_centered data or standardized mean_centered data
        if not self.corr_logic:
            dataForPca = xMeanCentered
        else:
            columnSD = np.std(x, axis=0)
            columnSDAll = np.tile(columnSD, reps=(x.shape[0], 1))
            dataForPca = x / columnSDAll

        # get covariance matrix of the data
        covarianceMatrix = np.cov(dataForPca, rowvar=False)

        # eigendecomposition of the covariance matrix
        eigenValues, eigenVectors = LA.eig(covarianceMatrix)

        eigenValues = eigenValues.real
        eigenVectors = eigenVectors.real

        # sort eigenvalues in descending order
        II = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[II]
        eigenVectors = eigenVectors[:, II]

        # percentage of variance explained by each PC
        totalVariance = sum(eigenValues)
        percentVariance = np.zeros(len(eigenValues))
        for i in range(len(eigenValues)):
            percentVariance[i] = eigenValues[i] / totalVariance

        # get scores
        pcaScores = np.matmul(dataForPca, eigenVectors)

        # collect PCA results
        pcaResults = {'data': x,
                      'mean_centered_data': xMeanCentered,
                      'percent_variance': percentVariance,
                      'loadings': eigenVectors,
                      'scores': pcaScores,
                      'data_after_pretreatment': dataForPca}

        return pcaResults