import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
from ML_toolbox import d_PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():

    # ----------------------------------------------------------------
    # parameters
    # ----------------------------------------------------------------
    marker_size = 1

    # example indices:
    # 1 = correlated x1 and x2
    # 2 = uncorrelated x1 and x2
    # 3 = correlated x1 and x2, uncorrelated x1 and x3
    # 4 = cell line data
    # 5 = use sklearn PCA on cell line data

    example_index = 3

    if example_index == 1:
        # ----------------------------------------------------------------
        # correlated x1 and x2
        # ----------------------------------------------------------------

        # 1. generate the raw data
        x1 = np.arange(start=0, stop=20, step=0.1)
        x2 = 2 * x1 + np.random.normal(loc=0, scale=0.5, size=len(x1))

        # 2. visualize the raw data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x1, x2, color='blue', s=marker_size)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('raw random data')
        fig.show()

        # 3. do PCA
        dataForAnalysis = np.column_stack((x1, x2))

        useCorr = True
        obj_PCA = d_PCA.d_PCA(num_of_components=2, corr_logic=useCorr)
        myPCAResults = obj_PCA.fit_transform(x=dataForAnalysis)

        # scree plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('scree plot')
        ax.scatter(range(len(myPCAResults['percent_variance'])), myPCAResults['percent_variance'],
                   color='blue', s=marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        fig.show()

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('scores plot')
        ax.scatter(myPCAResults['scores'][:, 0], myPCAResults['scores'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('loadings plot')
        ax.scatter(myPCAResults['loadings'][:, 0], myPCAResults['loadings'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # raw, pretreated data, and PC axis plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('raw, pretreated data, and PC axis')
        ax.scatter(x1, x2, color='blue', label='raw', s=marker_size)
        ax.scatter(myPCAResults['data_after_pretreatment'][:, 0],
                   myPCAResults['data_after_pretreatment'][:, 1],
                   color='green', label='pretreated', s=marker_size)
        k = 3
        ax.plot([0, k * myPCAResults['loadings'][0, 0]], [0, k * myPCAResults['loadings'][1, 0]],
                color='magenta', linewidth=3, label='PC axis')
        ax.plot([0, k * myPCAResults['loadings'][0, 1]], [0, k * myPCAResults['loadings'][1, 1]],
                color='magenta', linewidth=3)
        ax.set_aspect('equal', 'box')
        ax.legend()
        fig.show()

        # keep only the first dimension
        dataReconstructed = np.matmul(myPCAResults['scores'][:, 0].reshape((200, 1)),
                                      myPCAResults['loadings'][:, 0].reshape((1, 2)))
        # reconstructed data plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('reconstructed data')
        ax.scatter(dataReconstructed[:, 0], dataReconstructed[:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        fig.show()

    elif example_index == 2:
        # ----------------------------------------------------------------
        # PCA on independent x1 and x2
        # ----------------------------------------------------------------
        # 1. generate raw data
        num_of_samples = 100

        x1 = np.random.normal(loc=0, scale=0.5, size=num_of_samples)
        x2 = np.random.normal(loc=0, scale=0.5, size=num_of_samples)

        # 2. visualize the raw data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(x1, x2, color='blue', s=marker_size)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('raw data')
        fig.show()

        # 3. do PCA
        dataForAnalysis = np.column_stack((x1, x2))

        useCorr = True
        obj_PCA = d_PCA.d_PCA(num_of_components=2, corr_logic=useCorr)
        myPCAResults = obj_PCA.fit_transform(x=dataForAnalysis)

        # scree plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('scree plot')
        ax.scatter(range(len(myPCAResults['percent_variance'])), myPCAResults['percent_variance'],
                   color='blue', s=marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        fig.show()

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('scores plot')
        ax.scatter(myPCAResults['scores'][:, 0], myPCAResults['scores'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('loadings plot')
        ax.scatter(myPCAResults['loadings'][:, 0], myPCAResults['loadings'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # raw and pretreated data and PC axis
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('raw and pretreated data and PC axis')
        ax.scatter(x1, x2, color='blue', s=marker_size, label='raw')
        ax.scatter(myPCAResults['data_after_pretreatment'][:, 0],
                     myPCAResults['data_after_pretreatment'][:, 1],
                     color='green', s=marker_size, label='pretreated')
        k = 3
        ax.plot([0, (-1)*k*myPCAResults['loadings'][0, 0]],
                [0, (-1)*k*myPCAResults['loadings'][1, 0]],
                color='magenta', linewidth=3, label='PC axis')
        ax.plot([0, k * myPCAResults['loadings'][0, 1]], [0, k * myPCAResults['loadings'][1, 1]],
                color='magenta',linewidth=3)
        ax.set_aspect('equal', 'box')
        ax.legend()
        fig.show()

    elif example_index == 3:
        # ----------------------------------------------------------------
        # PCA on toy data
        # ----------------------------------------------------------------
        # 1. get the raw data
        in_file_name = "./data/dataset_1.csv"
        dataIn = pd.read_csv(in_file_name)

        # 2. visualize the raw data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("raw data")
        ax.scatter(dataIn['x'], dataIn['y'], color='blue', s=marker_size)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.show()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("raw data")
        ax.scatter(dataIn['x'], dataIn['z'], color='blue', s=marker_size)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        fig.show()

        # 3. do PCA
        dataForAnalysis = dataIn.values
        useCorr = True

        obj_PCA = d_PCA.d_PCA(num_of_components=2, corr_logic=useCorr)
        myPCAResults = obj_PCA.fit_transform(x=dataForAnalysis)

        # scree plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('scree plot')
        ax.scatter(range(len(myPCAResults['percent_variance'])),
                   myPCAResults['percent_variance'],
                   color='blue', s=marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('PC index')
        ax.set_ylabel('percent variance explained')
        fig.show()

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('scores plot')
        ax.scatter(myPCAResults['scores'][:, 0], myPCAResults['scores'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('loadings plot')
        ax.scatter(myPCAResults['loadings'][:, 0], myPCAResults['loadings'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()
        #plt.show()


    elif example_index == 4:
        # ----------------------------------------------------------------
        # PCA on real data
        # ----------------------------------------------------------------
        in_file_name = './data/SCLC_study_output_filtered.csv'
        dataIn = pd.read_csv(in_file_name, header=0, index_col=0)

        #dataForAnalysis = dataIn.as_matrix()
        dataForAnalysis = dataIn.values

        useCorr = True
        obj_PCA = d_PCA.d_PCA(num_of_components=2, corr_logic=useCorr)
        myPCAResults = obj_PCA.fit_transform(x=dataForAnalysis)

        # scree plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('scree plot')
        ax.scatter(range(len(myPCAResults['percent_variance'])),
                   myPCAResults['percent_variance'],
                   color='blue', s=marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel('PC index')
        fig.show()

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('scores plot')
        ax.scatter(myPCAResults['scores'][:, 0], myPCAResults['scores'][:, 1],
                   color='blue', s=marker_size, label='NSCLC')
        ax.scatter(myPCAResults['scores'][0:20, 0], myPCAResults['scores'][0:20, 1],
                   color='red', s=marker_size, label='SCLC')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
        fig.show()

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('loadings plot')
        ax.scatter(myPCAResults['loadings'][:, 0], myPCAResults['loadings'][:, 1],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        fig.show()

    elif example_index == 5:
        # ----------------------------------------------------------------
        # use sklearn
        # ----------------------------------------------------------------
        in_file_name = './data/SCLC_study_output_filtered.csv'
        dataIn = pd.read_csv(in_file_name, header=0, index_col=0)

        num_of_samples = dataIn.shape[0]
        num_of_variables = dataIn.shape[1]
        sample_names = dataIn.index.values
        variable_names = dataIn.columns.values

        # data pre-processing
        # standardize each variable by computing its z-score
        data_for_analysis_standardized = StandardScaler(with_mean=True, with_std=True).fit_transform(dataIn)

        # ---------------------------------------------------------
        # PCA
        # ---------------------------------------------------------
        num_of_components = min(num_of_samples, num_of_variables)

        obj_PCA = PCA(n_components=num_of_components)

        PCA_fit_results = obj_PCA.fit(data_for_analysis_standardized)

        # scree plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(range(len(PCA_fit_results.explained_variance_ratio_)),
                   PCA_fit_results.explained_variance_ratio_,
                   color='blue', s=marker_size)
        ax.set_ylim([-0.1, 1.1])
        ax.set_xlabel("principal component index")
        ax.set_ylabel("explained variance ratio")
        ax.set_title("scree plot")
        fig.show()
        out_file_name = "./results/PCA_scree_plot.pdf"
        fig.savefig(out_file_name)

        PCA_scores = obj_PCA.fit_transform(data_for_analysis_standardized)

        # scores plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(PCA_scores[0:20, 0], PCA_scores[0:20, 1], color='blue', label='NSCLC')
        ax.scatter(PCA_scores[20:40, 0], PCA_scores[20:40, 1], color='red', label='SCLC')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('scores plot')

        for i in range(PCA_scores.shape[0]):
            ax.text(PCA_scores[i, 0], PCA_scores[i, 1], sample_names[i])

        ax.legend()
        fig.show()
        out_file_name = "./results/PCA_scores_plot.pdf"
        fig.savefig(out_file_name)

        # loadings plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(PCA_fit_results.components_[0, :], PCA_fit_results.components_[1, :],
                   color='blue', s=marker_size)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('loadings plot')

        for i in range(num_of_variables):
            ax.text(PCA_fit_results.components_[0, i], PCA_fit_results.components_[1, i], variable_names[i])

        fig.show()
        out_file_name = "./results/PCA_loadings_plot.pdf"
        fig.savefig(out_file_name)

        #export loadings
        PCA_loadings = pd.DataFrame((PCA_fit_results.components_.T)[:, 0:2],
                                    index=variable_names,
                                    columns=['PC1', 'PC2'])
        out_file_name = "./results/PCA_loadings.xlsx"
        PCA_loadings.to_excel(out_file_name)

    else:
        print("unknown example index!\n")
        exit()

if __name__ == '__main__':
    main()
    k = input("Press any key to exit")
    plt.close('all')
