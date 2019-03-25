import numpy as np
import math
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

# for using latex
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rc('font', family='serif')

from ML_toolbox import d_LDA
from ML_toolbox import d_data_model

# -----------------------------------------------------
# plotting parameters
# -----------------------------------------------------
fig_width = 8
fig_height = 6

line_width = 2
marker_size = 26

axis_label_font_size = 9
legend_font_size = 9

# my customization of plotting
plot_params = {'figure.figsize': (fig_width, fig_height)}
plt.rcParams.update(plot_params)

# -----------------------------------------------------
# selections
# -----------------------------------------------------
bool_use_sklearn = True

index_example = 5 # apply d_LDA to different data
# index_example = 1: toy data
# index_example = 2: iris data set: setosa and virginica
# index_example = 3: iris data set: versicolor and virginica
# index_example = 4: iris data set: setosa, versicolor, and virginica
# index_example = 5: cell line data

index_example_sklearn = 4   # apply sklearn LDA to different data
# index_example_sklearn = 1: setosa and virginica
# index_example_sklearn = 2: virginica and versicolor
# index_example_sklearn = 3: setosa, versicolor, and virginica
# index_example_sklearn = 4: cell line

if not bool_use_sklearn:
    if index_example == 1:
        # -----------------------------------------------------
        # 1. Do LDA on toy data
        # -----------------------------------------------------

        file_name = "./data/toy_data_for_LDA.csv"
        data_in = pd.read_csv(file_name)

        data_for_analysis = d_data_model.data_model(data_in)

        # do LDA
        obj_LDA = d_LDA.LDA(num_of_classes=2)

        my_LDA_result = obj_LDA.fit(X=data_for_analysis.data,
                                    Y=data_for_analysis.target)

        W = my_LDA_result['W']

        W_scaled = W * 12.0 / W[0]

        # slope of W
        theta = math.atan(W[1] / W[0])

        # plot
        unique_targets = np.unique(data_for_analysis.target)

        II_0 = (np.where(data_for_analysis.target==unique_targets[0]))[0]
        II_1 = (np.where(data_for_analysis.target==unique_targets[1]))[0]

        N_0 = len(II_0)
        N_1 = len(II_1)

        x0 = data_for_analysis.data[II_0, :]
        x1 = data_for_analysis.data[II_1, :]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Apply LDA to a toy dataset')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')

        ax.scatter(x0[:, 0], x0[:, 1], color='blue', s=marker_size)
        ax.scatter(x1[:, 0], x1[:, 1], color='red', s=marker_size)

        ax.plot([0, W_scaled[0]], [0, W_scaled[1]], color='green')
        ax.scatter(-my_LDA_result['projection_0'] * math.cos(theta),
                -my_LDA_result['projection_0'] * math.sin(theta),
                color='blue', marker='x', s=marker_size)
        ax.scatter(-my_LDA_result['projection_1'] * math.cos(theta),
                -my_LDA_result['projection_1'] * math.sin(theta),
                color='red', marker='x', s=marker_size)

        ax.set_aspect(1)

        fig.show()

        print("\nDone with applying LDA to the toy data!\n")

        exit()

    elif index_example == 2:
        # -----------------------------------------------------
        # 2. Do LDA on iris data set: setosa and virginica
        # -----------------------------------------------------
        iris = load_iris()

        # get indices for each flower type
        II_setosa = np.where(iris.target==0)
        II_versicolor = np.where(iris.target==1)
        II_virginica = np.where(iris.target==2)

        II_setosa = II_setosa[0]
        II_versicolor = II_versicolor[0]
        II_virginica = II_virginica[0]

        # visualize the setosa and virginica data
        fig = plt.figure()
        fig.suptitle('iris data: setosa and virginica')

        ax = fig.add_subplot(1, 3, 1)
        ax.scatter(iris.data[II_setosa, 0], iris.data[II_setosa, 1],
                   marker='o', s=marker_size, color='blue', label='setosa')
        ax.scatter(iris.data[II_virginica, 0], iris.data[II_virginica, 1],
                   marker='o', s=marker_size, color='red', label='virginica')
        ax.set_xlabel('sepal length')
        ax.set_ylabel('sepal width')
        ax.legend()

        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(iris.data[II_setosa, 2], iris.data[II_setosa, 3],
                   marker='o', s=marker_size, color='blue')
        ax.scatter(iris.data[II_virginica, 2], iris.data[II_virginica, 3],
                   marker='o', s=marker_size, color='red')
        ax.set_xlabel('petal length')
        ax.set_ylabel('petal width')

        ax = fig.add_subplot(1, 3, 3)
        ax.scatter(iris.data[II_setosa, 1], iris.data[II_setosa, 2],
                   marker='o', s=marker_size, color='blue')
        ax.scatter(iris.data[II_virginica, 1], iris.data[II_virginica, 2],
                   marker='o', s=marker_size, color='red')
        ax.set_xlabel('sepal width')
        ax.set_ylabel('petal length')

        fig.show()

        # apply two-class LDA
        obj_LDA = d_LDA.LDA(num_of_classes=2)

        x0 = iris.data[II_setosa, :]
        x1 = iris.data[II_virginica, :]

        y0 = iris.target[II_setosa]
        y1 = iris.target[II_virginica]

        my_X = np.vstack((x0, x1))
        my_Y = np.concatenate((y0, y1))

        my_LDA_result = obj_LDA.fit(X=my_X, Y=my_Y)

        # # projection x onto W
        # projection_1 = np.matmul(W, x1.transpose())
        # projection_2 = np.matmul(W, x2.transpose())

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)

        ax.set_title('Results from applying LDA to setosa and virginica data')
        ax.set_xlabel('projection onto W')
        ax.set_ylabel('')

        ax.scatter(my_LDA_result['projection_0'],
                   np.zeros(len(my_LDA_result['projection_0'])),
                   marker='o', s=marker_size, color='blue')
        ax.scatter(my_LDA_result['projection_1'],
                   np.zeros(len(my_LDA_result['projection_1'])),
                   marker='o', s=marker_size, color='red')

        ax.scatter(my_LDA_result['mu_0_in_projection_space'], 0.0,
                   marker='*', s=marker_size, color='green')
        ax.scatter(my_LDA_result['mu_1_in_projection_space'], 0.0,
                   marker='*', s=marker_size, color='green')

        ax.scatter(my_LDA_result['decision_boundary'], 0.0,
                   marker='*', s=marker_size, color='magenta')

        fig.show()

        print("\nDone with applying LDA to the setosa and virginica!\n")

        exit()

    elif index_example == 3:
        # -----------------------------------------------------
        # 3. Do LDA on iris data set: versicolor and virginica
        # -----------------------------------------------------
        iris = load_iris()

        # get indices for each flower type
        II_setosa = np.where(iris.target==0)
        II_versicolor = np.where(iris.target==1)
        II_virginica = np.where(iris.target==2)

        II_setosa = II_setosa[0]
        II_versicolor = II_versicolor[0]
        II_virginica = II_virginica[0]

        # plot
        fig = plt.figure()
        fig.suptitle('iris data: versicolor and virginica')

        ax = fig.add_subplot(1, 3, 1)
        ax.scatter(iris.data[II_versicolor, 0], iris.data[II_versicolor, 1],
                   marker='o', s=marker_size, color='blue', label='versicolor')
        ax.scatter(iris.data[II_virginica, 0], iris.data[II_virginica, 1],
                   marker='o', s=marker_size, color='red', label='virginica')
        ax.set_xlabel('sepal length')
        ax.set_ylabel('sepal width')
        ax.legend()

        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(iris.data[II_versicolor, 2], iris.data[II_versicolor, 3],
                   marker='o', s=marker_size, color='blue')
        ax.scatter(iris.data[II_virginica, 2], iris.data[II_virginica, 3],
                   marker='o', s=marker_size, color='red')
        ax.set_xlabel('petal length')
        ax.set_ylabel('petal width')

        ax = fig.add_subplot(1, 3, 3)
        ax.scatter(iris.data[II_versicolor, 1], iris.data[II_versicolor, 2],
                   marker='o', s=marker_size, color='blue')
        ax.scatter(iris.data[II_virginica, 1], iris.data[II_virginica, 2],
                   marker='o', s=marker_size, color='red')
        ax.set_xlabel('sepal width')
        ax.set_ylabel('petal length')

        fig.show()

        # apply LDA
        x0 = iris.data[II_versicolor, :]
        x1 = iris.data[II_virginica, :]
        my_X = np.vstack((x0, x1))

        y0 = iris.target[II_versicolor]
        y1 = iris.target[II_virginica]
        my_Y = np.concatenate((y0, y1))

        obj_LDA = d_LDA.LDA(num_of_classes=2)
        my_LDA_result = obj_LDA.fit(X=my_X, Y=my_Y)

        # plot LDA result
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying LDA to versicolor and virginica data')
        ax.set_xlabel('projection onto W')
        ax.set_ylabel('')

        ax.scatter(my_LDA_result['projection_0'],
                   np.zeros(len(my_LDA_result['projection_0'])),
                   marker='o', s=marker_size, color='blue', label='versicolor')
        ax.scatter(my_LDA_result['projection_1'],
                   np.zeros(len(my_LDA_result['projection_1'])),
                   marker='o', s=marker_size, color='red', label='virginica')
        ax.legend()

        ax.scatter(my_LDA_result['mu_0_in_projection_space'], 0.0,
                   marker='*', s=20, color='green')
        ax.scatter(my_LDA_result['mu_1_in_projection_space'], 0.0,
                   marker='*', s=20, color='green')
        ax.scatter(my_LDA_result['decision_boundary'], 0.0,
                   marker='*', s=marker_size, color='magenta')

        fig.show()

        print("\nDone with applying LDA to the virginica and versicolor data!\n")

        exit()

    elif index_example == 4:
        # -----------------------------------------------------
        # 4. Do LDA on iris data set: setosa, versicolor, and virginica
        # -----------------------------------------------------

        iris = load_iris()

        my_X = iris.data
        my_Y = iris.target

        obj_LDA = d_LDA.LDA(num_of_classes=3)
        my_LDA_result = obj_LDA.fit(X=my_X, Y=my_Y)

        # # project iris data onto W
        # projection = np.matmul(W.transpose(), iris.data.transpose())
        # projection = projection.transpose()

        # plot the projections
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying LDA to iris')
        ax.set_xlabel(r'$W_1$')
        ax.set_ylabel(r'$W_2$')
        ax.scatter(my_LDA_result['projection'][0:50, 0],
                   my_LDA_result['projection'][0:50, 1],
                   marker='o', s=marker_size, color='blue', label='setosa')
        ax.scatter(my_LDA_result['projection'][50:100, 0],
                   my_LDA_result['projection'][50:100, 1],
                   marker='o', s=marker_size, color='red', label='versicolor')
        ax.scatter(my_LDA_result['projection'][100:150, 0],
                   my_LDA_result['projection'][100:150, 1],
                   marker='o', s=marker_size, color='green', label='setosa')
        ax.legend()

        for i in range(len(my_LDA_result['mu_tilde_dictionary'].keys())):
            ax.scatter(my_LDA_result['mu_tilde_dictionary'][i][0],
                       my_LDA_result['mu_tilde_dictionary'][i][1],
                       marker='*', s=marker_size, color='magenta')

        fig.show()

        print("\nDone with applying LDA to the setosa, virginica, and versicolor data!\n")

        exit()

    elif index_example == 5:
        # -----------------------------------------------------
        # 5. Do LDA on cell line data
        # -----------------------------------------------------
        in_file_name = "./data/SCLC_study_output_filtered_2.csv"
        data_in = pd.read_csv(in_file_name, index_col=0)

        my_X = data_in.values
        my_Y = np.concatenate((np.zeros(20), np.ones(20)))

        obj_LDA = d_LDA.LDA(num_of_classes=2)
        my_LDA_result = obj_LDA.fit(X=my_X, Y=my_Y)

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying LDA to cell line data')
        ax.set_xlabel('projection')
        ax.set_ylabel('')

        ax.scatter(my_LDA_result['projection_0'], np.zeros(20),
                   marker='o', s=marker_size, color='blue', label='NSCLC')
        ax.scatter(my_LDA_result['projection_1'], np.zeros(20),
                   marker='o', s=marker_size, color='red', label='NSCLC')
        ax.scatter(my_LDA_result['mu_0_in_projection_space'], 0.0,
                   marker='*', s=marker_size, color='magenta')
        ax.scatter(my_LDA_result['mu_1_in_projection_space'], 0.0,
                   marker='*', s=15, color='magenta')
        ax.legend()

        fig.show()

        print("\nDone with applying LDA to the cell line data!\n")

        exit()

        # sklearn_LDA_cell_line = LDA(n_components=2)
        # sklearn_LDA_cell_line.fit(X, y)

    else:
        print("unknown index for an example!")

        exit()

else:
    # -----------------------------------------------------
    # 6. use sklearn LDA
    # -----------------------------------------------------

    if index_example_sklearn == 1:
        # apply sklearn LDA to the setosa and virginica data
        iris = load_iris()

        # get indices for each flower type
        II_setosa = np.where(iris.target == 0)
        II_versicolor = np.where(iris.target == 1)
        II_virginica = np.where(iris.target == 2)

        II_setosa = II_setosa[0]
        II_versicolor = II_versicolor[0]
        II_virginica = II_virginica[0]

        X = np.vstack((iris.data[II_setosa, :], iris.data[II_virginica, :]))

        y = np.concatenate((iris.target[II_setosa], iris.target[II_virginica]))

        sklearn_LDA = LDA(n_components=1)
        sklearn_LDA_fit = sklearn_LDA.fit(X, y)
        sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)

        # plot the projections
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying sklearn LDA to iris setosa and virginica')
        ax.set_xlabel(r'$W_1$')
        ax.scatter(sklearn_LDA_projection[0:50], np.zeros(50),
                   marker='o', s=marker_size, color='blue', label='setosa')
        ax.scatter(sklearn_LDA_projection[50:100], np.ones(50),
                   marker='o', s=marker_size, color='red', label='virginica')

        ax.legend()

        fig.show()

        print(sklearn_LDA_fit.coef_)

        print("\n Done with applying sklearn LDA to iris setosa and virginica!\n")

    elif index_example_sklearn == 2:

        # apply sklearn LDA to the virginica and versicolor data
        iris = load_iris()

        # get indices for each flower type
        II_setosa = np.where(iris.target == 0)
        II_versicolor = np.where(iris.target == 1)
        II_virginica = np.where(iris.target == 2)

        II_setosa = II_setosa[0]
        II_versicolor = II_versicolor[0]
        II_virginica = II_virginica[0]

        X = np.vstack((iris.data[II_versicolor, :], iris.data[II_virginica, :]))

        y = np.concatenate((iris.target[II_versicolor], iris.target[II_virginica]))

        sklearn_LDA = LDA(n_components=1)
        sklearn_LDA_fit = sklearn_LDA.fit(X, y)
        sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)

        # plot the projections
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying sklearn LDA to iris setosa and virginica')
        ax.set_xlabel(r'$W_1$')
        ax.scatter(sklearn_LDA_projection[0:50], np.zeros(50),
                   marker='o', s=marker_size, color='blue', label='versicolor')
        ax.scatter(sklearn_LDA_projection[50:100], np.ones(50),
                   marker='o', s=marker_size, color='red', label='virginica')

        ax.legend()

        fig.show()

        print(sklearn_LDA_fit.coef_)

        print("\n Done with applying sklearn LDA to iris versicolor and virginica!\n")

    elif index_example_sklearn == 3:

        # apply sklearn LDA to the entire iris data
        iris = load_iris()

        sklearn_LDA = LDA(n_components=2)
        sklearn_LDA_fit = sklearn_LDA.fit(iris.data, iris.target)
        sklearn_LDA_projection = sklearn_LDA.fit_transform(iris.data, iris.target)
        sklearn_LDA_projection = -sklearn_LDA_projection

        # plot the projections
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying sklearn LDA to iris')
        ax.set_xlabel(r'$W_1$')
        ax.set_ylabel(r'$W_2$')
        ax.scatter(sklearn_LDA_projection[0:50, 0], sklearn_LDA_projection[0:50, 1],
                   marker='o', s=marker_size, color='blue', label='setosa')
        ax.scatter(sklearn_LDA_projection[50:100, 0], sklearn_LDA_projection[50:100, 1],
                   marker='o', s=marker_size, color='red', label='versicolor')
        ax.scatter(sklearn_LDA_projection[100:150, 0], sklearn_LDA_projection[100:150, 1],
                   marker='o', s=marker_size, color='green', label='setosa')
        ax.legend()

        fig.show()

        print(sklearn_LDA_fit.coef_)

        exit()

    elif index_example_sklearn == 4:

        # apply sklearn LDA to cell line data
        in_file_name = "./data/SCLC_study_output_filtered_2.csv"
        data_in = pd.read_csv(in_file_name, index_col=0)

        my_X = data_in.values
        my_Y = np.concatenate((np.zeros(20), np.ones(20)))

        II_0 = np.where(my_Y==0)
        II_1 = np.where(my_Y==1)

        II_0 = II_0[0]
        II_1 = II_1[0]

        sklearn_LDA = LDA(n_components=2)
        sklearn_LDA_projection = sklearn_LDA.fit_transform(my_X, my_Y)
        sklearn_LDA_projection = -sklearn_LDA_projection

        # plot the projections
        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Results from applying sklearn LDA to cell line data')
        ax.set_xlabel(r'$W_1$')
        ax.set_ylabel('')
        ax.scatter(sklearn_LDA_projection[II_0], np.zeros(len(II_0)),
                   marker='o', s=marker_size, color='blue', label='NSCLC')
        ax.scatter(sklearn_LDA_projection[II_1], np.zeros(len(II_1)),
                   marker='o', s=marker_size, color='red', label='SCLC')
        ax.legend()

        fig.show()

        print("\nDone with applying sklearn LDA to the cell line data!\n")

        exit()

    else:
        print("\n Unknown data to use!\n")

        exit()

# -----------------------------------------------------
# 7. LDA, PCA, and clustering
# -----------------------------------------------------
# PCA on cell line data

in_file_name = "./data/SCLC_study_output_filtered_2.csv"
data_in = pd.read_csv(in_file_name, index_col=0)

sklearn_PCA = PCA(n_components=2)
PCA_scores = sklearn_PCA.fit(data_in).transform(data_in)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('PCA of cell line data')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.plot(PCA_scores[0:20, 0], PCA_scores[0:20, 1], linestyle='None', color='blue', marker='o')
ax.plot(PCA_scores[20:40, 0], PCA_scores[20:40, 1], linestyle='None', color='red', marker='o')
fig.show()

# kmeans clustering of cell line data
X = data_in.values
y = np.concatenate((np.zeros(20), np.ones(20)))

sklearn_KMeans = KMeans(n_clusters=2, random_state=0)
sklearn_KMeans.fit(X)

print("KMeans clustering results: labels")
print(sklearn_KMeans.labels_)
print("KMeans clustering results: cluster centers")
print(sklearn_KMeans.cluster_centers_)

# hierarchical clustering
sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')
sklearn_agglomerative_clustering.fit(X)
print("agglomerative clustering results: complete linkage, euclidean distance")
print(sklearn_agglomerative_clustering.labels_)

sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='euclidean')
sklearn_agglomerative_clustering.fit(X)
print("agglomerative clustering results: average linkage, euclidean distance")
print(sklearn_agglomerative_clustering.labels_)

sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='manhattan')
sklearn_agglomerative_clustering.fit(X)
print("agglomerative clustering results: complete linkage, manhattan distance")
print(sklearn_agglomerative_clustering.labels_)

# -----------------------------------------------------
# 8. use logistic regression for classification of the cell line data
# -----------------------------------------------------
num_of_samples = X.shape[0]
num_of_features = X.shape[1]

sklearn_logistic_classifier = LogisticRegression(random_state=0)

prediction_all = np.zeros(num_of_samples)
# leave-one-out cross validation
for i in range(num_of_samples):
    cur_X_test = X[i, :]
    cur_X_test = cur_X_test.reshape((1, num_of_features))

    cur_X_train = np.delete(X, obj=i, axis=0)

    cur_y_test = y[i]

    cur_y_train = np.delete(y, obj=i)

    sklearn_logistic_classifier.fit(cur_X_train, cur_y_train)

    cur_y_prediction = sklearn_logistic_classifier.predict(cur_X_test)

    prediction_all[i] = cur_y_prediction

print("leave-one-out CV: prediction")
print(prediction_all)
