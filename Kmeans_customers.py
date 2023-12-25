import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings

warnings.filterwarnings('ignore')


def preprocessed_data():
    df = pd.read_csv("./Data/Mall_Customers.csv")
    df = df.drop('CustomerID', axis=1)

    X = df.values[:, 2:]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


X = preprocessed_data()


def Elbow():
    inertia_data = []
    for k in range(1, 15):
        modelElbow = KMeans(n_clusters=k)
        modelElbow.fit(X)
        inertia_data.append(modelElbow.inertia_)
    plt.plot(range(1, 15), inertia_data, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.title("Elbow Method")
    plt.xlabel("Number Of Clusters")
    plt.ylabel("Inertia Of Model")
    plt.show()


# Elbow()


def ElbowVisualizer():
    modelVisualizer = KMeans(random_state=1)
    visualizer = KElbowVisualizer(modelVisualizer, k=(1, 15))
    visualizer.fit(X)
    visualizer.show()


# ElbowVisualizer()

# --> Conclusion: k = 5 is the best number of clusters

def train_model():
    model = KMeans(n_clusters=5, random_state=42)
    y_pred = model.fit_predict(X)
    return model, y_pred


model, y_pred = train_model()


def displayCluster():
    # train_model

    plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=100, c='red', label="Cluster 1")
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s=100, c='yellow', label="Cluster 2")
    plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s=100, c='green', label="Cluster 3")
    plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s=100, c='blue', label="Cluster 4")
    plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s=100, c='black', label="Cluster 5")
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, c="magenta")
    plt.title("Customer Segmentation")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.show()


displayCluster()


def predict(X_predict):
    result = model.predict([X_predict])
    print(result)
    if result[0] == 0:
        print("Customers with medium annual income and medium annual spend")
    elif result[0] == 1:
        print("Customers with high annual income but low annual spend")
    elif result[0] == 2:
        print("Customers with low annual income and low annual spend")
    elif result[0] == 3:
        print("Customers low annual income but high annual spend")
    elif result[0] == 4:
        print("Customers with high annual income and high annual spend")


predict([125, 79])

score = silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouetter Score: %.3f' % score)

cluster_counts = np.unique(y_pred, return_counts=True)
print(cluster_counts)
