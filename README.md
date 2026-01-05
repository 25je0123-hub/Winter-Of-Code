# Winter-Of-Code
As a part of this project, I implemented a basic machine learning library from scratch using only NumPy, Pandas and Matplotlib. The library includes implementation of Linear Regression, Polynomial Regression, Logistic Regression, K-Nearest Neighbors (KNN), K-Means Clustering, Decision Tree( Classifier and Regressor), and a simple neural network. a special attention was given to core cocepts like normal equation, distance metrics, impurity measures (Gini and Entropy), centroid initialization. This helped me gain a deeper understanding of how these models learn from data. Models were tested on provided datasets to verify correctness. This scratch implementation strengthened my conceptual understanding and helped me make informed desicions while using optimized library implementations later in the Kaggle competition.
# Linear regression
what I understood about the algorithm: Linear regression is a supervised machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.
# Polynomial Regression
what I understood about the algorithm: Polynomial regression is an extention of linear regression that uses polynomial feature expansion to model non-linear decision boundaries. While standard linear regression assumes a linear boundary, polynomial features allow the model to capture more complex patterns in the data.
# Logistic Regression
what I understood about the algorithm: Though named as regression it is a classification algorithm the model applies sigmoid function to a linear combination of polynomial features and is trained by minimizing the binary cross entropy loss using gradient based methods such as gradient descent and numerical stability techniques like clipping are commonly used to ensure stable training.
# Knn-regressor
what I understood about the algorithm: Knn regressor predicts the continues values by averaging the target values of the k nearest neighbors of given input point. similar to KNN classification it relies on distance metrics and does not learn explicit model parameters.
# Knn calssifier
K - nearest neighbors is a non parametric, instance based supervised learning algorithm used for classification. it predicts the class of a data point by finding the majority class among its k closed neighbors in the feature space, based on distance metrics.
# K means
what I understood about the algorithm: K-Means is an unsupervised learning algorithm used for clustering data into k distinct groups based on feature similarity. The algorithm iteratively assigns data points to the nearest centroid and updates the centroids to minimize the within-cluster sum of squared distances (inertia).
# mini batch K means
what I understood about algorithm:Mini-Batch K-Means is a faster, memory-efficient variant of K-Means that updates cluster centroids using small random batches of data instead of the full dataset. This allows it to scale well to large datasets while providing comparable clustering performance.
# Decision Tree classifier
what I understood about algorithm: Decision Tree Regression predicts continuous values by splitting the feature space to minimize variance in each subset. The model recursively partitions the data and outputs the mean value of samples in a leaf node.
# Decision Tree Regressor
what I understood about the algorithm: Decision Tree Regression predicts continuous values by splitting the feature space to minimize variance in each subset. The model recursively partitions the data and outputs the mean value of samples in a leaf node.
# Neural Networks
what I understood: Neural networks learn model parameters (weights and biases) via forward and backward propagation using optimizers like SGD, Adam, or RMSProp to minimize a loss function (MSE, cross-entropy, etc.). Additional techniques like batch normalization, dropout, and adaptive learning rates improve convergence and generalization.
