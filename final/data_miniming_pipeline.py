# import needed libraries
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.decomposition
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.tree
import sklearn.svm

# load the dataset
data = sklearn.datasets.fetch_openml(name='GermanCreditData')
df = data.data
target = data.target
print(f"df.shape={df.shape}")
print(f"target.shape={target.shape}")

# one hot encode categorical columns
df = pd.get_dummies(df)
print(f"df.shape={df.shape}")

# convert non-numeric columns to numeric
le = sklearn.preprocessing.LabelEncoder()
df = df[df.columns[:]].apply(le.fit_transform)
print(f"df.shape={df.shape}")

# apply the polynomial kernel
poly = sklearn.preprocessing.PolynomialFeatures(2)
df = poly.fit_transform(df)
print(f"df.shape={df.shape}")

# apply a random projection
rng = np.random.RandomState(42)
proj = sklearn.random_projection.GaussianRandomProjection(n_components=120, random_state=rng)
df = proj.fit_transform(df)
print(f"df.shape={df.shape}")

# create train/validation/test sets
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

x_train0, x_test, y_train0, y_test = sklearn.model_selection.train_test_split(
    df,
    target,
    test_size=test_ratio,
    random_state=0,
    )

x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train0,
    y_train0,
    test_size=validation_ratio/(train_ratio + validation_ratio),
    random_state=0,
    )

print(f"len(x_train0)={len(x_train0)}")
print(f"len(x_train)={len(x_train)}")
print(f"len(x_val)={len(x_val)}")
print(f"len(x_test)={len(x_test)}")

# scale the data
scaler = sklearn.preprocessing.StandardScaler().fit(x_train0)
x_train0 = scaler.transform(x_train0)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
print(f"x_train0.shape={x_train0.shape}")

# PCA the data
pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(x_train0)
x_train0 = pca.transform(x_train0)
x_train = pca.transform(x_train)
x_val = pca.transform(x_val)
x_test = pca.transform(x_test)
print(f"x_train0.shape={x_train0.shape}")

# train a model
model = sklearn.linear_model.LogisticRegression(
    C=1e1,
    penalty='l2',
    dual=False,
    fit_intercept=True,
    intercept_scaling=True,
    max_iter=100,
    tol=1e-3,
    random_state=42,
    verbose=1,
    )
model = sklearn.linear_model.SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=1e-6,
    l1_ratio=0.1,
    fit_intercept=True,
    max_iter=100,
    tol=1e-6,
    shuffle=True,
    verbose=1,
    random_state=42,
    )
model = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=[10, 10],
    activation='relu',
    alpha=1e-6,
    max_iter=1000,
    tol=1e-6,
    verbose=1,
    )
model = sklearn.tree.DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=2,
    min_samples_leaf=10,
    max_features=None,
    max_leaf_nodes=25,
    random_state=42,
    )
model = sklearn.ensemble.AdaBoostClassifier(
    estimator=model,
    n_estimators=50,
    )
model = sklearn.neighbors.KNeighborsClassifier(
    n_neighbors=3,
    weights='uniform',
    algorithm='ball_tree',
    )
model = sklearn.svm.SVC(
    C=1e-3,
    kernel='rbf',
    degree=3,
    gamma='scale',
    tol=1e-3,
    max_iter=200,
    cache_size=200,
    random_state=42,
    )
model.fit(x_train, y_train)

# report validation accuracy
validation_accuracy = model.score(x_val, y_val)
print(f"validation_accuracy={validation_accuracy}")

# report test accuracy
if False:
    # WARNING:
    # this code should be run only once;
    # after the hyperparameters have been decided based on the validation performance,
    # then the False can be changed to True to run this code
    model.fit(x_train0, y_train0)
    test_accuracy = model.score(x_test, y_test)
    print(f"test_accuracy={test_accuracy}")
