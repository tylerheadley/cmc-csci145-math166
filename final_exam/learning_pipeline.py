########################################
# STEP 0: import libraries
########################################
import pandas as pd
import sklearn.datasets
import sklearn.decomposition
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.random_projection
import sklearn.tree
import sklearn.svm

########################################
# STEP 1: Load the dataset
########################################

# NOTE:
# This code is problem specific.
# All the other code in this file is generic to work with any dataset.
# The "German Credit" dataset is a famous dataset for predicting whether someone will default on a loan.
# It has been used in previous practicum projects for banks.
# It is a publicly accessible dataset that is similar in format/content to their private datasets.
data = sklearn.datasets.fetch_openml(name='GermanCreditData')
df = data.data
target = data.target
print(f"df.shape={df.shape}")           # d = dimensions ; N = number of datapoints
print(f"target.shape={target.shape}")

########################################
# STEP 2: Apply "non-learned" data transformations
########################################

# one hot encode categorical columns
df = pd.get_dummies(df)
print(f"df.shape={df.shape}")

# convert non-numeric columns to numeric
le = sklearn.preprocessing.LabelEncoder()
df = df[df.columns[:]].apply(le.fit_transform)
print(f"df.shape={df.shape}")

# apply the polynomial feature map
poly = sklearn.preprocessing.PolynomialFeatures(3)
df = poly.fit_transform(df)
print(f"df.shape={df.shape}")

# apply a random projection
proj = sklearn.random_projection.GaussianRandomProjection(
    n_components=120,
    random_state=42,
    )
df = proj.fit_transform(df)
print(f"df.shape={df.shape}")

########################################
# STEP 3: Create train/test sets
########################################

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# ensure that the ratios sum to 1.0
epsilon = 1e-10
assert(1 - epsilon <= train_ratio + validation_ratio + test_ratio <= 1 + epsilon)

# create train0/test set
x_train0, x_test, y_train0, y_test = sklearn.model_selection.train_test_split(
    df,
    target,
    test_size=test_ratio,
    random_state=0,
    )
print(f"len(x_train0)={len(x_train0)}")
print(f"len(x_test)={len(x_test)}")

# create train/validation set
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train0,
    y_train0,
    test_size=validation_ratio/(train_ratio + validation_ratio),
    random_state=0,
    )
print(f"len(x_train)={len(x_train)}")
print(f"len(x_val)={len(x_val)}")

########################################
# STEP 4: Apply "learned" data transformations
########################################

# Any transformation from STEP 2 could also appear here.

# standardize the data
standardize = sklearn.preprocessing.StandardScaler(
    with_mean=True,
    with_std=True,
    )
standardize.fit(x_train) # all of these feature transformations have; => "learned"
x_train0 = standardize.transform(x_train0)
x_train = standardize.transform(x_train)
x_test = standardize.transform(x_test)
x_val = standardize.transform(x_val)
print(f"x_train0.shape={x_train0.shape}")

# scale the data to a finite range
scaler = sklearn.preprocessing.MaxAbsScaler()
scaler.fit(x_train)
x_train0 = scaler.transform(x_train0)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
print(f"x_train0.shape={x_train0.shape}")

# PCA the data
pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(x_train0)
x_train0 = pca.transform(x_train0)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_val = pca.transform(x_val)
print(f"x_train0.shape={x_train0.shape}")

# normalize the data
normalize = sklearn.preprocessing.Normalizer(norm='l1')
normalize.fit(x_train)
x_train0 = normalize.transform(x_train0)
x_train = normalize.transform(x_train)
x_test = normalize.transform(x_test)
x_val = normalize.transform(x_val)
print(f"x_train0.shape={x_train0.shape}")

########################################
# STEP 5: Train a model
########################################

# NOTE:
# the models below are listed in the order we covered them in class;
# the parameters are listed in the order of the documentation;
# you are responsible for understanding how all specified parameters impact the runtime and/or statistical errors
model = sklearn.linear_model.Perceptron(
    max_iter=1000,
    )
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
model = sklearn.naive_bayes.GaussianNB()
model = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
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
model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='sqrt',
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
    max_iter=2000,
    cache_size=200,
    random_state=42,
    )
model.fit(x_train, y_train)

# most of our discussions in class about "error"
# accuracy is just 1 - error

# report validation accuracy
validation_accuracy = model.score(x_val, y_val)
print(f"validation_accuracy={validation_accuracy:0.4f}")
train_accuracy = model.score(x_train, y_train)
print(f"train_accuracy={train_accuracy:0.4f}")

########################################
# STEP 6: Evaluate on test set
########################################

# WARNING:
# this code should be run only once;
# after the hyperparameters have been decided based on the validation performance,
# then the False can be changed to True to run this code
if False:
    model.fit(x_train0, y_train0)
    test_accuracy = model.score(x_test, y_test)
    print(f"test_accuracy={test_accuracy}")
