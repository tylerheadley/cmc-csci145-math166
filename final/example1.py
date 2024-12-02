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
data = sklearn.datasets.fetch_openml(name='GermanCreditData')
df = data.data
target = data.target
print(f"df.shape={df.shape}")
print(f"target.shape={target.shape}")

########################################
# STEP 2: Apply "non-learned" data transformations
########################################

# convert non-numeric columns to numeric
le = sklearn.preprocessing.LabelEncoder()
df = df[df.columns[:]].apply(le.fit_transform)
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

# scale the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train0 = scaler.transform(x_train0)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
print(f"x_train0.shape={x_train0.shape}")

########################################
# STEP 5: Train a model
########################################

model = sklearn.linear_model.LogisticRegression(
    C=1e-5,
    penalty='l1',
    dual=False,
    fit_intercept=True,
    intercept_scaling=True,
    max_iter=100,
    tol=1e-3,
    random_state=42,
    verbose=1,
    )
model.fit(x_train, y_train)

# report validation accuracy
validation_accuracy = model.score(x_val, y_val)
print(f"validation_accuracy={validation_accuracy:0.4f}")
train_accuracy = model.score(x_train, y_train)
print(f"train_accuracy={train_accuracy:0.4f}")
