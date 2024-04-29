# full script for csp and lda. 
# takes a participant/treatment method as a command line argument.

# imports

import numpy as np
import scipy.io

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

import sys

if len(sys.argv) < 2:
    print("Usage: python3 csp_lda.py {subject_treatment}\n\te.g. python3 csp_lda P1_post")
    sys.exit()

fname = sys.argv[1]
subject_file = f"data/filtered/{fname}_training_filtered.mat"
subject = scipy.io.loadmat(subject_file)

#LOAD TRAIN DATA

# initialise epoch arrays
epochs = []
start_idx = 0
end_idx = -1 
for idx, cue in enumerate(subject["trig"]):
    if idx+1 == subject["trig"].shape[0] or cue != subject["trig"][idx+1]: 
        start_idx = end_idx + 1
        end_idx = idx
        epochs.append(
            np.zeros(
                (end_idx + 1 - start_idx, 16)
            )
        )
        if idx +1 == subject["trig"].shape[0]:
            break

# load the filtered data into the epochs
sample_index = 0
for epoch in epochs:
    epoch_len = epoch.shape[0]
    for epoch_sample_index in range(epoch_len):
        epoch[epoch_sample_index,:] = subject["filteredData"][sample_index,:]
        sample_index += 1

# now build X and y
# X : ndarray, shape (n_epochs, n_channels, n_times)
#  y : ndarray, shape (n_epochs,)
            # The class for each epoch.
n_channels = 16
n_times = 2048

all_epoch_labels = []
for idx, cue in enumerate(subject["trig"]):
    if idx+1 == subject["trig"].shape[0] or cue != subject["trig"][idx+1]: 
        all_epoch_labels.append(cue) 
        if idx +1 == subject["trig"].shape[0]:
            break

X = []
y = []
for epoch_idx, epoch in enumerate(epochs):
    if len(epoch) == n_times:
        X.append(epoch)
        y.append(all_epoch_labels[epoch_idx])

X = np.asarray(X)

all_epoch_labels = 0
epochs = 0

y = np.asarray(y)
y = y.flatten()
X = X.reshape((80, 16, 2048))

# LOAD TEST DATA

test_subject_file = f"data/filtered/{fname}_test_filtered.mat"
test_subject = scipy.io.loadmat(test_subject_file)

# initialise test epoch arrays. 
test_epochs = []
start_idx = 0
end_idx = -1 
for idx, cue in enumerate(test_subject["trig"]):
    if idx+1 == test_subject["trig"].shape[0] or cue != test_subject["trig"][idx+1]: 
        start_idx = end_idx + 1
        end_idx = idx
        test_epochs.append(
            np.zeros(
                (end_idx + 1 - start_idx, 16)
            )
        )
        if idx +1 == test_subject["trig"].shape[0]:
            break

# load the filtered test data into the test epochs
sample_index = 0
for epoch in test_epochs:
    epoch_len = epoch.shape[0]
    for epoch_sample_index in range(epoch_len):
        epoch[epoch_sample_index,:] = test_subject["filteredData"][sample_index,:]
        sample_index += 1

#now build X_test and y_test
n_channels = 16
n_times = 2048

all_test_epoch_labels = []
for idx, cue in enumerate(test_subject["trig"]):
    if idx+1 == test_subject["trig"].shape[0] or cue != test_subject["trig"][idx+1]: 
        all_test_epoch_labels.append(cue) 
        if idx +1 == test_subject["trig"].shape[0]:
            break

X_test = []
y_test = []
for epoch_idx, epoch in enumerate(test_epochs):
    if len(epoch) == n_times:
        X_test.append(epoch)
        y_test.append(all_test_epoch_labels[epoch_idx])

X_test = np.asarray(X_test)

X_test = X_test.reshape((80, 16, 2048))
y_test = np.asarray(y_test)
y_test = y_test.flatten()

# TUNE PARAMETERS IN CSP AND LDA

# Define the parameter grid
param_grid = {
    'csp__reg': [0.1, 0.01, 0.001, 0.0001],
    'csp__n_components': [2,3,4,6,8,10]
}

# Instantiate the CSP and LDA classifiers
csp = CSP()
lda = LinearDiscriminantAnalysis()

# Create a pipeline with CSP and LDA
pipeline = Pipeline([
    ('csp', csp),
    ('lda', lda)
])

# Instantiate RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=3,  # Adjust the number of iterations as needed
    scoring='accuracy',  # Adjust the scoring metric as needed
    cv=5  # Adjust the number of cross-validation folds as needed
)

# Fit RandomizedSearchCV to your training data
random_search.fit(X, y)

# Get the best parameters
best_params = random_search.best_params_

# USE THE BEST PARAMS

# test with best params
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

scores = cross_val_score(random_search, X_test, y_test, cv=cv, n_jobs=None)
output = ""
with open("results/" + fname + "_result.txt", 'w+') as f:
    output += f"best parameters: {best_params}\n"
    output += f"mean score {scores.mean()} with std {scores.std()}\n"
    f.write(output)