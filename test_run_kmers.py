# %%
from Bio import SeqIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# %%
def generate_kmers(k):
    """Generate all possible k-mers for a given k."""
    import itertools
    return [''.join(p) for p in itertools.product('ACGT', repeat=k)]

def get_kmer_freq(sequence, kmers):
    """Calculate k-mer frequencies for a sequence."""
    kmer_freq = {kmer: 0 for kmer in kmers}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_freq:
            kmer_freq[kmer] += 1
    total = sum(kmer_freq.values())
    return [freq / total for freq in kmer_freq.values()]

def extract_features_from_fasta(file, kmers, label):
    """Extract k-mer features from fasta file."""
    features = []
    labels = []
    for record in SeqIO.parse(file, "fasta"):
        sequence = str(record.seq).upper()
        features.append(get_kmer_freq(sequence, kmers))
        labels.append(label)
    return features, labels

# Define k-mer size
k = 6  # example, can be adjusted
kmers = generate_kmers(k)

# Extract features and labels
accessible_features, accessible_labels = extract_features_from_fasta("accessible.fasta", kmers, 1)
not_accessible_features, not_accessible_labels = extract_features_from_fasta("notaccessible.fasta", kmers, 0)

# Combine data
X = accessible_features + not_accessible_features
y = accessible_labels + not_accessible_labels

# Convert to numpy arrays for ML models
X = np.array(X)
y = np.array(y)


# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)


# %%
# Evaluate models
for model in [lr, rf, svm]:
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")


# %% [markdown]
# Test it on the unknown chromtain sites

# %%
# Extract features from test.fasta
test_features, _ = extract_features_from_fasta("test.fasta", kmers, None)  # Label is None as we don't have it
test_features = np.array(test_features)


# %%
# Predict with each model
def predict_and_save(model, features, filename):
    predictions = model.predict(features)
    pd.DataFrame(predictions, columns=['Predicted_Accessibility']).to_csv(filename, index=False)

predict_and_save(lr, test_features, "lr_predictions.csv")
predict_and_save(rf, test_features, "rf_predictions.csv")
predict_and_save(svm, test_features, "svm_predictions.csv")



