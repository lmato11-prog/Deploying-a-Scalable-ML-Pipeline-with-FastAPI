# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was developed by Luis Matos as part of the Udacity "Deploying a Scalable
ML Pipeline with FastAPI" project. It is a RandomForestClassifier from
scikit-learn (version 1.5.1) trained with default hyperparameters and a fixed
random_state of 42 for reproducibility. The model performs binary classification,
predicting whether an individual's annual income exceeds $50,000 based on census
demographic attributes. Categorical features are one-hot encoded with a
OneHotEncoder and the target label is binarized with a LabelBinarizer.

## Intended Use
The model is intended for educational and demonstration purposes to illustrate an
end-to-end ML pipeline served through a FastAPI application. It predicts an income
category (<=50K or >50K) from publicly available census features. It is not
intended for use in real-world decisions about individuals, such as lending,
hiring, or eligibility determinations.

## Training Data
The training data is the Census Income (Adult) dataset derived from the 1994 U.S.
Census database. The full dataset contains roughly 32,500 records with 14 input
features plus the income label. The data was split into a training set (80%) and a
test set (20%) using train_test_split with random_state=42. The eight categorical
features (workclass, education, marital-status, occupation, relationship, race,
sex, and native-country) were one-hot encoded.

## Evaluation Data
The evaluation data is the 20% hold-out test split from the same Census Income
dataset. The same encoder and label binarizer fitted on the training data were
applied to the test data to ensure consistent feature representation.

## Metrics
The model was evaluated using precision, recall, and F-beta (F1) score. On the
hold-out test set the model achieved:

- Precision: 0.7419
- Recall: 0.6384
- F1 (F-beta, beta=1): 0.6863

Performance was also computed on slices of the data for each unique value of every
categorical feature; these per-slice metrics are recorded in slice_output.txt.

## Ethical Considerations
The dataset reflects historical census data from 1994 and encodes societal biases
present at that time, including imbalances across sex, race, and native-country.
Because of this, model predictions may systematically differ in accuracy across
demographic groups. The model should not be used to make consequential decisions
about real people, and any deployment would require careful fairness auditing.

## Caveats and Recommendations
The model uses default hyperparameters and has not been tuned, so predictive
performance could likely be improved through cross-validation and hyperparameter
search. The underlying data is several decades old and may not reflect current
income distributions. Users should review the per-slice metrics in slice_output.txt
to understand where the model underperforms before relying on any individual
prediction.
