# Term Deposit Prediction - Bank Marketing (UCI)

Machine Learning - Assignment 2

Sanjoy Mukherjee (2026AJ05819)

## a. Problem statement

A Portuguese bank ran phone campaigns to sell a term deposit to its clients. Using what is
known about a client and the campaign before the call is made, the goal is to predict
whether the client will subscribe (`y` = yes) or not (`y` = no).

This is a binary classification problem with imbalanced classes: only about 11 % of the
clients said yes. The five classifiers required by the brief are fitted on the same training rows and
compared with six measures: Accuracy, AUC, Precision, Recall, F1 and MCC. A Streamlit app loads the saved models and evaluates
them on a test CSV.

## b. Dataset description

- **Source:** Bank Marketing, UCI Machine Learning Repository, dataset id 222
  (Moro et al., 2014). https://archive.ics.uci.edu/dataset/222/bank+marketing
- **File used:** `bank-additional.csv` from the UCI download. It is a random 10 % sample of
  the full data, with 4,119 clients, 20 input columns and the target `y`. UCI provides
  this smaller file for slower algorithms such as SVM.
- **Quality:** no empty cells and no duplicate rows. Six text columns use "unknown" for a
  missing answer, and it is kept as a category of its own.
- **Target:** 3,668 "no" and 451 "yes" (10.9 %).
- **Removed column:** `duration` (length of the last call). The UCI notes say it is only
  known after the call, when the result is already known, so it should not be used in a
  realistic model. This leaves **19 input features**.
- **Split:** 80 % train (3,295 rows) and 20 % test (824 rows), stratified on `y`,
  `random_state = 42`. The test rows are saved as `test_data.csv`.

| Feature | Type | Meaning | Preprocessing |
|---|---|---|---|
| age | numeric | age of the client | standard scaling |
| job, marital, education | categorical | client details | one-hot |
| default, housing, loan | categorical (yes / no / unknown) | credit in default, housing loan, personal loan | one-hot |
| contact | categorical | cellular or telephone | one-hot |
| month, day_of_week | categorical | when the last call was made | one-hot |
| campaign | numeric | calls made to the client in this campaign | Yeo-Johnson power transform |
| pdays | numeric | days since the last call of an earlier campaign (999 = never called) | turned into a 0/1 "never contacted" flag |
| previous | numeric | calls made before this campaign | Yeo-Johnson power transform |
| poutcome | categorical | result of the previous campaign | one-hot |
| emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed | numeric | economic indicators at the time of the call | standard scaling |
| **y** | **target** | **did the client subscribe? (yes / no)** | |

The preprocessing is fitted on the training part only. It is stored inside every saved
model pipeline, and also saved on its own as `model/preprocessing.pkl`. A category that was
not seen in training is ignored by the one-hot encoder.

## c. GitHub repository link

https://github.com/2026aj05819/bank-marketing-classifier

```
bank-marketing-classifier/
|-- app.py                        Streamlit app
|-- bank_marketing_models.ipynb   training and evaluation notebook
|-- requirements.txt
|-- README.md
|-- test_data.csv                 test split: 19 features + y
|-- .streamlit/config.toml        app colour theme
|-- data/
|   `-- bank-additional.csv       data file from UCI
`-- model/
    |-- lr_pipeline.pkl           Logistic Regression
    |-- dt_pipeline.pkl           Decision Tree
    |-- knn_pipeline.pkl          KNN
    |-- nb_pipeline.pkl           Naive Bayes
    |-- svm_pipeline.pkl          Support Vector Machine
    |-- preprocessing.pkl         fitted preprocessing on its own
    |-- columns.json              feature list, target, model file names
    `-- comparison.csv            test-set metrics (table below)
```

## d. Models used

Each model is a scikit-learn pipeline (preprocessing followed by the classifier). Its
settings were picked with a grid search and 5-fold stratified cross-validation that only
sees the training rows, with MCC as the score:

| Model | Chosen settings |
|---|---|
| Logistic Regression | C = 0.1, class_weight = balanced |
| Decision Tree | criterion = entropy, max_depth = 5, min_samples_leaf = 10, class_weight = balanced |
| KNN | k = 7, distance-weighted, Manhattan distance (p = 1) |
| Naive Bayes | Gaussian, var_smoothing = 1.0 |
| Support Vector Machine | RBF kernel, C = 10, gamma = 0.01, class_weight = balanced |

The predicted class comes from each model's `predict()`. AUC uses the probability of "yes"
from `predict_proba()`. The SVM has no probability output, so its AUC uses
`decision_function()` instead.

### Comparison table (test set, 824 clients)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8337 | 0.7776 | 0.3464 | 0.5889 | 0.4362 | 0.3631 |
| Decision Tree | 0.8556 | 0.7520 | 0.3821 | 0.5222 | 0.4413 | 0.3665 |
| KNN | 0.8774 | 0.6898 | 0.3333 | 0.1222 | 0.1789 | 0.1468 |
| Naive Bayes | 0.8908 | 0.7598 | 0.5000 | 0.2667 | 0.3478 | 0.3116 |
| Support Vector Machine | 0.8580 | 0.7414 | 0.3969 | 0.5778 | 0.4706 | 0.4011 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.778), so it is the best at ranking clients from most to least likely to subscribe. With balanced class weights it finds 53 of the 90 subscribers in the test set (recall 0.589), but precision is only 0.346: about two out of three clients it flags do not subscribe. Cross-validation chose fairly strong regularisation (C = 0.1). Its MCC (0.363) is third of the five. |
| Decision Tree | Best cross-validation MCC (about 0.40), but on the test set it comes second on F1 (0.441) and MCC (0.367), behind the SVM. It finds 47 of the 90 subscribers with precision 0.382. The tree (depth 5, at least 10 clients per leaf) relies mostly on `nr.employed`, which has about 68 % of the importance, so it mainly splits on the economic situation at the time of the call. |
| KNN | Weakest model: lowest AUC (0.690), F1 (0.179) and MCC (0.147). It finds only 11 of the 90 subscribers. Its accuracy of 0.877 is even below the 0.891 that a model gets by always saying "no". KNN has no class weights, so most neighbours of any client are "no" clients, and 53 of the 62 columns after preprocessing are one-hot 0/1 columns, which makes the distances less meaningful. |
| Naive Bayes | Highest accuracy (0.891) and precision (0.500), but that accuracy is exactly the "always no" level, and it finds only 24 of the 90 subscribers (recall 0.267). Its independence assumption does not hold here: `emp.var.rate`, `euribor3m` and `nr.employed` are correlated at 0.90 to 0.97, so the same information is counted more than once, and the one-hot columns are not Gaussian. Cross-validation picked the largest smoothing value in the grid (1.0), which suggests the normal curves fit these columns badly. |
| Support Vector Machine | Best F1 (0.471) and best MCC (0.401). It finds 52 of the 90 subscribers, almost as many as Logistic Regression, with 79 false alarms instead of 100. Its AUC (0.741) is lower than that of Logistic Regression, so its ranking is a little worse, but at its own decision boundary it gives the best balance. It is also the heaviest model to run: it keeps about 1,900 of the 3,295 training clients (more than half) as support vectors. |
| **Overall, winner for your dataset?** | **Support Vector Machine**. It scores highest on both F1 and MCC, and these are the fairer measures here because only about 11 % of the clients are positive. Logistic Regression is a close second and has the best AUC, so it would be the better choice if the bank wants a ranked list of clients to call first. All five models are only moderate (MCC at most 0.40). This is expected once `duration` is removed, because the model has to decide before the call is made. |

## Streamlit app

Link: https://bank-marketing-classifier-2026aj05819.streamlit.app/

The app has five sections:

1. **Test data:** upload a CSV (comma or semicolon separated), or use `test_data.csv`
   from this repository. There is also a button to download `test_data.csv`.
2. **Model:** a dropdown with the five saved classifiers.
3. **Metrics:** the six scores (Accuracy, AUC, Precision, Recall, F1, MCC) of the selected model.
4. **Confusion matrix** of the selected model, next to its **classification report**.
5. **All five models:** a table with the six scores of every model on the loaded data, and a
   bar chart of F1 and MCC.

The app only loads the models from `model/` and predicts. It never trains on the uploaded
file.

## How to run locally

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

To train the models again (this rewrites `model/` and `test_data.csv`):

```bash
pip install notebook
jupyter nbconvert --to notebook --execute --inplace bank_marketing_models.ipynb
```

Tested with Python 3.12 and scikit-learn 1.7.2. The saved models need the same
scikit-learn version, which is why `requirements.txt` pins it.
