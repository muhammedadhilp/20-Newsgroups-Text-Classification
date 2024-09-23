# 20 Newsgroups Text Classification

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Exploration](#data-exploration)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)

## Introduction
This project performs text classification on the 20 Newsgroups dataset using a machine learning pipeline. The dataset is fetched from the `sklearn.datasets` library and contains text from 20 different newsgroups. The goal is to train a model that can classify text into one of the 20 categories.

## Project Structure
- `data`: Directory containing any raw and processed data (in this case, from the 20 Newsgroups dataset).
- `notebooks`: Jupyter notebooks for data exploration, visualization, and experimentation.
- `src`: Contains the custom Python classes and preprocessing scripts used in the pipeline.
- `README.md`: This file.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```
2. Install the required dependencies:
```bash

pip install -r requirements.txt
```
3.Download necessary NLTK data:
```bash
python -m nltk.downloader punkt
```
## Data Exploration
The dataset is imported using fetch_20newsgroups from sklearn.datasets. It includes training, testing, and validation data. The following keys are available:

- `data`: The text data from the newsgroups.
- `filenames`: The filenames corresponding to each document.
- `target_names`: The names of the 20 categories.
- `target`: The category label for each document.
- `DESCR`: A description of the dataset.
## Category Distribution
The distribution of the 20 categories is explored using the np.bincount() function, which counts the number of samples in each category. Visualized with a bar plot, we observe that the dataset is fairly balanced, though categories like "soc.religion.christian" and "talk.politics.guns" have fewer samples compared to others.

## Category Distribution Plot
```python

# Visualize sample counts for each category
sns.barplot(data=df, x='Sample Count', y='Category')
plt.title('Sample Count in Each Category')
plt.show()
```
## Preprocessing
A custom text preprocessing pipeline is built using a `TextPreprocessor` class:

- Convert text to lowercase.
- Remove punctuation and numbers.
- Tokenize the text and apply stemming using the PorterStemmer from nltk.
```python
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.preprocess_text(doc) for doc in X]
```
## Modeling
A machine learning pipeline is created using Pipeline from sklearn. The pipeline performs the following steps:

- `Text Preprocessing`: Cleans and prepares the text using the custom TextPreprocessor class.
- `TF-IDF Vectorization`: Converts the cleaned text into TF-IDF vectors using TfidfVectorizer.
- `Classification`: A Naive Bayes classifier (MultinomialNB) is trained on the TF-IDF vectors.
```python
pipeline = Pipeline([
    ('preprocess', TextPreprocessor()),
    ('vectorize', TfidfVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])
```
## Train-Test Split
The dataset is split into training and testing sets with an 80-20 ratio. The split is stratified to ensure equal representation of all classes in both sets.

```python

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```
## Evaluation
The trained model is evaluated using the following metrics:

- `Accuracy`: Measures the overall correctness of the model.
- `Precision`: The ratio of correctly predicted positive observations to total predicted positives.
- `Recall`: The ratio of correctly predicted positive observations to all observations in the actual class.
- `F1 Score`: A weighted average of Precision and Recall.
## Classification Report
```python

print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
```
## Sample output:

```plaintext
Copy code
                          precision    recall  f1-score   support
    alt.atheism            0.86      0.79      0.83       160
    comp.graphics          0.83      0.87      0.85       195
    ...
    talk.religion.misc     1.00      0.23      0.37       126
    accuracy                                  0.88      3770
    macro avg             0.89      0.86      0.86      3770
    weighted avg          0.89      0.88      0.87      3770
```
## Conclusion
The model performs well, achieving an accuracy of 87.6%. Most categories are classified with high precision and recall, but a few underrepresented categories show lower performance. This could be due to class imbalance, which can be further explored in future work.

## Dependencies
To run the project, ensure you have the following dependencies installed:

- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `matplotlib`: For data visualization
- `seaborn`: For enhanced visualizations
- `scikit-learn`: For machine learning models
- `nltk`: For natural language processing tasks
```css

This README file covers all aspects of the project, from installation to evaluation.
```
