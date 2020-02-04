# ----- Loading Dependencies ------
import numpy as np
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from nltk.stem.snowball import SnowballStemmer

# Loading dataset - training data
train_twenty = fetch_20newsgroups(subset='train', shuffle=True)

# Check categories (target names)
train_twenty.target_names

# Print the first line of the first data file
print('\n'.join(train_twenty.data[0].split('\n')[:3]))

# Extracting features from text files
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_twenty.data)
X_train_counts.shape

# Feature extactoin using TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# ------------------ Machine Learning -------------
# Using NB Classifier on training data
clf = MultinomialNB().fit(X_train_tfidf, train_twenty.target)

# Build pipeline (to write less code same as above)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf = text_clf.fit(train_twenty.data, train_twenty.target)

# Performance of NB Classifier
test_twenty = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(test_twenty.data)

np.mean(predicted == test_twenty.target)

# Training SVM and calculating its performance
text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge',
                                                   penalty='l2',
                                                   alpha=1e-3,
                                                   max_iter=5,
                                                   random_state=42))])

text_clf_svm = text_clf_svm.fit(train_twenty.data, train_twenty.target)
predicted_svm = text_clf_svm.predict(test_twenty.data)

np.mean(predicted_svm == test_twenty.target)

#------------- Performance Tuning with Grid Search --------------
# Note - all parameter names start with the classifier name
parameters = {'vect__ngram_range':[(1, 1), (1, 2)],
              'tfidf__use_idf':(True, False),
              'clf__alpha':(1e-2, 1e-3)}

# Create instance of the grid search by passing classifier, parameters, and n_jobs=-1 (to use multple cores)
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train_twenty.data, train_twenty.target)

# Display best mean score and parameters
gs_clf.best_score_
gs_clf.best_params_

# ------------------- Using NLTK Package --------------
# Removing stop words
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

# Stemming 
nltk.download()
stemmer = SnowballStemmer('english', ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                             ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(train_twenty.data, train_twenty.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(test_twenty.data)

np.mean(predicted_mnb_stemmed == test_twenty.target)