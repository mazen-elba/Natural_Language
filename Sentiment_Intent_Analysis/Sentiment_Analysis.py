# Importing necessary packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import string
import nltk
from nltk import PorterStemmer

# Reading train.csv Pandas file
train = pd.read_csv("http://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv")
train_original = train.copy()

# Reading test.csv Pandas file
test = pd.read_csv("http://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/test.csv")
test_original = test.copy()

#### Data Pre-Processing ###################
# Step 1 - Combine train.csv and test.csv files
combine = train.append(test, ignore_index=True, sort=True)

# Step 2 - Removing Twitter Handles (@user)
def remove_pattern(text, pattern):
    """
    Function to remove unwanted text patterns from tweets.
    Takes two arguments: original string of text, pattern of text to remove.
    Returns same input string without given pattern.
    """
    
    # re.findall() fines the pattern (@user), and puts it in a list
    r = re.findall(pattern, text)
    
    # re.sub() removes the pattern (@user) from sentences in a dataset
    for i in r:
        text = re.sub(i, "", text)
        
    return text

# Using np.vectorize() instead of standard for-loops (optional)
combine["Tidy_Tweets"] = np.vectorize(remove_pattern)(combine["tweet"], "@[\w]*")

# Step 3 - Removing Punctuation, Numbers, and Special Characters
combine["Tidy_Tweets"] = combine["Tidy_Tweets"].str.replace("[^a-zA-Z#]", " ")

# Step 4 - Removing Short Words (words with length <= 3, ie; stop words)
combine["Tidy_Tweets"] = combine["Tidy_Tweets"].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))

# Step 5 - Tokenization (Pre-Process for Stemming)
#... Tokens: individual terms or words
#... Tokenization: process of splitting a string of text into tokens
tokenized_tweet = combine["Tidy_Tweets"].apply(lambda x: x.split())

# Step 6 - Stemming
#... Stemming: rule-based process of stripping suffixes from a word
#... Suffixes: "ing", "ly", "es", "s", etc.
ps = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

# Stitching tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    
combine["Tidy_Tweets"] = tokenized_tweet

#### Data Visualization ###################
# Importing packaged necessary for generating a WordCloud
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import urllib
import requests

# Function to extract hashtags from tweets
def Hashtags_Extract(x):
    hashtags = []
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
        
    return hashtags

#----- Positive Sentiment - store all words which are positive with the label "0"
all_words_positive = " ".join(text for text in combine["Tidy_Tweets"][combine["label"]==0])
Mask = np.array(Image.open(requests.get("http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png", stream=True).raw)) 	# Combine the image with the dataset
image_colors = ImageColorGenerator(Mask) 	# Take tge cikir if the image and impose it over wordcloud
wc = WordCloud(background_color="black", height=1500, width=4000, mask=Mask).generate(all_words_positive) 	# Use the WordCloud function from the wordcloud library
plt.figure(figsize=(10, 20)) 	# Generate image size
plt.imshow(wc.recolor(color_func=image_colors), interpolation="hamming") 	# Re-color the words to the image's color (re-colors the default colors)
plt.axis("off")
# plt.show()

#----- Negative Sentiment - store all words which are negative with the label "1"
all_words_negative = " ".join(text for text in combine["Tidy_Tweets"][combine["label"]==1])
Mask = np.array(Image.open(requests.get("http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png", stream=True).raw)) 	# Combine the image with the dataset
image_colors = ImageColorGenerator(Mask) 	# Take tge cikir if the image and impose it over wordcloud
wc = WordCloud(background_color="black", height=1500, width=4000, mask=Mask).generate(all_words_negative) 	# Use the WordCloud function from the wordcloud library
plt.figure(figsize=(10, 20)) 	# Generate image size
plt.imshow(wc.recolor(color_func=image_colors), interpolation="gaussian") 	# Re-color the words to the image's color (re-colors the default colors)
plt.axis("off")
# plt.show()

#------ Positive Sentiment
ht_positive = Hashtags_Extract(combine["Tidy_Tweets"][combine["label"]==0]) 	# Nested list of all the hashtags from positive reviews
ht_positive_unnest = sum(ht_positive, []) 	# Un-nest the list
word_freq_positive = nltk.FreqDist(ht_positive_unnest) 	# Count the frequency of words having a positive sentiment

# Create a dataframe for the most frequently used words in hashtags
df_positive = pd.DataFrame({"Hashtags":list(word_freq_positive.keys()),
                            "Count":list(word_freq_positive.values())})

# Plotting a Bar-Plot for the 20 most frequent words used for hashtags
df_positive_plot = df_positive.nlargest(20, columns="Count")
sns.barplot(data=df_positive_plot, y="Hashtags", x="Count")
# sns.despine()

#----- Negative Sentiment 
ht_negative = Hashtags_Extract(combine["Tidy_Tweets"][combine["label"]==1]) 	# Nested list of all hashtags from negative reviews
ht_negative_unnest = sum(ht_negative, []) 	# Un-nest the list
word_freq_negative = nltk.FreqDist(ht_negative_unnest) 	# Count the frequency of words having negative sentiment

# Create a dataframe for the most frequently used words in hashtags
df_negative = pd.DataFrame({"Hashtags":list(word_freq_negative.keys()),
                            "Count":list(word_freq_negative.values())})

df_negative_plot = df_negative.nlargest(20, columns="Count") 	# Plotting a Bar-Plot for the 20 most frequent words used for hashtags
sns.barplot(data=df_negative_plot, y="Hashtags", x="Count")
# sns.despine()

####### Feature Extraction via Word Embedding ################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, 
                                   max_features=1000, stop_words="english")

# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combine["Tidy_Tweets"])
df_bow = pd.DataFrame(bow.todense())
tfidf = TfidfVectorizer(max_df=0.90, min_df=2,
                        max_features=1000, stop_words="english")

# TF-IDF feature matrix
tfidf_matrix = tfidf.fit_transform(combine["Tidy_Tweets"])
df_tfidf = pd.DataFrame(tfidf_matrix.todense())

##### Splittint dataset into Training and Validation sets
train_bow = bow[:31962] 	# Using features from Bag-of-Words for training set
train_tfidf_matrix = tfidf_matrix[:31962] 	# Using features from TF-IDF for training set

# Using Bag-of-Words features
x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,
                                                                      train["label"],
                                                                      test_size=0.3,
                                                                      random_state=2)

# Using TF-IDF features
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,
                                                                              train["label"],
                                                                              test_size=0.3,
                                                                              random_state=17)

####### Applying Machine Learning Models ###################
# Importing Machine Learning models and F1 Score validation technique
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

#------ Applying Logistic Regression model on Feature Extractions
Log_Reg = LogisticRegression(random_state=0, solver="lbfgs")

# Applying the model on dataset using Bag-of-Words features
Log_Reg.fit(x_train_bow, y_train_bow) 	#... fitting the model
prediction_bow = Log_Reg.predict_proba(x_valid_bow) 	#... predicting the probabilities
prediction_int = prediction_bow[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
prediction_int = prediction_int.astype(np.int) 	# Converting the results to integer type
log_bow = f1_score(y_valid_bow, prediction_int) 	# Calculating F1 score

# Applying Logistic Regression model on dataset using TF-IDF features
Log_Reg.fit(x_train_tfidf, y_train_tfidf) 	#... fitting the model
prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf) 	#... predicting the probabilities
prediction_int = prediction_tfidf[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
prediction_int = prediction_int.astype(np.int) 	# Converting the results to integer type
log_tfidf = f1_score(y_valid_tfidf, prediction_int) 	# Calculating F1 score

#------ Applying XGBoost model on Bag-of-Words
model_bow = XGBClassifier(random_state=22, learning_rate=0.9)

# Applying the model on dataset
model_bow.fit(x_train_bow, y_train_bow) 	#... fitting the model
xgb = model_bow.predict_proba(x_valid_bow) 	#... predicting the probabilities
xgb = xgb[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
xgb_int = xgb.astype(np.int) 	# Converting the results to integer type
xgb_bow = f1_score(y_valid_bow, xgb_int) 	# Calculating F1 score

#------ Applying Decision Trees model on TF-IDF
model_tfidf = XGBClassifier(random_state=29, learning_rate=0.7)

# Applying XGBoost model on dataset using TF-IDF features
model_tfidf.fit(x_train_tfidf, y_train_tfidf) 	#... fitting the model
xgb_tfidf = model_tfidf.predict_proba(x_valid_tfidf) 	#... predicting the probabilities
xgb_tfidf = xgb_tfidf[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
xgb_int_tfidf = xgb_tfidf.astype(np.int) 	# Converting the results to integer type
score = f1_score(y_valid_tfidf, xgb_int_tfidf) 	# Calculating F1 score

#------ Applying Decision Trees model on Feature Extractions
dct = DecisionTreeClassifier(criterion="entropy", random_state=1)

# Applying Decision Trees model on dataset using Bag-of-Words features
dct.fit(x_train_bow, y_train_bow) 	#... fitting the model
dct_bow = dct.predict_proba(x_valid_bow) 	#... predicting the probabilities
dct_bow = dct_bow[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
dct_int_bow = dct_bow.astype(np.int) 	# Converting the results to integer type
dct_score_bow = f1_score(y_valid_bow, dct_int_bow) 	# Calculating F1 score

# Applying Decision Trees model on dataset using TF-IDF features
dct.fit(x_train_tfidf, y_train_tfidf) 	#... fitting the model
dct_tfidf = dct.predict_proba(x_valid_tfidf) 	#... predicting the probabilities
dct_tfidf = dct_tfidf[:,1] >= 0.3 	# If prediction is >= 0.3 then sentiment is 1 (negative), else 0 (positive)
dct_int_tfidf = dct_tfidf.astype(np.int) 	# Converting the results to integer type
dct_score_tfidf = f1_score(y_valid_tfidf, dct_int_tfidf) 	# Calculating F1 score

######## MODEL COMPARISON ###########################
##### Bag-of-Words
Algo_1 = ["LogisticRegression(Bag-of-Words)", "XGBoost(Bag-of-Words)", "DecisionTree(Bag-of-Words)"]
score_1 = [log_bow, xgb_bow, dct_score_bow]
compare_1 = pd.DataFrame({"Model":Algo_1, "F1_Score":score_1},
                          index=[i for i in range(1,4)])

# Comparison Graph
plt.figure(figsize=(18,5))
sns.pointplot(x="Model", y="F1_Score", data=compare_1)
plt.title("Bag-of-Words")
plt.xlabel("MODEL")
plt.ylabel("SCORE")
# plt.show()

###### TF-IDF
Algo_2 = ["LogisticRegression(TF-IDF)", "XGBoost(TF-IDF)", "DecisionTree(TF-IDF)"]
score_2 = [log_tfidf, score, dct_score_tfidf]
compare_2 = pd.DataFrame({"Model":Algo_2, "F1_Score":score_2},
                          index=[i for i in range(1,4)])

# Comparison Graph
plt.figure(figsize=(18,5))
sns.pointplot(x="Model", y="F1_Score", data=compare_2)
plt.title("TF-IDF")
plt.xlabel("MODEL")
plt.ylabel("SCORE")
# plt.show()

#### Comparison of Logistic Regression (best model)
Algo_best = ["LogisticRegression(Bag-of-Words)", "LogisticRegression(TF-IDF)"]
score_best = [log_bow, log_tfidf]
compare_best = pd.DataFrame({"Model":Algo_best, "F1_Score":score_best},
                          index=[i for i in range(1,3)])

# Comparison Graph for Logistic Regression model
plt.figure(figsize=(18,5))
sns.pointplot(x="Model", y="F1_Score", data=compare_best)
plt.title("Logistic Regression (Bag-of-Words & TF-IDF)")
plt.xlabel("MODEL")
plt.ylabel("SCORE")
# plt.show()

###### Predicting Results of Test data #######
test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test["label"] = test_pred_int

submission = test[["id", "label"]]
submission.to_csv("result.csv", index=False)

# Results after prediction
res = pd.read_csv("result.csv")