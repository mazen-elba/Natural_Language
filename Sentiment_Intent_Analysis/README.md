# Summary
### 1. Dataset Description
- An overview about the dataset and how to the train (labeled) and test (unlabeled) are divided, with the number of tweets present in each dataset
  - **Label (1)** denotes negative sentiment
  - **Label (0)** denotes positive sentiment
- Objective: predict the labels on a given test dataset
  - **id**: associated with the tweets in a given dataset
  - **tweets**: collected from various sources; having either positive (0) or negative (1) sentiments associated with it

### 2. Data Pre-Processing
- Removing Twitter Handles (@user)
- Removing Punctuation, Numbers, Special Characters
- Removing Short Words
- Tokenization
- Stemming

### 3. Data Visualization Techniques
- WordCloud
- Bar Plots

### 4. Feature Extraction Techniques Used
- Bag-of-Words
- TF-IDF

### 5. Machine Learning Models Used
- Logistic Regression
- XGBoost
- Decision Trees

### 6. Evaluation Metrics Used
- F1 Score

# References
- Deepak Das' "Social Media Sentiment Analysis using Machine Learning" (Part I & II)
