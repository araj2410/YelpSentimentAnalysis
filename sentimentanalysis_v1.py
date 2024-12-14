import nltk
import ssl
import matplotlib.pyplot as plt
import pandas as pd
import time

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

csv_file_path = '/Users/rajeev/PycharmProjects/pythonProject/yelp.csv'
# Create empty lists to store sentiment scores
compound_scores = []
positive_scores = []
negative_scores = []
neutral_scores = []
sia = SentimentIntensityAnalyzer()
# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path)
# Iterate through each row and column and print the values
for index, row in df.iterrows():
    text = row['text']
    sentiment_scores = sia.polarity_scores(text)
    compound_scores.append(sentiment_scores['compound'])
    positive_scores.append(sentiment_scores['pos'])
    negative_scores.append(sentiment_scores['neg'])
    neutral_scores.append(sentiment_scores['neu'])

# Append the sentiment scores to the DataFrame
df['compound_score'] = compound_scores
df['positive_score'] = positive_scores
df['negative_score'] = negative_scores
df['neutral_score'] = neutral_scores
print(compound_scores)
df.to_csv(csv_file_path, index=False)


