import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import utils
import matplotlib.pyplot as plt
import seaborn as sns

VALID_FIRST_DATETIME = '2019-01-01'
VALID_LAST_DATETIME = '2020-01-01'
NUM_REVIEWS = 5_000 # Should be less than 8k

df = pd.read_csv("yelp_reviews_PA.csv")
# Add 'review_length' field
df['review_length'] = df.text.map(len)
# Filter reviews for restaurants in 2019
filtered_df = utils.filter_df(df, VALID_FIRST_DATETIME, VALID_LAST_DATETIME)
ax = sns.FacetGrid(data=filtered_df, col='review_stars', xlim=(0, 2000)).map(plt.hist, 'review_length', bins=50)
ax.axes[0][0].set(ylabel='number of reviews');
ax.savefig('review_length')

# Save top reviews and bottom reviews
positive_reviews = filtered_df.text[df.review_stars >= 4]
negative_reviews = filtered_df.text[df.review_stars <= 2]

# Collect stop words
extra_words = ['ve', 'like', 'got', 'just', 
                'restaurant', 'great',
                'topping', 'toppings',
               'don', 'really', 'said', 'told', 'ok',
               'came', 'went', 'did', 'didn', 'good']
stop_words = text.ENGLISH_STOP_WORDS.union(extra_words)

# Generate random vectors for positive and negative reviews
# Create a vectorizer object to generate term document counts
tfidf_positive = TfidfVectorizer(stop_words=stop_words, min_df=10, max_df=0.5, 
                        ngram_range=(1,1), token_pattern='[a-z][a-z]+')

tfidf_negative = TfidfVectorizer(stop_words=stop_words, min_df=10, max_df=0.5, 
                        ngram_range=(1,1), token_pattern='[a-z][a-z]+')

# Get random subset of reviews
np.random.seed(38)
num_reviews = NUM_REVIEWS
random_negative = np.random.choice(negative_reviews, size=num_reviews)
random_positive = np.random.choice(positive_reviews, size=num_reviews)

# Replace some common words to avoid separating them in the method
dict_words = {'ordered':'order',
              'prices':'price', 
              'pizzas':'pizza', 
              'burgers': 'burguer'}
def replace_words(text, dict_words):
    for i,j in dict_words.items():
        text = text.replace(i,j)
    return text


random_negative = [replace_words(w, dict_words) for w in random_negative]
random_positive = [replace_words(w, dict_words) for w in random_positive]

# Get the vectors
negative_vectors = tfidf_negative.fit_transform(random_negative)
positive_vectors = tfidf_positive.fit_transform(random_positive)

# TODO I think we don`t need to add this
# Store TFIDF vectors in a Pandas DataFrame to investigate further
neg_df = pd.DataFrame(negative_vectors.todense(), columns=[tfidf_negative.get_feature_names()])
pos_df = pd.DataFrame(positive_vectors.todense(), columns=[tfidf_positive.get_feature_names()])

# get mean for each column (word): highest means are most important words
col_means_neg = {}
for col in neg_df:
    col_means_neg[col] = neg_df[col].mean()

col_means_pos = {}
for col in pos_df:
    col_means_pos[col] = pos_df[col].mean()

no_top_words = 8

print('Top %d words in POSITIVE reviews:' %no_top_words, end='')
print(sorted(col_means_pos, key=col_means_pos.get, reverse=True)[:no_top_words])

print('Top %d words in NEGATIVE reviews:' %no_top_words, end='')
print(sorted(col_means_neg, key=col_means_neg.get, reverse=True)[:no_top_words])

# NMF algorithm
# change num_topics
num_topics = 5

nmf_positive = NMF(n_components=num_topics, max_iter=1000)
W_positive = nmf_positive.fit_transform(positive_vectors)
H_positive = nmf_positive.components_

nmf_negative = NMF(n_components=num_topics)
W_negative = nmf_negative.fit_transform(negative_vectors)
H_negative = nmf_negative.components_

def display_topics(model, feature_names, num_topics, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx < num_topics:
            print("{:11}".format("Topic %d:" %(topic_idx)), end='')
            print(", ".join(['{:04.3f}*'.format(topic[i])+feature_names[i] \
                             for i in topic.argsort()[:-no_top_words-1:-1]]))

no_topics = num_topics
no_top_words = 10

print('Top topics for POSITIVE reviews')
print('-'*39)
display_topics(nmf_positive, tfidf_positive.get_feature_names(), no_topics, no_top_words)

print('\nTop topics for NEGATIVE reviews')
print('-'*39)
display_topics(nmf_negative, tfidf_negative.get_feature_names(), no_topics, no_top_words)