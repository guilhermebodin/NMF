import pandas as pd
import numpy as np
import sklearn as sk
import datetime
import utils

VALID_FIRST_DATETIME = '2019-01-01'
VALID_LAST_DATETIME = '2020-01-01'

df = pd.read_csv("yelp_reviews_PA.csv")
# Filter indices that have restaurant in Pitsburgh
idx = []
for i in range(0, len(df['categories'])):
    if isinstance(df['categories'][i], str) and utils.has_restaurant(df['categories'][i]):
        idx.append(i)

df_restaurants_capital = df.iloc[idx]
df_filtered = df_restaurants_capital[(df_restaurants_capital['date'] > VALID_FIRST_DATETIME) & (df_restaurants_capital['date'] < VALID_LAST_DATETIME)]

df_good_reviews = df_filtered[df_filtered['review_stars'] == 5]
df_bad_reviews = df_filtered[df_filtered['review_stars'] == 1]
df_good_reviews['processed_text'] = df_good_reviews['text'].apply(utils.process_text)
df_bad_reviews['processed_text'] = df_bad_reviews['text'].apply(utils.process_text)

df_bad_reviews['processed_text']
df_good_reviews['processed_text']


