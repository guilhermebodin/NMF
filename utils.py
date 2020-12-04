def has_restaurant(text):
    return 'Restaurants' in text


def filter_df(df, first_datetime, last_datetime):
    idx = []
    for i in range(0, len(df['categories'])):
        if isinstance(df['categories'][i], str) and has_restaurant(df['categories'][i]):
            idx.append(i)
    df_restaurants = df.iloc[idx]
    df_filtered = df_restaurants[(df_restaurants['date'] > first_datetime) & (df_restaurants['date'] < last_datetime)]
    return df_filtered

