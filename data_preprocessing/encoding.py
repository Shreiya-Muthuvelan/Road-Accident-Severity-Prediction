from sklearn.preprocessing import LabelEncoder
def frequency_encode(df, col):
    freq = df[col].value_counts(normalize=False)
    df[col] = df[col].map(freq)
    return df

def group_rare_categories(df, column, coverage=0.95, other_label="OTHER"):
    counts = df[column].value_counts()
    counts_cumsum = counts.cumsum()
    total = counts.sum()
    keep_categories = counts_cumsum[counts_cumsum / total <= coverage].index
    df[column] = df[column].apply(lambda x: x if x in keep_categories else other_label)
    return df

def categorical_encoding(df,columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df