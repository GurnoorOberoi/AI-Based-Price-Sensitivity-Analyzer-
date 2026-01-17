import numpy as np

def preprocess(df):
    df['REVENUE'] = df['PRICE'] * df['QUANTITY']
    df['LOG_PRICE'] = np.log(df['PRICE'])
    df['LOG_QUANTITY'] = np.log(df['QUANTITY'])

    df.fillna(0, inplace=True)
    return df
