import pandas as pd

def load_data():
    transactions = pd.read_csv("data/raw/Cafe Transaction store.csv")
    metadata = pd.read_csv("data/raw/Cafe Sell MetaData.csv")
    dateinfo = pd.read_csv("data/raw/Cafe DateInfo.csv")

    transactions['CALENDAR_DATE'] = pd.to_datetime(transactions['CALENDAR_DATE'])
    dateinfo['CALENDAR_DATE'] = pd.to_datetime(dateinfo['CALENDAR_DATE'])

    df = transactions.merge(
        metadata,
        on=['SELL_ID', 'SELL_CATEGORY'],
        how='left'
    )

    df = df.merge(
        dateinfo,
        on='CALENDAR_DATE',
        how='left'
    )

    return df
