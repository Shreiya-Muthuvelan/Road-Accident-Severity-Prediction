import pandas as pd

def rename_categories(df):
    df['Two-hour intervals'] = df['Two-hour intervals'].str.replace('Midnight', '24:00', regex=False)

    metro_areas = ['Sydney metro. area', 'Newcastle met. area', 'Wollongong met. area']
    df['Urbanisation'] = df['Urbanisation'].apply(lambda x: 'Metropolitan area' if x in metro_areas else x)
    df['Urbanisation'] = df['Urbanisation'].replace({'Country urban': 'Urban', 'Country non-urban': 'Non urban'})

    df['Street lighting'] = df['Street lighting'].replace({'Unknown / not stated': 'Unknown', 'Nil': 'No lighting'})
    df['Signals operation'] = df['Signals operation'].replace({'Unknown / not stated': 'Unknown', 'Nil': 'No signals'})
    df['Other traffic control'] = df['Other traffic control'].replace({'Other traf. control': 'Other'})
    df['Speed limit'] = df['Speed limit'].str.replace(' km/h', '', regex=False)

    unknown = ['Unknown type', 'Unknown motor vehicle']
    df['Other TU type'] = df['Other TU type'].apply(lambda x: 'Unknown' if x in unknown else x)
    df['Key TU type'] = df['Key TU type'].apply(lambda x: 'Unknown' if x in unknown else x)

    df.rename(columns={'Road classification (admin)': 'Road classification'}, inplace=True)
    return df


def remove_unknown(df):
    df = df[df['School zone location'] != 'Unknown']
    df = df[df['Type of location'] != 'Unknown']
    df = df[df['Urbanisation'] != 'Country unknown']
    df=df[df['Speed limit']!='Unknown']
    df['Route no.'] = df['Route no.'].astype(str).str.strip().str.title()

    df = df[df['Route no.'] != 'Unknown']


    return df



