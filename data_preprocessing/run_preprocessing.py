import pandas as pd
from pathlib import Path

from .cleaning import rename_categories, remove_unknown
from .encoding import frequency_encode, group_rare_categories, categorical_encoding
from .inspection import inspect_null_values,inspect_class_imbalance,outlier_detection
from .download_dataset import download_dataset

def run_preprocessing():
    file_path = "data/raw/dataset.csv" 
    file_to_check = Path(file_path) 
    if file_to_check.exists(): 
        print("Found existing dataset.csv") 
    else: 
        print("Downloading dataset.csv...") 
        download_dataset()

    df = pd.read_csv(file_path)
    print("Dataset loaded")

    # Rename and clean categories
    df = rename_categories(df)
    df = remove_unknown(df)

    # Drop irrelevant columns
    # Our target variable is Degree of crash - detailed, so lets drop Degree of crash to avoid confusion
    drop_cols = [
        'Distance','Identifying feature','Direction','Identifying feature type',
        'RUM - description','DCA - description','Degree of crash'
    ]
    df = df.drop(columns=drop_cols)

    # Encoding
    for col in ['Street of crash','Town','Route no.','RUM - code','DCA - code']:
        df = frequency_encode(df, col)
    df = group_rare_categories(df, 'LGA')

    # Handle nulls & duplicates
    inspect_null_values(df)
    drop_missing = [
        'Primary permanent feature','Primary temporary feature',
        'Primary hazardous feature','DCA supplement'
    ]
    df = df.drop(columns=drop_missing)

    for col in ['Route no.', 'Other TU type']:
        df[col] = df[col].fillna('Unknown')

    df = df.drop_duplicates()
    print(f" Duplicates removed. Final shape: {df.shape}")

    # Categorical encoding
    columns_cat_encod=[ 'Day of week of crash','Month of crash','Road surface','Surface condition','Two-hour intervals','Degree of crash - detailed','School zone location', 'Street type',
                       'School zone active','Type of location','Urbanisation', 'Conurbation 1',
                       'Alignment','Street lighting','Weather','Natural lighting',
                       'Signals operation','Other traffic control','Road classification','First impact type',
                      'Key TU type','Other TU type','LGA']
    df=categorical_encoding(df,columns_cat_encod)
    print("Dataset after categorical encoding")
    print(df.head())

    #Checking for class imbalance
    inspect_class_imbalance(df)

    #Checking for outliers
    outlier_detection(df)

    
    # Save cleaned dataset
    output_path = Path("data/processed/cleaned_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    run_preprocessing()
