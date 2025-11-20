import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def inspect_null_values(df):
    print("Null values per feature:")
    print(df.isnull().sum())
    print("-" * 50)

def inspect_categories(df, column):
    counts = df[column].value_counts().to_dict()
    print(f"\nNumber of categories for {column}: {len(counts)}")
    print(counts)

def inspect_class_imbalance(df):
    # Target variable is Degree of crash - detailed
    print("Lets visualise the distrubution of target variable")
    plt.figure(figsize=(7,5))
    sns.countplot(x='Degree of crash - detailed', data=df, palette='pastel')
    plt.title('Class Distribution', fontsize=14)
    plt.xlabel('Degree of crash')
    plt.ylabel('Count')
    plt.savefig('outputs/class_imbalance.png', dpi=300, bbox_inches='tight')
    #plt.show()

def outlier_detection(df):
    plt.figure(figsize=(8,6))
    sns.boxplot(y=df['Speed limit'],color='skyblue')
    plt.savefig('outputs/outlier_speed_limit.png', dpi=300, bbox_inches='tight')
    
    sns.boxplot(y=df['Weather'],color='skyblue')
    plt.savefig('outputs/outlier_weather.png', dpi=300, bbox_inches='tight')
    

