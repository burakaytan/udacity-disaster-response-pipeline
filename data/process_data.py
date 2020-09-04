import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.slice(start=0, stop=-2).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # load categories dataset
    org_categories = pd.read_csv(categories_filepath)
    # concat last categories with original one to get id
    categories = pd.concat([org_categories,categories], axis=1)
    # drop the original categories column from `categories`
    categories.drop("categories", axis=1, inplace=True)
    # merge datasets
    df = pd.merge(messages,categories,on='id')
    
    return df

def clean_data(df):
    #remove rows not containing 0 or 1
    binary_columns = ['related', 'request', 'offer',
        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']
    for column in binary_columns:
        df.drop(df[(df[column]!=0) & (df[column]!=1)].index, inplace=True)
    # drop duplicates
    df2 = df.drop_duplicates()
    return df2

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ str (database_filename))
    df.to_sql('messages', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()