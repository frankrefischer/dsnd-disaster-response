import sys
import pandas as pd
import sqlalchemy

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('{} rows in cleaned data'.format(len(df)))
        
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

def load_data(messages_filepath, categories_filepath):
    """Load the message data and the category data from csv files and join them to a dataframe.
    
    Keyword arguments:
    messages_filepath   -- load messages from this csv file
    categories_filepath -- load categories from this csv file
    
    Return value:
    the joined dataframe
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer')

    return df

def clean_data(df_orig):
    """Clean the joined dataframe.
    
    All cleaning steps do not alter the original dataframe
    
    Keyword arguments:
    df_orig -- the original dataframe to clean. It is not altered.
    
    Return value:
    the cleaned dataframe
    
    """
    df = df_orig
    df = split_categories_into_separate_numerical_columns(df)
    df = remove_duplicates(df)
    
    return df

def save_data(df, database_filename):
    """Save dataframe to a sqlite database. Tablename is 'DisasterResponse'
    
    Keyword arguments:
    df                -- the dataframe to save
    database_filename -- store into this file
    """

    sqlite_engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filename))
    
    df.to_sql('DisasterResponse', sqlite_engine, index=False)

    print("===== TEST: rereading database file")
    pd.read_sql_table('DisasterResponse', sqlite_engine)
    print("=== SUCCESS ===")

def split_categories_into_separate_numerical_columns(df_orig):
    """Transforms the column named 'categories' in dataframe df.
    It expects that each value is a string, consisting of parts separated by ';'.
    Each part has the form <name>-0 or <name>-1.
    The <name>s will become columns in df, the 0 and 1 will become numerical values of these columns.
    The original column 'categories' will be dropped.
    The original dataframe remains unaltered.
    
    Keyword arguments:
    df -- the original dataframe
    
    Return value:
    a dataframe with column 'categories' replaced with new columns for each category.
    """
    
    cats_splitted = df_orig.categories.str.split(pat=';', expand=True)
    
    def set_meaningful_columnnames(cats):
        def extract_columnnames_from_row(row):
            def get_namepart_of_category_value(c):
                return c.split('-')[0]
            return row.apply(get_namepart_of_category_value).values

        first_row = cats.iloc[0]
        cats.columns = extract_columnnames_from_row(first_row)
        print("{} category columns inserted: {}".format(len(cats.columns), cats.columns.values))

    set_meaningful_columnnames(cats_splitted)
        
    def convert_column_to_numeric(c):
        def get_numerical_part_of_category_value(c):
            return c.split('-')[1]
        return pd.to_numeric( c.apply(get_numerical_part_of_category_value) )
    
    for c in cats_splitted.columns:
        cats_splitted[c] = convert_column_to_numeric(cats_splitted[c])
        
    df = pd.concat([df_orig.drop('categories', axis=1), cats_splitted], axis=1)
    
    return df

def remove_duplicates(df_orig):
    """Removes duplicate rows from the dataframe.
    
    Keyword arguments:
    df_orig -- original dataframe, will not be altered.
    
    Return value:
    the dataframe without duplicates
    """
    df = df_orig.drop_duplicates()
    print("{} duplicate rows removed".format(len(df_orig) - len(df)))
          
    return df
    

if __name__ == '__main__':
    main()