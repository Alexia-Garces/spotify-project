import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

###################### Acquire Spotify Data ######################

def acquire_spotify():
    '''
    This function reads the spotify data from the kaggle into a df, 
    combines all decades into one df,
    writes it to a csv file, and returns the df.
    '''
    #import all decades CSVs from kaggle
    sixties = pd.DataFrame(pd.read_csv('dataset_60s.csv')) #Spotify dataset of all songs from the 60s
    seventies = pd.DataFrame(pd.read_csv('dataset_70s.csv'))  #Spotify dataset of all songs from the 70s
    eighties = pd.DataFrame(pd.read_csv('dataset_80s.csv'))  #Spotify dataset of all songs from the 80s
    nineties = pd.DataFrame(pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/496640/1108669/dataset-of-90s.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211013%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211013T164338Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=62e18280c978225bd671255625bfe818c8fb76fe2d9736e6635d1fa3a12197e0b705545d4b36dc0621fb4b82cb155055bb8282e1b917a88f24ec17b61b313927ba6170d551596f940d6bebfda91871f295ea8c13dd38eacfe920479b8e3f17d995db5bf65a73ce1204cca15d559ae29e6026ec32f3176ecca97a4df7feaf95c9714c83233d0cfebcef04a011a6bc88359afb763c589bddf57298ae94e8f7a69bb9b61248995817aeb6eba446bb23fb90b9da430571485b5fe2a84a2f2da2f08e8b3e8a926b6e7f3c15fb38c97816803b4c696297513ed3459d3a066c35a7c19e0f90185a6035487e34f326dd173a9f292a6458956a3b5adb2a32d8339313e4a1')) #Spotify dataset of all songs from the 90s
    aughts = pd.DataFrame(pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/496640/1108669/dataset-of-00s.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211013%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211013T164833Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=76e7e6a32b4cc9d901cc8e91927e86a508100076e20b033c87f22cd440e7d8cbb134bc8c4165f08bb6db9fae83eca610e32f57ded2185eb587e6be27512c0f9648baf93bfe3bd092b04a4e988331f798f86c505018ce3d6b1d92a1a4b4ec0b4ef5f3628eeea3d870db90108d3d9082a17a57496d01e761aad41a47851ad1eb4f599f7ebba31e5848c693d0d2e938ef653e8d99713b7ebf1fcf0f47d610e7dc461ecf59370df9893f50038d4b6cf2b6caeda72eedac83c8d4aa7e3c1edd124b99f1d5ceb993f19b0af546f72810b05116220af5c2156dbb445eb8a09afb75334eea0733941d80c76877cf29b4fa55ba0cea3288db61aa70a4dc321a0f02788eb6'))  #Spotify dataset of all songs from the 2000s
    tens = pd.DataFrame(pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/496640/1108669/dataset-of-10s.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20211013%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20211013T164905Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=8a096269bf9a272244ce146361144e8ca50f98a91bf2267bb2e625896623f99d6f59820f4c057060899a23764635418b3967cd14d49468631d76d26b750a62b9ab543956652676cd849c375fe15599b6d5e2e6935e4f8788a17e9da209a3d8e9d5f30c41e6e1595e8ed78f4b02b69035d9feb4959270de965ea968c578e7456bd8c76ce234b46ec2b6c1574c5510455cac3d91fc6d7a5a9e1ba08422a20151beaba59776f6219525d7a8943224ca2d6985098d86ae72cb2942f764e2a41dab34f06a44ed72a57a16a1c24c7533f917aad926c63b5caf01dd24e4c1b61491c1cc2323d67dd79047fe37c63c2b7b259dcd2d75646d2e5c014ac72eb7a74471799a'))  #Spotify dataset of all songs from the 2010s
    
    #add decade column
    sixties['decade'] = '1960s'
    seventies['decade'] = '1970s'
    eighties['decade'] = '1980s'
    nineties['decade'] = '1990s'
    aughts['decade'] = '2000s'
    tens['decade'] = '2010s'
    
    #Combine all dataframes into one
    all_dfs = [sixties, seventies, eighties, nineties, aughts, tens]
    df = pd.concat(all_dfs)
    
    #create csv of all songs
    df.to_csv('songs_from_all_decades.csv', index = False)
    
    return df

def get_spotify_data():
    '''
    This function reads in spotify data, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('songs_from_all_decades.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('songs_from_all_decades.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = acquire_spotify()
        
        # Write DataFrame to a csv file.
        df.to_csv('songs_from_all_decades.csv')
        
    return df

###################### Prepare Spotify Data ######################

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    
    # take a look at the shape of each df
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')

    return train, validate, test

def prep_spotify_model(df):
    '''
    This function takes in a dataframe and returns the cleaned, encoded and split data.
    Adds autopay column and encodes categoricals.
    Use this function before modeling.
    returns train, validate, test
    '''
    #drop redundant columns
    df = df.drop(columns =['track', 'artist', 'uri'])
    
    #make a dummy df, and combining it back to the original df. Dropping redundant columns again.
    dummy_df = pd.get_dummies(df[['decade']], drop_first=False)
    df = pd.concat([df, dummy_df], axis =1)
    
    #drop redundant columns
    df = df.drop(columns =['decade'])
    
    # split into train validate and test 
    train, validate, test = train_validate_test_split(df, target='target', seed=123)

    return train, validate, test

############################### Scaler Function##################################
def tip_the_scale(train, validate, test, column_names, scaler, scaler_name):
    
    '''
    This function takes in the train validate and test dataframes, list of columns you want to scale, a scaler type,
    scaler_name
    column_names: list of columns to scale
    scaler_name, the name for the new dataframe columns
    adds columns to the train validate and test dataframes
    outputs scaler for doing inverse transforms
    ouputs a list of the new column names
    
    '''
    
    #create the scaler (input here should be scaler type used)
    mm_scaler = scaler
    
    #make empty list for return
    scaled_column_list = []
    
    #loop through columns in col names
    for col in column_names:
        
        #fit and transform to train, add to new column on train df
        train[f'{col}_{scaler_name}'] = mm_scaler.fit_transform(train[[col]]) 
        
        #df['col'].values.reshape(-1, 1)
        
        #transform cols from validate and test (only fit on train)
        validate[f'{col}_{scaler_name}']= mm_scaler.transform(validate[[col]])
        test[f'{col}_{scaler_name}']= mm_scaler.transform(test[[col]])
        
        #add new column name to the list that will get returned
        scaled_column_list.append(f'{col}_{scaler_name}')
    
    #returns scaler, and a list of column names that can be used in X_train, X_validate and X_test.
    return scaler, scaled_column_list 
