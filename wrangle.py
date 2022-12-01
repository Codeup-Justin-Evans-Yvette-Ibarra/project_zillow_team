######################### IMPORTS #########################

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import env
from env import user, password, host



######################### ACQUIRE DATA #########################

def get_db_url(db):

    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#------------------------- ZILLOW DATA FROM SQL -------------------------

def fresh_zillow_data():
    ''' fresh_zillow_data acquires Zillow data using properties_2017 
    table from Code up Data Base. '''
    
    # Create SQL query.
    query = '''
            SELECT
            prop.*,
            predictions_2017.logerror as log_error,
            predictions_2017.transactiondate as transaction_date,
            air.airconditioningdesc as aircondition,
            arch.architecturalstyledesc as architectural_style,
            build.buildingclassdesc as bulding_class,
            heat.heatingorsystemdesc as heat_systems,
            landuse.propertylandusedesc as land_use,
            story.storydesc as story,
            construct.typeconstructiondesc as construction_type
            FROM properties_2017 prop
            JOIN (
                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid
            ) as pred USING(parcelid)
            JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                                  AND pred.max_transactiondate = predictions_2017.transactiondate
            LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
            LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
            LEFT JOIN storytype story USING (storytypeid)
            LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
            WHERE prop.latitude IS NOT NULL
              AND prop.longitude IS NOT NULL
              AND transactiondate <= '2017-12-31'
              AND propertylandusedesc like '%%Single%%'
            '''

    # Read in DataFrame from Codeup db using defined arguments.
    df = pd.read_sql(query, get_db_url('zillow'))

    return df

def get_zillow_data(new = False):
    '''
    This function checks for a local file and reads it in as a Datafile.  If the csv file does not exist, it calls the new_wrangle function then writes the data to a csv file.
    '''
   
    filename = 'zillow.csv'

    # obtain cvs file
    if (os.path.isfile(filename) == False) or (new == True):
        df = fresh_zillow_data()
    #save as csv
    df.to_csv(filename,index=False)

    #cached data
    else:
        df = pd.read_csv(filename)

    return df

######################### PREPARE DATA #########################

def zillow_prep(df):

    """
    This function is used to clean the zillow_2017 data as needed 
    ensuring not to introduce any new data but only remove irrelevant data 
    or reshape existing data to useable formats.
    """

    # remove outliers
    df = handle_outliers(df)
    
    # Replace Null with 0 to create Categorical Features
    df = convert_null_to_cat(df)
    
    # Feature Engineer: Home_age and optional_feature
    df = new_features(df)

    #encode categorical features or turn to 0 & 1:
    df = encode_features(df)

    # rename dummy county to matching county name
    df = rename_county(df)

    #--------------- DROP NULL/NaN ---------------#

    # Drop all columns with more than 19,000 NULL/NaN
    df = df.dropna(axis='columns', thresh=19_000)

    # Drop rows with NULL/NaN since it is only 3% of DataFrame 
    df = df.dropna()

    
    
    return df




###################################### Outliers 

# filter down outliers to more accurately align with realistic expectations of a Single Family Residence

def handle_outliers(df):

    # Set no_outliers equal to df
    no_outliers = df

    # Keep all homes that have > 0 and <= 8 Beds and Baths
    no_outliers = no_outliers[no_outliers.bedroomcnt > 0]
    no_outliers = no_outliers[no_outliers.bathroomcnt > 0]
    no_outliers = no_outliers[no_outliers.bedroomcnt <= 8]
    no_outliers = no_outliers[no_outliers.bathroomcnt <= 8]
    
    # Keep all homes that have tax value <= 2 million
    # no_outliers = no_outliers[no_outliers.taxvaluedollarcnt >= 40_000]
    no_outliers = no_outliers[no_outliers.taxvaluedollarcnt <= 2_000_000]
    
    # Keep all homes that have sqft > 400 and < 10_000
    no_outliers = no_outliers[no_outliers.calculatedfinishedsquarefeet > 400]
    no_outliers = no_outliers[no_outliers.calculatedfinishedsquarefeet < 10_000]

    # Assign no_outliers back to the DataFrame
    df = no_outliers
    
    return df


####################################### Features 

def convert_null_to_cat(df):
    """
    Replace Null with 0 to create Categorical Features
    """
    
    columns_to_convert = ['basementsqft',
                          'decktypeid', 
                          'pooltypeid10', 
                          'poolsizesum', 
                          'pooltypeid2', 
                          'pooltypeid7', 
                          'poolcnt', 
                          'hashottuborspa', 
                          'taxdelinquencyyear', 
                          'fireplacecnt', 
                          'numberofstories', 
                          'garagecarcnt', 
                          'garagetotalsqft']

    df = df[columns_to_convert].fillna(0)
    
    return df



def new_features(df):
    ''' new_features takes in dataframe'''
    #Creating new column for home age using year_built, casting as float
    df['home_age'] = 2017- df['yearbuilt']
    df["home_age"] = df["home_age"].astype('float')
    
    df['optional_features'] = (df.garagecarcnt==1)|(df.decktypeid == 1)|(df.poolcnt == 1)|(df.fireplacecnt == 1)
    
    return df
    
def encode_features(df):
    df.fireplacecnt = df.fireplacecnt.replace({2:1, 3:1, 4:1, 5:1})
    df.decktypeid= df.decktypeid.replace({66:1})
    df.garagecarcnt = df.garagecarcnt.replace({2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 13:1,14:1})
    df.optional_features = df.optional_features.replace({False:0, True: 1})
    temp = pd.get_dummies(df['county'], drop_first=False)
    df = pd.concat([df, temp],axis =1)
    return df 

def rename_county(df):
    # 6111 Ventura County, 6059  Orange County, 6037 Los Angeles County 
    df = df.rename(columns={6111.0: 'ventura_county',6059.0: 'orange_county',
            6037: 'los_angeles_county'}) 
    return df
####################################### NULL VALUES ############################################
def null_counter(df):
    ''' null_counter takes in a dataframe anc calculates the percent and amount of null cells in each column and row
    returns a dataframe with the results'''
    # name of dataframe names
    new_columns = ['name', 'num_rows_missing', 'pct_rows_missing']
    
    # create data frame
    new_df = pd.DataFrame(columns = new_columns)
   
    # for loop to calculate missing /percent by columns
    for col in list(df.columns):
        num_missing = df[col].isna().sum()
        pct_missing = num_missing / df.shape[0]
        
        # create data frame
        add_df = pd.DataFrame([{'name': col,
                               'num_rows_missing': num_missing,
                               'pct_rows_missing': pct_missing}])
       
        # concat and index by row by seting axis to 0   
        new_df = pd.concat([new_df, add_df], axis = 0)
        
    # sets the index name
    new_df.set_index('name', inplace = True)
    
    return new_df

def null_dropper(df,prop_required_column,prop_required_row):

    ''' null_dropper takes in a dataframe a percent of required columns and rows to keep columns.
    all columns and rows outside of the null threshold will be dropped
    returns a clean dataframe dropped nulls'''
    
    # this is a decimal = 1- decimal
    prop_null_column = 1-prop_required_column
    
    # for columns, check null percentage and drop if a certain proportion is null (set by definition)
    for col in list(df.columns):
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns = col, inplace = True)
    
    # for rows, drop if a certain proportion is null. (set by definition)
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis = 0, thresh=row_threshold, inplace = True)
    
    return df


###################################### Split Data

def split_data(df):
    '''
    split_data takes in data Frame and splits into  train , validate, test.
    The split is 20% test 80% train/validate. Then 30% of 80% validate and 70% of 80% train.
    Aproximately (train 56%, validate 24%, test 20%)
    Returns train, validate, and test 
    '''
    # split test data from train/validate
    train_and_validate, test = train_test_split(df, random_state=123, test_size=.2)

    # split train from validate
    train, validate = train_test_split(train_and_validate, random_state=123, test_size=.3)
                                   
    return train, validate, test



######################### SCALE SPLIT #########################

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               scaler,
               return_scaler = False):
    
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data 
    splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    
    Imports Needed:
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import QuantileTransformer
    
    Arguments Taken:
               train = Assign the train DataFrame
            validate = Assign the validate DataFrame 
                test = Assign the test DataFrame
    columns_to_scale = Assign the Columns that you want to scale
              scaler = Assign the scaler to use MinMaxScaler(),
                                                StandardScaler(), 
                                                RobustScaler(), or 
                                                QuantileTransformer()
       return_scaler = False by default and will not return scaler data
                       True will return the scaler data before displaying the _scaled data
    """
    
    # make copies of our original data so we dont corrupt original split
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # fit the scaled data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        return train_scaled, validate_scaled, test_scaled


    
######################### DATA SCALE VISUALIZATION #########################

# Function Stolen from Codeup Instructor Andrew King
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This Function takes input arguments, 
    creates a copy of the df argument, 
    scales it according to the scaler argument, 
    then displays subplots of the columns_to_scale argument 
    before and after scaling.
    """    

    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    #return df_scaled.head().T
    #return fig, axs