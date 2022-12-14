######################### IMPORTS #########################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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

    #encode categorical features or turn to 0 & 1:
    df = encode_features(df)

    # Feature Engineer: Home_age and optional_feature
    df = new_features(df)

    # Converts FIPS code to State and County and pivot to categorical features by county
    df = fips_conversion(df)

    # Rearange Columns for Human Readability
    df = rearange_columns(df)

    # Rename Columns
    df = rename_columns(df)

      #--------------- DROP NULL/NaN ---------------#

    # Drop all columns with more than 19,000 NULL/NaN
    df = df.dropna(axis='columns', thresh = 50_000)

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

    df[columns_to_convert] = df[columns_to_convert].fillna(0)
    
    return df
   
def encode_features(df):
    # Replace Conditional values
    df["taxdelinquencyyear"] = np.where(df["taxdelinquencyyear"] > 0, 1, 0)
    df["basementsqft"] = np.where(df["basementsqft"] > 0, 1, 0)
    df.fireplacecnt = np.where(df["fireplacecnt"] > 0, 1, 0)
    df.decktypeid = np.where(df["decktypeid"] > 0, 1, 0)
    df.garagecarcnt = np.where(df["garagecarcnt"] > 0, 1, 0)

    return df

def new_features(df):
    ''' new_features takes in dataframe'''
    # Create a feature that shows the age of the home in 2017
    df['age'] = 2017 - df.yearbuilt
    
    # Create Categorical Feature that shows count of "Optional Additions" 
    df['optional_features'] = (df.garagecarcnt==1)|(df.decktypeid == 1)|(df.poolcnt == 1)|(df.fireplacecnt == 1)|(df.basementsqft == 1)|(df.hashottuborspa == 1)
    df.optional_features = df.optional_features.replace({False:0, True: 1})
 
    # add absolute log_error to dataframe
    df['abs_log_error'] = np.abs(df.log_error)

    return df 

def fips_conversion(df):
    """
    Found a csv fips master list on github. Used it to convert FIPS to State and County Features.    
    """

    # Read in as a DataFrame using raw url
    url = 'https://raw.githubusercontent.com/kjhealy/fips-codes/master/state_and_county_fips_master.csv'
    fips_df = pd.read_csv(url)
    
    # Cache data into a new csv file
    fips_df.to_csv('state_and_county_fips_master.csv')
    
    # left merge to join the name and state to the original df
    left_merged_fips_df = pd.merge(df, fips_df, how="left", on=["fips"])
    
    # Create seperate catagorical features for each county "name" 
    temp = pd.get_dummies(left_merged_fips_df['name'], drop_first=False)
    left_merged_fips_df = pd.concat([left_merged_fips_df, temp],axis =1)
    
    # assign to df
    df = left_merged_fips_df

    return df

def rearange_columns(df):
    """
    REARANGE Columns for Human Readability.  
    The following columns were dropped by not adding them into the rearrange assignment.

'id', 
'garagetotalsqft', 
'poolsizesum',
'pooltypeid10', 
'pooltypeid2', 
'pooltypeid7',
'propertycountylandusecode', 
'propertylandusetypeid',
'roomcnt',
'numberofstories', 
'assessmentyear', 
'finishedsquarefeet12',
'transaction_date', 
'land_use', 
    """

    df = df[['parcelid',
            'bedroomcnt',
            'bathroomcnt', 
            'calculatedbathnbr', 
            'fullbathcnt',
            'age', 
            'yearbuilt', 
            'basementsqft', 
            'decktypeid', 
            'fireplacecnt', 
            'garagecarcnt', 
            'hashottuborspa', 
            'poolcnt', 
            'optional_features', 
            'taxdelinquencyyear', 
            'fips',
            'state', 
            'name',
            'Los Angeles County', 
            'Orange County', 
            'Ventura County',
            'longitude', 
            'latitude',
            'regionidzip', 
            'regionidcounty', 
            'rawcensustractandblock', 
            'censustractandblock', 
            'calculatedfinishedsquarefeet',
            'lotsizesquarefeet', 
            'structuretaxvaluedollarcnt',
            'taxvaluedollarcnt', 
            'landtaxvaluedollarcnt',
            'taxamount', 
            'log_error',
            'abs_log_error']]

    return df

def rename_columns(df):
    """
    This Function renames the Binary Categorical Columns to better identify them.
    It also renames several Features to be more Human Readable and less cumbersome to call.
    """

    #### Rename Binary Categoricals
    df.rename(columns = {'hashottuborspa': 'has_hottuborspa',
                        'taxdelinquencyyear': 'has_taxdelinquency', 
                        'basementsqft': 'has_basement', 
                        'poolcnt': 'has_pool', 
                        'decktypeid': 'has_deck',
                        'fireplacecnt': 'has_fireplace',
                        'garagecarcnt': 'has_garage',
                        'Los Angeles County': 'la_county',
                        'Ventura County': 'ventura_county',
                        'Orange County': 'orange_county'}
            , inplace = True)

    #### Rename Human Readable
    df.rename(columns = {'name': 'county',
                        'bedroomcnt': 'bedrooms',
                        'bathroomcnt': 'bathrooms',
                        'structuretaxvaluedollarcnt': 'tax_value_bldg',
                        'taxvaluedollarcnt': 'tax_value',
                        'landtaxvaluedollarcnt': 'tax_value_land',
                        'regionidzip': 'zipcode',
                        'lotsizesquarefeet': 'lot_sqft',
                        'calculatedfinishedsquarefeet': 'sqft'}
            , inplace = True)

    return df


###################################### Split Data

def split(df):
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