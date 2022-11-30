import env
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

########################################## Acquire ##########################################
def fresh_zillow_data():
    ''' fresh_zillow_data acquires Zillow data using properties_2017 
    table from Code up Data Base. '''
    
    sql_query = '''
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

    # create data frame
    df = pd.read_sql(sql_query, env.get_db_url('zillow'))

    return df

def get_zillow_data(new = False):
        ''' Acquire Zillow data using properties_2017 table from Code up Data Base. Columns bedroomcnt, 
            bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
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
################################################## Prepare Data ##################################
def zillow_prep(df):
    ''' This function takes in a data frame and preps data frame. returns dataframe'''
    
    # remove outliers
    df = handle_outliers(df)
    
    # removed rows with 0 beds and 0 baths
    df = df[~(df.bathrooms==0) & ~(df.bedrooms ==0)]
    
    # process nulls in luxury features:
    df = process_optional_features(df)
    
    # drop nulls
    df = df.dropna()

    # feature engineer: Home_age and optiona_feature
    df = new_features(df)

    #encode categorical features or turn to 0 & 1:
    df = encode_features(df)

    # rename dummy county to matching county name
    df = rename_county(df)

    return df



###################################### Outliers 
def handle_outliers(df):
    """Manually handle outliers '"""
    # drop bathrooms, bedrooms and home_value
    df = df[df.bathrooms <= 6] 
    df = df[df.bedrooms <= 6] 
    df = df[df.home_value <= 1_750_000]
    
    return df



####################################### Features 
def process_optional_features(df):
    ''' process_optional_features will take in zillow data and clean up columns
    fireplace, deck, pool, garage by replacing nulls with 0
    returns data frame'''
    columns = ['fireplace','deck','pool','garage']    
    for feature in columns:
        df[feature]=df[feature].replace(r"^\s*$", np.nan, regex=True)     
        # fill optional features with 0 assumption that if it was not mark it did not exist
        df[feature] = df[feature].fillna(0)
    return df

def new_features(df):
    ''' new_features takes in dataframe'''
    #Creating new column for home age using year_built, casting as float
    df['home_age'] = 2017- df['yearbuilt']
    df["home_age"] = df["home_age"].astype('float')
    
    df['optional_features'] = (df.garage==1)|(df.deck == 1)|(df.pool == 1)|(df.fireplace == 1)
    
    return df
    
def encode_features(df):
    df.fireplace = df.fireplace.replace({2:1, 3:1, 4:1, 5:1})
    df.deck= df.deck.replace({66:1})
    df.garage = df.garage.replace({2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 13:1,14:1})
    df.optional_features = df.optional_features.replace({False:0, True: 1})
    temp = pd.get_dummies(df['county'], drop_first=False)
    df = pd.concat([df, temp],axis =1)
    return df 

def rename_county(df):
    # 6111 Ventura County, 6059  Orange County, 6037 Los Angeles County 
    df = df.rename(columns={6111.0: 'ventura_county',6059.0: 'orange_county',
            6037: 'los_angeles_county'}) 
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

def null_dropper(df,prop_required_column,prop_requred_row):

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
    row_threshold = int(prop_requred_row * df.shape[1])
    
    df.dropna(axis = 0, thresh=row_threshold, inplace = True)
    
    return df