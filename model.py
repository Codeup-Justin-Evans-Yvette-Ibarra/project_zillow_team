import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans



################################ scaled data ####################################################
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               scaler=MinMaxScaler(),
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



############################################# cluster ###############################################

def create_cluster (X_cluster_name,train,validate, test, feature_columns,n_clusters=3):
    ''' create_cluster takes in 
        X_cluster_name: name of cluster
        train: train data
        validate: validate data
        test: test data
        feature_columns: column list of features for clustering and scaling
        cluster: default to 3 number of desired clusters
        
        *scales using scaled_data minmaxscaler()
        *fits cluster into train_scaled data
        *creates new column for cluster in train, validate and test
        
        returns train, validate ,test, train_scaled, validate_scaled, test_scaled
        '''
    # name the cluster
    name= X_cluster_name
    
    # scaled the data
    columns_to_scale = ['longitude','latitude','age','bedrooms',
    'bathrooms', 'yearbuilt','optional_features', 'sqft',
    'lot_sqft','taxamount','tax_value','tax_value_bldg','tax_value_land']
    train_scaled, validate_scaled, test_scaled = scale_data(train, 
               validate, 
               test, 
               columns_to_scale)
    
    # fit in to train and create cluster
    X_cluster_name = train_scaled[feature_columns]
    kmeans = KMeans(n_clusters=n_clusters, random_state = 123)
    kmeans.fit(X_cluster_name)
    

    # save into train_scaled and train
    train_scaled[str(name) ] = kmeans.predict(X_cluster_name)
    train[str(name)]=kmeans.predict(X_cluster_name)
    
    # Create seperate catagorical features for each cluster in train
    temp = pd.get_dummies(train[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    train = pd.concat([train, temp],axis=1)
    
    # Create seperate catagorical features for each cluster in train_scaled
    temp = pd.get_dummies(train_scaled[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    train_scaled = pd.concat([train_scaled, temp],axis=1)
    
    # save cluster in to validate_scaled and validate
    X_cluster_name = validate_scaled[feature_columns]
    validate_scaled[str(name) ] = kmeans.predict(X_cluster_name)
    validate[str(name)]=kmeans.predict(X_cluster_name)
    
    # Create seperate catagorical features for each cluster in train
    temp = pd.get_dummies(validate[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    validate = pd.concat([validate, temp],axis=1)
    
    # Create seperate catagorical features for each cluster in train_scaled
    temp = pd.get_dummies(validate_scaled[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    validate_scaled = pd.concat([train_scaled, temp],axis=1)
    
    # save cluser into test_scaled and test
    X_cluster_name = test_scaled[feature_columns]
    test_scaled[str(name) ] = kmeans.predict(X_cluster_name)
    test[str(name)]=kmeans.predict(X_cluster_name)
    
    # Create seperate catagorical features for each cluster in train
    temp = pd.get_dummies(test[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    test = pd.concat([test, temp],axis=1)
    
    # Create seperate catagorical features for each cluster in train_scaled
    temp = pd.get_dummies(test_scaled[name], drop_first=False).rename(columns=lambda x:f'{name}_'+str(x))
    test_scaled = pd.concat([test_scaled, temp],axis=1)
    

    return train, validate, test, train_scaled, validate_scaled, test_scaled


    ################################ seperate target, split #############################################

def model_data_prep(train_scaled, validate_scaled,test_scaled, features_to_model):
    '''model_data_prep takes in train validate,test and scales using scale_data and sets up
    features and target ready for modeling
    '''
    train1 = train_scaled[features_to_model]
    validate1 = validate_scaled[features_to_model]
    test1 = test_scaled[features_to_model]
    
    X_train_scaled = train1
    X_validate_scaled = validate1
    X_test_scaled =test1


    # Setup X and y
    X_train_scaled = X_train_scaled.drop(columns=['log_error'])
    y_train = train1.log_error

    X_validate_scaled = X_validate_scaled.drop(columns=['log_error'])
    y_validate = validate1.log_error

    X_test_scaled = X_test_scaled.drop(columns=['log_error'])
    y_test = test1.log_error
    
    return X_train_scaled,y_train, X_validate_scaled,y_validate, X_test_scaled, y_test