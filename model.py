import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
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
    validate_scaled = pd.concat([validate_scaled, temp],axis=1)
    
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

def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)



##################################### modeling 1###################




def modeling(X_train_scaled, y_train, X_validate_scaled,y_validate, X_test_scaled, y_test):    
    model = LinearRegression().fit(X_train_scaled, y_train)
    predictions = model.predict(X_train_scaled)

    #  y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict log_pred_mean
    log_pred_mean = y_train.log_error.mean()
    y_train['logerror_pred_mean'] = log_pred_mean
    y_validate['logerror_pred_mean'] =log_pred_mean

    # 2. RMSE of log_pred_mean
    rmse_train = mean_squared_error(y_train.log_error,y_train.logerror_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.logerror_pred_mean) ** (1/2)

    
    
    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame()
    # make our first entry into the metric_df with median baseline
    metric_df = make_metric_df(y_train.log_error,
                               y_train.logerror_pred_mean,
                               'mean_baseline',
                              metric_df)

    
    # Simple Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_scaled, y_train.log_error)
    y_train['log_pred_lm'] = lm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_pred_lm) ** (1/2)

    # predict validate
    y_validate['log_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_pred_lm) ** (1/2)
    print('Features set 1')
    print("RMSE for F1: OLS using LinearRegression\nTraining/In-Sample: ", rmse_train,"\nValidation/Out-of-Sample: ", rmse_validate)
    metric_df = metric_df.append({
        'model': 'F1: OLS Regressor', 
        'RMSE_validate': rmse_validate,
        'r^2_validate': explained_variance_score(y_validate.log_error, y_validate.logerror_pred_mean)}, ignore_index=True)

    print('_______________')
    
    
    #Polynomial 2 degrees
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 =  pf.transform(X_test_scaled)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.log_error)

    # predict train
    y_train['log_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_pred_lm2) ** (1/2)

    # predict validate
    y_validate['log_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_pred_lm2) ** 0.5

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, "\nValidation/Out-of-Sample: ", rmse_validate)

    # append to metric df
    metric_df = make_metric_df(y_validate.log_error, y_validate.log_pred_lm2,'F1: degree2',metric_df)

    return metric_df[['model', 'RMSE_validate']]


################################################ model prep and models 2#########################

def model_data_prep2(train_scaled, validate_scaled,test_scaled, features_to_model):
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

def modeling2(X_train_scaled, y_train, X_validate_scaled,y_validate, X_test_scaled, y_test):    
    model = LinearRegression().fit(X_train_scaled, y_train)
    predictions = model.predict(X_train_scaled)

    #  y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict log_pred_mean
    log_pred_mean = y_train.log_error.mean()
    y_train['logerror_pred_mean'] = log_pred_mean
    y_validate['logerror_pred_mean'] =log_pred_mean

    # 2. RMSE of log_pred_mean
    rmse_train = mean_squared_error(y_train.log_error,y_train.logerror_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.logerror_pred_mean) ** (1/2)

    
    
    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame()
    # make our first entry into the metric_df with median baseline
    metric_df = make_metric_df(y_train.log_error,
                               y_train.logerror_pred_mean,
                               'mean_baseline',
                              metric_df)

    
    # Simple Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train_scaled, y_train.log_error)
    y_train['log_pred_lm'] = lm.predict(X_train_scaled)
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_pred_lm) ** (1/2)

    # predict validate
    y_validate['log_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_pred_lm) ** (1/2)
    print ('Features Set 2')
    print("RMSE for F2: OLS using LinearRegression\nTraining/In-Sample: ", rmse_train,"\nValidation/Out-of-Sample: ", rmse_validate)
    metric_df = metric_df.append({
        'model': 'F2: OLS Regressor', 
        'RMSE_validate': rmse_validate,
        'r^2_validate': explained_variance_score(y_validate.log_error, y_validate.logerror_pred_mean)}, ignore_index=True)

    print('_______________')
    
    
    #Polynomial 2 degrees
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 =  pf.transform(X_test_scaled)

    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.log_error)

    # predict train
    y_train['log_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.log_error, y_train.log_pred_lm2) ** (1/2)

    # predict validate
    y_validate['log_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.log_error, y_validate.log_pred_lm2) ** 0.5

    print("RMSE for F2: Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, "\nValidation/Out-of-Sample: ", rmse_validate)

    # append to metric df
    metric_df = make_metric_df(y_validate.log_error, y_validate.log_pred_lm2,'F2: degree2',metric_df)

    return metric_df[['model', 'RMSE_validate']]