# Imports
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
 

def Q1_viz_1(df):
    sns.lmplot(y='longitude', x='log_error', data=df, line_kws={'color': 'red'})
    return plt.show()

def Q1_viz_2(df):
    sns.lmplot(y='latitude', x='log_error', data=df, line_kws={'color': 'red'})
    plt.show()

def Q2_viz_1(df):
    sns.barplot(x='county', y='log_error', hue='loc_clusters',
            palette='colorblind', data = df)

def Q2_viz_2(df):
    sns.barplot(x='loc_clusters', y='log_error', data=df)
    plt.show()

def split_stats(df, train, validate, test):
    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')

def Q2_test_1(df):
    alpha = 0.05

    group_list = [df[df.loc_clusters == x].log_error.to_numpy() for x in range(5)]
    t,p_val = stats.kruskal(group_list[0],group_list[1],group_list[2],group_list[3],group_list[4])
    
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    
    return t,p_val

def pearson_r(df, sample_1, sample_2):
    """
    """
    alpha = 0.05
    r, p_val = stats.pearsonr(df[sample_1], df[sample_2])
    
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    
    return r, p_val


