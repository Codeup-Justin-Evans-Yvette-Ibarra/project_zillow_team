# Imports
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
 

def Q1_viz_1(df):
    # Plot data and a linear regression
    plt.figure(figsize=(20,8))

    plt.subplot(131)
    sns.regplot(y="longitude", 
                x="log_error", 
                data=df, 
                line_kws={'color': 'red'})
    plt.title("Longitude & Log Error") #Add plot title
    plt.ylabel("longitude") #Adjust the label of the y-axis
    plt.ticklabel_format(style='plain', axis='y') #repress the scientific notation
    plt.xlabel("log_error") #Adjust the label of the x-axis
    plt.ticklabel_format(style='plain', axis='x') #repress the scientific notation
    #plt.ylim(0,100) #Adjust the limits of the y-axis
    #plt.xlim(0,10) #Adjust the limits of the x-axis
    plt.tight_layout() #Adjust subplot params

    plt.subplot(132)
    sns.regplot(y="latitude", 
                x="log_error", 
                data=df, 
                line_kws={'color': 'red'})
    plt.title("Latitude & Log Error") #Add plot title
    plt.ylabel("latitude") #Adjust the label of the y-axis
    plt.ticklabel_format(style='plain', axis='y') #repress the scientific notation
    plt.xlabel("log_error") #Adjust the label of the x-axis#plt.ylim(0,100) #Adjust the limits of the y-axis
    plt.ticklabel_format(style='plain', axis='x') #repress the scientific notation
    #plt.xlim(0,10) #Adjust the limits of the x-axis
    plt.tight_layout() #Adjust subplot params

    plt.subplot(133)
    sns.regplot(x="longitude",
                y="latitude", 
                data=df, 
                line_kws={'color': 'red'})
    plt.title("Longitude & Latitude") #Add plot title
    plt.ylabel("latitude") #Adjust the label of the y-axis
    plt.ticklabel_format(style='plain', axis='y') #repress the scientific notation
    plt.xlabel("longitude") #Adjust the label of the x-axis
    plt.ticklabel_format(style='plain', axis='x') #repress the scientific notation
    #plt.ylim(0,100) #Adjust the limits of the y-axis
    #plt.xlim(0,10) #Adjust the limits of the x-axis
    plt.tight_layout() #Adjust subplot params


def Q2_viz_1(df):
    plt.figure(figsize=(20,8))

    plt.subplot(131)
    sns.barplot(x ='county', 
                y ='log_error', 
                hue ='loc_clusters',
                palette='colorblind', 
                data = df)#, estimator=np.mean)#, ci=95)#, capsize=.2)
    plt.title("Location Clusters by county on logerror") #Add plot title
    plt.ylabel("logerror") #Adjust the label of the y-axis
    plt.xlabel("county") #Adjust the label of the x-axis
    
    plt.subplot(132)
    sns.barplot(x ='loc_clusters', 
                y ='log_error', 
                data = df)
    plt.title("Bar Plot: Log Error of Location Clusters") #Add plot title
    plt.ylabel("logerror") #Adjust the label of the y-axis
    plt.xlabel("Location Clusters") #Adjust the label of the x-axis
    
    plt.subplot(133)
    sns.stripplot(x = "loc_clusters",
                  y = "log_error", 
                  data = df) 
    plt.title("Strip Plot: Log Error of Location Clusters") #Add plot title
    plt.ylabel("logerror") #Adjust the label of the y-axis
    plt.xlabel("loc_clusters") #Adjust the label of the x-axis
   

def split_stats(df, train, validate, test):
    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')

def Q2_kruskal_test_1(df):
    alpha = 0.05

    group_list = [df[df.loc_clusters == x].log_error.to_numpy() for x in range(5)]
    t,p_val = stats.kruskal(group_list[0],group_list[1],group_list[2],group_list[3],group_list[4])
    
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    print('_____________________')  
    print(f't-stat {t.round(4)}')
    print(f'p-value {p_val.round(4)}')

def pearson_r(df, sample_1, sample_2):
    """
    """
    alpha = 0.05
    r, p_val = stats.pearsonr(df[sample_1], df[sample_2])
    
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    r= r.round(4)
    p_val = p_val.round(4)
    print('_____________________')  
    print(f'correlation {r}')
    print(f'p-value {p_val}')


################################### delinquency vs log error #####################################
def get_loliplot_delinquency(train):
    # create data frame for loliplot
    loli= pd.DataFrame(
        {'Tax Delinquency':['No Delinquency', 'Has Deliquency'],
         'Mean Log Error':[train[train.has_taxdelinquency==0].log_error.mean(),
                            train[train.has_taxdelinquency==1].log_error.mean()]
        })
    # set fig size
    fig, axes = plt.subplots(figsize=(6,5))
    # set font and style
    sns.set_theme('talk')
    sns.set_style('white')

    # using subplots() to draw vertical lines
    axes.vlines(loli['Tax Delinquency'], ymin=0, ymax=loli['Mean Log Error'],color = '#06C2AC',lw=4)


    # drawing the markers (circle)
    axes.plot(loli['Tax Delinquency'], loli['Mean Log Error'], "o",color ='#E79C66',markersize=13) 
    axes.set_ylim(0)
    axes.set_xlim(-1,2)

    # formatting axis and details 
    #plt.xlabel('')
    plt.ylabel('Mean Log Error', fontsize =20)
    plt.title('Log Error increases with tax delinquency',fontsize =20)
    #axes.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.xticks(loli['Tax Delinquency'],fontsize = 14)
    plt.yticks(fontsize = 15 )
    axes.set_yticks(ticks=[0,0.010, 0.020,0.030,0.040])
    axes.set_xticks(ticks=[0,1]);


################################# stat test
def get_ttest_delinquency(df):
    
    
    # create two independent sample groups of customers: has_taxdelinquency True (=1) and False (=0).
    subset_no_feature =df[df.has_taxdelinquency==0]
    subset_feature = df[df.has_taxdelinquency==1]

    # # stats Levene test - returns p value. small p-value means unequal variances
    stat, pval =stats.levene( subset_no_feature.log_error, subset_feature.log_error)


    # high p-value suggests that the populations have equal variances
    if pval < 0.05:
        variance = False
      
    else:
        variance = True
        

    # set alpha to 0.05
    alpha = 0.05

    # perform t-test
    t_stat, p_val = stats.ttest_ind(subset_no_feature.log_error, subset_feature.log_error,equal_var=variance,random_state=123)
    
    # print hypotheis status
    if p_val/2 < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    print('_____________________')  

    # round  and print results, divide p by 2  
    t_stat = t_stat.round(4)
    p_val = (p_val.round(4))/2
    print(f't-stat {t_stat}')
    print(f'p-value {p_val}')



################################################ home age vs log_error
def get_scatterplot_age(train):   
    sns.scatterplot(y='age', x='log_error',
                    data=train[train.age<= 81], color='#06C2AC')

    sns.scatterplot(y='age', x='log_error',
                   data=train[train.age> 81], 
                   color='#E79C66')


    plt.title("Does yearbuilt  make a diffirence? ")
    plt.show()


################################# stats test
def get_ttest_age(train):
    
    
    # create two independent sample group of customers: churn and not churn.
    subset_older =train[train.age> 81]
    subset_younger = train[train.age<= 81]

    # # stats Levene test - returns p value. small p-value means unequal variances
    stat, pval =stats.levene( subset_older.log_error, subset_younger.log_error)


    # high p-value suggests that the populations have equal variances
    if pval < 0.05:
        variance = False
      
    else:
        variance = True
        

    # set alpha to 0.05
    alpha = 0.05

    # perform t-test
    t_stat, p_val = stats.ttest_ind(subset_older.log_error, subset_younger.log_error,equal_var=variance,random_state=123)
    
    # print hypotheis status
    if p_val/2 < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')
    print('_____________________')  

    # round  and print results, divide p by 2  
    t_stat = t_stat.round(4)
    p_val = (p_val.round(4))/2
    print(f't-stat {t_stat}')
    print(f'p-value {p_val}')

############################################### group subclusters######################

def group_clusters(train):    
    plt.figure(figsize=(18, 6))

    # subplot #1
    plt.subplot(131)
    for cluster, subset in train.groupby('cluster_price_size'):
        plt.scatter(x=subset.sqft, y=subset.log_error, label='cluster' + str(cluster), alpha=.4, )
    plt.legend()
    plt.xlabel('Square Feet')
    plt.ylabel('Log Error')
    plt.title('Cluster:Price & Size')
    plt.legend(loc='lower right')
    
    # subplot #2
    plt.subplot(132)
    for cluster, subset in train.groupby('cluster_delinquency_value'):
        plt.scatter(x=subset.sqft, y=subset.log_error, label='cluster' + str(cluster), alpha=.4, )
    plt.legend()
    plt.xlabel('Square Feet')
    plt.legend(loc='lower right')
    plt.title('Cluster:Delinquency & home value')

    # subplot #3
    plt.subplot(133)
    for cluster, subset in train.groupby('loc_clusters'):
        plt.scatter(x=subset.sqft, y=subset.log_error, label='cluster' + str(cluster), alpha=.4, )
    plt.legend()
    plt.xlabel('Square Feet')
    plt.legend(loc='lower right')
    plt.title('Cluster:Location' );