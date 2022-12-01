######################### Display Imports #########################

def imports():
    """
    Prints a list of standard import functions than can be quickly copy pasted for use.
    """

    print("""
            # standardized modules
            import seaborn as sns
            import pandas as pd
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            import scipy.stats as stats
            from sklearn.model_selection import train_test_split
            
            # Decision Tree and Model Evaluation Imports
            from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
            
            #my modules
            import QMCBT_wrangle as w
            import QMCBT_explore_evaluate as ee
            import QMCBT_[00]quick_tips as tips
            import QMCBT_[01]acquire as acquire
            import QMCBT_[02]prepare as prepare
            import QMCBT_[03]explore as explore
            import QMCBT_[04]evaluate as evaluate""")

    
    
######################### TABLE OF CONTENTS #########################

def TOC():
    """
    Prints a Table of Contents for quick reference of what functions are available for use.
    """    
    
    print('imports')
    print('')
    print('JUPYTER MARK UP')
    print('* cell_color')
    print('')
    print('CHEAT SHEETS (cs_)')
    print('* cs_confusion_matrix')
    print('* cs_hypothesis')
    print('* cs_train_val_test')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')


    
######################### Tips and Tricks #########################

def tips():
    """
    Prints useful code tips.
    """    
    
    print('%who')
    print("df.describe(include='all')")
    print('')
    print('')
    print('')
    print('')
    

    
######################### JUPYTER Markup Colors #########################

def cell_color():
    """
    Prints a short list of Jupyter Workbook markup code tips and tricks.
    """
    
    print('BLUE: <div class="alert alert-info"></div>')
    print('GREEN: <div class="alert alert-success"></div>')
    print('YELLOW: <div class="alert alert-warning"></div>')
    print('RED: <div class="alert alert-danger"></div>')
   
    
    
######################### Confusion Matrix Cheat Sheet #########################

def cs_confusion_matrix():

    print(' **POSITIVE (+)** = insert Positive statement here  ')
    print(' **NEGATIVE (-)** = insert Negative statement here  ')  
    print() 
    print(' **RECALL**  ')  
    print(' TP / (TP + FN)  ')  
    print(' Use for less **Type II** errors when **FN** is worst outcome  ')  
    print(' Maximize for **RECALL** if Cost of **FN** > Cost of **FP**  ')  
    print() 
    print(' **PRECISION**  ')  
    print(' TP / (TP + FP)  ')  
    print(' Use for less **Type I** errors when **FP** is worst outcome  ')  
    print(' Maximize for **PRECISION** if Cost of **FP** > Cost of **FN**  ')  
    print() 
    print(' **ACCURACY**  ')  
    print(' (TP + TN)/(FP+FN+TP+TN)  ')  
    print(' prediction TRUE / total  ')  
    print(' Maximize for **ACCURACY** if neither **RECALL** or **PRECISION** outweigh eachother  ')  
    print()
    print(' * **Classification Confusion Matrix** (actual_col, prediction_row)(Positive_first, Negative_second)  ')
    print('                       +------------------------------------------+  ')
    print('                       | actual Positive (+) | actual Negative(-) |  ')
    print(' +---------------------+---------------------+--------------------+  ')
    print(' |  pred Positive (+)  |     TP              |     FP (Type I)    |  ')
    print(' +---------------------+---------------------+--------------------+  ')
    print(' |  pred Negative (-)  |     FN (Type II)    |     TN             |  ')
    print(' +---------------------+---------------------+--------------------+  ')
    print()
    print(' * <b>sklearn Confusion Matrix</b> (prediction_col, actual_row)(Negative_first, Positive_second)  ')
    print('                       +--------------------------------------+  ')
    print('                       | pred Negative(-) | pred Positive (+) |  ')
    print(' +---------------------+------------------+-------------------+  ')
    print(' | actual Negative (-) |        TN        |    FP (Type I)    |  ')
    print(' +---------------------+------------------+-------------------+  ')
    print(' | actual Positive (+) |   FN (Type II)   |         TP        |  ')  
    print(' +---------------------+------------------+-------------------+  ')
    print() 
    print(' **FP**: We **predicted** it was a **POSITIVE** when it was **actually** a **NEGATIVE**  ')  
    print(' *    FP = We **FALSE**LY predicted it was **POSITIVE**  ')  
    print(' * False = Our prediction was False, it was actually the opposite of our prediction  ')  
    print(' * Oops... **TYPE I** error!  ')  
    print() 
    print(' **FN**: We **predicted** it was a **NEGATIVE** when it was **actually** a **POSITIVE**  ')  
    print(' *    FN = We **FALSE**LY predicted it was **NEGATIVE**  ')  
    print(' * False = Our prediction was False, it was actually the opposite of our prediction  ')  
    print(' * Oops... **TYPE II** error!  ')  
    print() 
    print(' **TP**: We **predicted** it was a **POSITIVE** and it was **actually** a **POSITIVE**  ')  
    print(' *   TP = We **TRUE**LY predicted it was **POSITIVE**  ')  
    print(' * True = Our prediction was True, it was actually the same as our prediction  ')  
    print() 
    print(' **TN**: We **predicted** it was a **NEGATIVE** and it was **actually** a **NEGATIVE**  ')  
    print(' *   TN = We **TRUE**LY predicted it was **NEGATIVE**  ')  
    print(' * True = Our prediction was True, it was actually the same as our prediction  ') 

######################### Hypothesis Cheat Sheet #########################

def cs_hypothesis():

    print(' **A. Set Hypothesis**  ')  
    print('  ') 
    print(' * One Tail (```<= | >```) or Two Tails (```== | !=```)?\  ')
    print('  **two_tail (gender, been_manager)**  ')  
    print('  ') 
    print('  ') 
    print(' * One Sample or Two Samples?\  ')
    print('  **two_sample (gender, been_manager)**  ')  
    print('  ') 
    print('  ') 
    print(' * Continuous or Discreat?\  ')
    print('  **Discreat (gender) vs Discreat (been_manager) = $Chi^2$**  ')  
    print('      * T-Test = ```Discreat``` vs ```Continuous```  ')
    print('      * Pearson’s = ```Continuous``` vs ```Continuous``` (linear)  ')
    print('      * $Chi^2$ = ```Discreat``` vs ```Discreat```  ')
    print('  ') 
    print('  ') 
    print(' * $𝐻_0$: The opposite of what I am trying to prove\  ')  
    print('  **$H_{0}$: The employee gender is **NOT** ```dependent``` on whether the employee has been a manager**\  ')
    print('  ```employees.gender ``` != ```employees.been_manager```  ')  
    print('  ') 
    print('  ') 
    print(' * $𝐻_𝑎$: What am I trying to prove\  ')  
    print('  **$H_{a}$: The employee gender is ```dependent``` on whether the employee has been a manager**\  ')  
    print('  ```employees.gender ``` == ```employees.been_manager```  ')

######################### Train, Validate, Test SPLIT Cheat Sheet #########################

def cs_train_val_test():

    print(' _______________________________________________________________  ')
    print(' |                              DF                             |  ')
    print(' |-------------------+-------------------+---------------------|  ')
    print(' |       Train       |       Validate    |          Test       |  ')
    print(' +-------------------+-------------------+-----------+---------+  ')
    print(' | x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |  ')
    print(' +-------------------------------------------------------------+  ')
    print('') 
    print(' * 1. tree_1 = DecisionTreeClassifier(max_depth = 5)  ')
    print(' * 2. tree_1.fit(x_train, y_train)  ')
    print(' * 3. predictions = tree_1.predict(x_train)  ')
    print(' * 4. pd.crosstab(y_train, predictions)  ')
    print(' * 5. val_predictions = tree_1.predict(x_val)  ')
    print(' * 6. pd.crosstab(y_val, val_predictions)  ')

######################### Display Confusion Matrix Graphic #########################

# import image module
from IPython.display import Image

# ------ Display confusion matrix function ------
def display_confusion_matrix_graphic():

# get the image
    return Image(url="confusion_matrix.png", width=920, height=474)

######################### MISC WORKING NOTES #########################

"""
# Temporarily set display to max for full visibility of DataFrame
import pandas as pd
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3):
    display('INSERT_FUNCTION_HERE')
    

# show df based on results of criteria
rslt_df = df[df.transactiondate > '2017-12-31']
rslt_df


# show df based on results of criteria
df.loc[df['calculatedfinishedsquarefeet'] == 1].T

# Left align tables in Jupyter markdown
# run this code at beginning of notebook
%%html
<style>
table {float:left}
</style>






"""