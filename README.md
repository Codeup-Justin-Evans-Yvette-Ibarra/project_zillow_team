# Final Report Team Zillow Project

## Group 4
* Justin Evans
* Yvette Ibarra

# Project Overview
Zillow’s Zestimate is a home valuation in the real estate industry and created to give consumers as much information as possible about homes and the housing market. We will be looking into the log error of Zillow data from 2017 of single family properties to evaluate and analysis the drivers of log error.

# Project Goal
* Discover key attributes that drive error in Zestimate .

* Use those attributes to develop a machine learning model to predict log error.



# Initial Thoughts
Our initial thoughts is that outliers, age, and L.A. County are drivers of errors in Zestimate.

# The Plan
* Acquire data from Codeup database
* Prepare data
* Explore data in search of drivers of home_value
    * Answer the following initial question:
        * 
        * 
        * 
        * 

* Develop a Model to predict error in zestimate.
    * Use drivers identified in explore to build predictive models of error using...
    * Evaluate models on train and validate data using RMSE (Root mean square Error)
    * Select the best model based on the least RMSE
    * Evaluate the best model on test data
* Draw conclusions


# Data Dictionary

| Target            | Description|
| :--------------: | :-------------|
| log_error | Zestimate value subract the actual value|

 Feature          | Description|
| :---------------: | :---------------------------------- |
| home_value | The total tax assessed value of the parcel  |
| squarefeet:  | Calculated total finished living area of the home |
| bathrooms:   |  Number of bathrooms in home including fractional bathrooms |
| bedrooms: | Number of bedrooms in home  |
| yearbuilt:  |  The Year the principal residence was built   |
| fireplace: | fireplace on property (if any) |
| deck:  | deck on property (if any) |
| pool:  | pool on property (if any) |
| garage: | garage on property (if any) |
| county: | FIPS code for californian counties: 6111 Ventura County, 6059  Orange County, 6037 Los Angeles County |
| home_age: | The age of the home in 2017 |
| optional_features: | If a home has any of the follwing: fireplace, deck, pool, garage it is noted as 1 |
| additional features: | Encoded and values for categorical data |

# Steps to Reproduce
1. Clone this repository
2. Get Zillow data from Codeup Database:
    * Must have access to Codeup Database
    * Save a copy env.py file containing Codeup: hostname, username and password
    * Save file in cloned repository
3.Run notebook

# Takeaways and Conclusions
* 
* 
* 
* 

# Recommendations
..........

