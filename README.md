# <b><i>Final Report Team Zillow Project</i></b>

## Group 4
* Justin Evans
* Yvette Ibarra

# Project Overview:
What is driving the errors in the Zestimates?

This team has been tasked to collect, clean and alayze Zillow data from 2017 in order to improve a previous prediction model that was designed to predict the Home Sale Value for Single Family Properties based on available realestate data.

# Project Goal:
* Use clusters to assist in our exploration, understanding, and modeling of the Zillow data, with a target variable of logerror for this regression project.
* Discover key attributes that drive error in Zestimate logerror.
* Use those attributes to develop a machine learning model to predict impact on logerror.

# Reproduction of this Data:
* Can be accomplished using a local env.py containing user, password, host information for access to the Codeup SQL database server.
* All other step by step instructions can be found by reading the below Jupyter Notebook files located in our Codeup-Justin-Evans-Yvette-Ibarra github repository.
* Final_Report_Zillow_Team_Project.ipynb
* wrangle.py
* explore.py
* model.py
* Justin_workbook.ipynb
* Yvette_workbook_1.ipynb
* Yvette_workbook_2.ipynb

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


# Data Dictionary:
<div class="alert alert-success">

    
## Continuous Categorical Counts
|Feature    |Description|
|:----------|:-----------------|
|parcelid|Unique Property Index| 
|bedrooms|Number of bedrooms in home|
|bathrooms|Number of bathrooms in home including fractional bathrooms| 
|calculatedbathnbr|Continuous float64 count of Bathrooms (including half and 3/4 baths)| 
|fullbathcnt|Count of only Full Bathrooms (no half or 3/4 baths)|
|age|The age of the home in 2017| 
|yearbuilt|The Year the principal residence was built| 

## Categorical Binary
|Feature    |Description           |
|:----------|:-----------------|
|has_basement|Basement on property (if any = 1)| 
|has_deck|Deck on property (if any = 1)| 
|has_fireplace|Fireplace on property (if any = 1)| 
|has_garage|Garage on property (if any = 1)| 
|has_hottuborspa|Hot Tub or Spa on property (if any = 1)| 
|has_pool|Pool on property (if any = 1)| 
|optional_features|Property has at least one optional feature listed above (if any = 1)| 
|has_tax_delinquency|Property has had Tax Delinquncy (if any = 1)| 

## Location
|Feature    |Description    |
|:----------|:-----------------|
|fips|Federal Information Processing Standards (FIPS), now known as Federal Information Processing Series, are numeric codes assigned by the National Institute of Standards and Technology (NIST). Typically, FIPS codes deal with US states and counties. US states are identified by a 2-digit number, while US counties are identified by a 3-digit number. For example, a FIPS code of 06111, represents California -06 and Ventura County -111.|
|state|This is the two letter abbreviation for the State as defined by the FIPS code| 
|county|FIPS code for california counties|
|la_county|fips: 6037; Categorical Binary Feature for Los Angeles County (if True = 1)| 
|orange_county|fips: 6059; Categorical Binary Feature for Orange County (if True = 1)| 
|ventura_county|fips: 6111; Categorical Binary Feature for Los Angeles County (if True = 1)|
|longitude|Longitude is a measurement of location east or west of the prime meridian at Greenwich, London, England, the specially designated imaginary north-south line that passes through both geographic poles and Greenwich. Longitude is measured 180° both east and west of the prime meridian.| 
|latitude|Latitude is a measurement on a globe or map of location north or south of the Equator. Technically, there are different kinds of latitude, which are geocentric, astronomical, and geographic (or geodetic), but there are only minor differences between them.|
|zipcode|A group of five or nine numbers that are added to a postal address to assist the sorting of mail.| 
|regionidcounty|Location code that identifies the Region and County of the property within the state| 
|rawcensustractandblock|| 
|censustractandblock|Census tracts are small, relatively permanent geographic entities within counties and Block numbering areas (BNAs) are geographic entities similar to census tracts, and delineated in counties (or the statistical equivalents of counties)
without census tracts.| 

## Size
|Feature    |Description           |
|:----------|:-----------------|
|sqft|Calculated total finished living area of the home|
|lotsizesquarefeet|| 

## Value
|Feature    |Description           |
|:----------|:-----------------|
|tax_value_bldg|The total tax assessed value of the structure|
|tax_value|The total tax assessed value of the parcel| 
|tax_value_land|The total tax assessed value of the land|
|taxamount|The total tax fee to be collected on the parcel| 

## Target
|Feature    |Description           |
|:----------|:-----------------|
|log_error|This is the logerror of the Zillow Zestimate|

## Clusters
|Feature    |Description          |
|:----------|:-----------------|
|loc_clusters|Created using 'longitude', 'latitude', 'age' with n_clusters = 4|
|cluster_price_size|Created using 'taxamount', 'sqft', 'lot_sqft' with n_clusters = 4|
|cluster_delinquency_value|Created using ‘tax_value’, ‘sqft’,‘has_taxdelinquency’ with n_clusters = 4|



# Takeaways and Conclusions
* 
* 
* 
* 

# Recommendations
..........

