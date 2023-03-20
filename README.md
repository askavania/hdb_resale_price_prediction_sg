# Singapore Housing Data & Kaggle Challenge

# Table of Contents:
1. [Problem Statement](#Problem_Statement)
2. [Data Import & Cleaning](#Data_Import_&_Cleaning)
3. [Exploratory Data Analysis](#Basic_Exploratory_Data_Analysis)
4. [Preprocessing, Modeling & Evaluation](#Preprocessing_&_Modeling)
    - [Baseline Model](#Creating_a_baseline_model)
    - [Feature Engineering](#Feature_Engineering)
    - [Training a new model](#Training_a_new_model)
    - [Regularization for a better model](#Regularization_for_a_better_model)
5. [Model Conclusion](#Model_Conclusion)
6. [Prediction using Chosen Model](#Prediction_using_Chosen_Model)

# Problem Statement:

We will be using Singapore public housing data to create a regression model that predicts the price of Housing Development Board (HDB) flats in Singapore. This model will predict the price of a house at sale. We will be evaluating the models using their RMSE values. 

##### About the Datasets:

Our housing data is sourced from a Kaggle Competition and comprise of;

1. [Train Dataset for creation of model](https://git.generalassemb.ly/benedictyong/project/blob/master/Project%202/datasets/train.csv)
2. [Test Dataset for Prediction](https://git.generalassemb.ly/benedictyong/project/blob/master/Project%202/datasets/test.csv)

Our output data;
1. [Predicted Results](https://git.generalassemb.ly/benedictyong/project/blob/master/Project%202/datasets/sub_reg7.csv)


### Data Dictionary

This Dataset is an exceptionally detailed one with over 70 columns of different features relating to houses. Below are their features and corresponding description;

|Feature|Description|
|---|---|
|resale_price|the property's sale price in Singapore dollars - ***Prediction target variable***|
|Tranc_YearMonth| year and month of the resale transaction, e.g. 2015-02|
|town| HDB township where the flat is located, e.g. BUKIT MERAH|
|flat_type| type of the resale flat unit, e.g. 3 ROOM|
|block| block number of the resale flat, e.g. 454|
|street_name| street name where the resale flat resides, e.g. TAMPINES ST 42|
|storey_range| floor level (range) of the resale flat unit, e.g. 07 TO 09|
floor_area_sqm| floor area of the resale flat unit in square metres|
flat_model| HDB model of the resale flat, e.g. Multi Generation|
|lease_commence_date| commencement year of the flat unit's 99-year lease|
|Tranc_Year| year of resale transaction|
|Tranc_Month| month of resale transaction|
|mid_storey| median value of storey_range|
|lower| lower value of storey_range|
|upper| upper value of storey_range|
|mid| middle value of storey_range|
|full_flat_type| combination of flat_type and flat_model|
|address| combination of block and street_name|
|floor_area_sqft| floor area of the resale flat unit in square feet|
precinct_pavilion|hdb_age| number of years from lease_commence_date to present year|
|max_floor_lvl| highest floor of the resale flat|
|year_completed| year which construction was completed for resale flat|
|residential| boolean value if resale flat has residential units in the same block|
|commercial| boolean value if resale flat has commercial units in the same block|
|market_hawker| boolean value if resale flat has a market or hawker centre in the same block|
|multistorey_carpark| boolean value if resale flat has a multistorey carpark in the same block|
|precinct_pavilion| boolean value if resale flat has a pavilion in the same block|
|total_dwelling_units| total number of residential dwelling units in the resale flat|
|1room_sold| number of 1-room residential units in the resale flat|
|2room_sold| number of 2-room residential units in the resale flat|
|3room_sold| number of 3-room residential units in the resale flat|
|4room_sold| number of 4-room residential units in the resale flat|
|5room_sold| number of 5-room residential units in the resale flat|
|exec_sold| number of executive type residential units in the resale flat block|
|multigen_sold| number of multi-generational type residential units in the resale flat block|
|studio_apartment_sold| number of studio apartment type residential units in the resale flat block|
|1room_rental| number of 1-room rental residential units in the resale flat block|
|2room_rental| number of 2-room rental residential units in the resale flat block|
|3room_rental| number of 3-room rental residential units in the resale flat block|
|other_room_rental| number of "other" type rental residential units in the resale flat block|
|postal| postal code of the resale flat block|0|
|Latitude| Latitude based on postal code|
|Longitude| Longitude based on postal code|
|planning_area| Government planning area that the flat is located|
|Mall_Nearest_Distance| distance (in metres) to the nearest mall|
|Mall_Within_500m| boolean value if there is a mall within 500 metres|
|Mall_Within_1km| boolean value if there is a mall within 1 kilometre|
|Mall_Within_2km| boolean value if there is a mall within 2 kilometres|
|Hawker_Nearest_Distance| distance (in metres) to the nearest hawker centre|
|Hawker_Within_500m| boolean value if there is a hawker centre within 500 metres|
|Hawker_Within_1km| boolean value if there is a hawker centre within 1 kilometre|
|Hawker_Within_2km| boolean value if there is a hawker centre within 2 kilometres|
|hawker_food_stalls| number of hawker food stalls in the nearest hawker centre|
|hawker_market_stalls| number of hawker and market stalls in the nearest hawker centre|
|mrt_nearest_distance| distance (in metres) to the nearest MRT station|
|mrt_name| name of the nearest MRT station|
|bus_interchange| boolean value if the nearest MRT station is also a bus interchange|
|mrt_interchange| boolean value if the nearest MRT station is a train interchange station|
|mrt_latitude| latitude (in decimal degrees) of the the nearest MRT station|
|mrt_longitude| longitude (in decimal degrees) of the nearest MRT station|
|bus_stop_nearest_distance| distance (in metres) to the nearest bus stop|
|bus_stop_name| name of the nearest bus stop|
|bus_stop_latitude| latitude (in decimal degrees) of the the nearest bus stop|
|bus_stop_longitude| longitude (in decimal degrees) of the nearest bus stop|
|pri_sch_nearest_distance| distance (in metres) to the nearest primary school|
|pri_sch_name| name of the nearest primary school|
|vacancy| number of vacancies in the nearest primary school|
|pri_sch_affiliation| boolean value if the nearest primary school has a secondary school affiliation|
|pri_sch_latitude| latitude (in decimal degrees) of the the nearest primary school|
|pri_sch_longitude| longitude (in decimal degrees) of the nearest primary school|
|sec_sch_nearest_dist| distance (in metres) to the nearest secondary school|
|sec_sch_name| name of the nearest secondary school|
|cutoff_point| PSLE cutoff point of the nearest secondary school|
|affiliation| boolean value if the nearest secondary school has an primary school affiliation|
|sec_sch_latitude| latitude (in decimal degrees) of the the nearest secondary school|
|sec_sch_longitude| longitude (in decimal degrees) of the nearest secondary school|

# Model Conclusion
The ridge regression model of the feature engineered dataset '(ridge_cv1) performs better than the first dataset(ridge_cv) with accuracy scores of 90.83% vs 89.52%, and RMSE of 42235.186436377015 vs 46126.23167497795 respectively.
We will use the model 'ridge_cv1' on our test dataset to predict the resale prices. 
