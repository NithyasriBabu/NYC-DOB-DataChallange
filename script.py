#%% Importing library and data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import datetime

#%% Importing data with data types and default values

date_columns = ['Latest Action Date','Pre- Filing Date', 'Paid', 'Fully Paid', 
                'Assigned', 'Approved', 'Fully Permitted','DOBRunDate','SIGNOFF_DATE','SPECIAL_ACTION_DATE']

default_values = {
        'Existing Occupancy': '',
        'Landmarked': 'N',
        'Adult Estab': 'N',
        'Loft Board': 'N', 
        'City Owned': 'N', 
        'Little e': 'N',
        'PC Filed': 'N',
        'eFiling Filed': 'N',
        'Plumbing': 'N', 
        'Mechanical': 'N', 
        'Boiler': 'N',
        'Fuel Burning': 'N', 
        'Fuel Storage': 'N', 
        'Standpipe': 'N', 
        'Sprinkler': 'N', 
        'Fire Alarm': 'N',
        'Equipment': 'N', 
        'Fire Suppression': 'N', 
        'Curb Cut': 'N', 
        'Other': 'N',
        'Professional Cert': 'N',
        'Horizontal Enlrgmt': 'N', 
        'Vertical Enlrgmt': 'N',
        'Non-Profit': 'N',
        }

columns_types = {
        'Job #': str, 
        'Doc #': np.int32,
        'Borough': str,
        'House #': str,
        'Street Name': str,
        'Block': str,
        'Lot': str,
        'Bin #': np.int32,
        'Job Type': str, 
        'Job Status': str, 
        'Job Status Descrp': str,
        'Building Type': str, 
        'Community - Board': str, 
        'Cluster': str,
        'Landmarked': str, 
        'Adult Estab': str, 
        'Loft Board': str, 
        'City Owned': str, 
        'Little e': str,
        'PC Filed': str, 
        'eFiling Filed':str, 
        'Plumbing':str, 
        'Mechanical':str, 
        'Boiler':str,
        'Fuel Burning':str, 
        'Fuel Storage':str, 
        'Standpipe':str, 
        'Sprinkler':str, 
        'Fire Alarm':str,
        'Equipment':str, 
        'Fire Suppression':str, 
        'Curb Cut':str, 
        'Other':str,
        'Other Description':str, 
        'Applicant\'s First Name':str, 
        'Applicant\'s Last Name':str,
        'Applicant Professional Title':str, 
        'Applicant License #':str,
        'Professional Cert':str, 
        'Initial Cost': str,
        'Total Est. Fee': str, 
        'Fee Status':str, 
        'Existing Zoning Sqft': str,
        'Proposed Zoning Sqft': str, 
        'Horizontal Enlrgmt': str, 
        'Vertical Enlrgmt': str,
        'Enlargement SQ Footage': str, 
        'Street Frontage': str, 
        'ExistingNo. of Stories': np.int32,
        'Proposed No. of Stories': np.int32, 
        'Existing Height': np.int32, 
        'Proposed Height': np.int32,
        'Existing Dwelling Units': str, 
        'Proposed Dwelling Units': str,
        'Existing Occupancy': str, 
        'Proposed Occupancy': str, 
        'Site Fill': str, 
        'Zoning Dist1': str,
        'Zoning Dist2': str, 
        'Zoning Dist3': str, 
        'Special District 1': str,
        'Special District 2': str, 
        'Owner Type': str, 
        'Non-Profit': str, 
        'Owner\'s First Name': str,
        'Owner\'s Last Name': str ,
        'Owner\'s Business Name': str, 
        'Owner\'s House Number': str,
        'Owner\'sHouse Street Name': str, 
        'City ': str, 
        'State': str, 
        'Zip': str, 
        'Owner\'sPhone #': str,
        'Job Description': str, 
        'JOB_S1_NO': str,
        'TOTAL_CONSTRUCTION_FLOOR_AREA': str, 
        'WITHDRAWAL_FLAG': str,
        'SPECIAL_ACTION_STATUS':str, 
        'BUILDING_CLASS':str,
        'JOB_NO_GOOD_COUNT':str, 
        'GIS_LATITUDE':str, 
        'GIS_LONGITUDE':str,
        'GIS_COUNCIL_DISTRICT':str, 
        'GIS_CENSUS_TRACT':str, 
        'GIS_NTA_NAME':str,
        'GIS_BIN':str}

print('Reading Data from File')
data = pd.read_csv('data/DOB_Job_Application_Filings.csv',
                   dtype=columns_types, 
                   na_values=default_values,
                   na_filter=False)

#%% Pre Processing

# Converting required columns to proper datetime objects

print('Converting Dates')
data['Pre- Filing Date'] = pd.to_datetime(data['Pre- Filing Date'], format='%m/%d/%Y')
data['DOBRunDate'] = pd.to_datetime(data['DOBRunDate'], format='%m/%d/%Y %I:%M:%S %p')

#%% Filtering applications from 2013 to 2018

print('Extracting Prefiling year')

data['Pre-Filing_Year'] = data['Pre- Filing Date'].dt.year
year_filter = (data['Pre-Filing_Year'] >= 2013) & (data['Pre-Filing_Year'] <= 2018)

filtered_prefiling_2013_2018 = data.loc[year_filter]

#%%

print('Filtering data with Most Recent DOBRunDate')

fil_data = filtered_prefiling_2013_2018
JobIDsWithDoc1 = fil_data[fil_data['Doc #'] == 1]['Job #'].tolist()

fil_data = fil_data.loc[fil_data['Job #'].isin(JobIDsWithDoc1)]
                                 
idx = fil_data.groupby(['Job #'])['DOBRunDate'].transform(max) == fil_data['DOBRunDate']
fil_data = fil_data[idx]

idx = fil_data.groupby(['Job #'])['Doc #'].transform(max) == fil_data['Doc #']
fil_data = fil_data[idx]

#%% 1) Report the number of unique DOB job applications with a "Pre- Filing Date" in 2018

q1_data = fil_data[fil_data['Pre-Filing_Year'] == 2018]
print("Number of Unique Job Applications in 2018 is {}".format(q1_data.shape[0]))


#%% 2) What proportion of the job applications pertain to buildings with residential existing occupancy types in MANHATTEN?

q2_data = fil_data.loc[fil_data['Borough'].str.upper() == 'MANHATTAN']

res_codes = ['RES', 'R-']
search_string = '|'.join(res_codes)

q2_data['Existing Occupancy'].fillna('')
ResidentialManBuild = q2_data[q2_data['Existing Occupancy'].str.contains(search_string)].shape[0]
NumUniqueRecordsMan = q2_data.shape[0]

propotion = round(ResidentialManBuild / NumUniqueRecordsMan, 2)

print("Propotion of Job Applications in Manhattan that pertain to residential exsisting occupancy types {}".format(propotion))

#%% 3) What is the ratio of the highest to the second-highest value of these proportions?

boroughs = fil_data['Borough'].unique().tolist()

borough_prop=[]

for borough in boroughs:
    borough_data = fil_data.loc[fil_data['Borough'].str.upper() == borough]
    OwnedByCorpPart = borough_data.loc[borough_data['Owner Type'].isin(['CORPORATION','PARTNERSHIP'])]
    borough_prop.append(OwnedByCorpPart.shape[0]/borough_data.shape[0])
    
borough_prop = sorted(borough_prop,reverse=True)
ratio = round(borough_prop[0]/borough_prop[1],2)

print("Ratio of the highest to the second-highest value of these proportion of CORP/PART Owned buildings in each bouroughs is {}".format(ratio))

#%% 4) Ratio of the highest to the second-highest value of these New Buildings per sqMile of each bouroughs

boroughs = fil_data['Borough'].unique().tolist()

borough_permile_buildings=[]

Land_Area_perBorough_Google = {'MANHATTAN': 22.82, 'QUEENS': 108.1, 'BRONX': 42.47, 'BROOKLYN': 69.5, 'STATEN ISLAND': 58.69}

for borough in boroughs:
    filter_2018_perborough = (fil_data['Pre-Filing_Year'] == 2018) & (fil_data['Borough'] == borough) & (fil_data['Job Type'] == 'NB')
    borough_data = fil_data.loc[filter_2018_perborough]
    num_newBuildings = borough_data.shape[0]
    borough_area = Land_Area_perBorough_Google[borough]
    
    newBuilding_perSqMile = num_newBuildings/borough_area
    
    borough_permile_buildings.append(newBuilding_perSqMile)
    
borough_permile_buildings = sorted(borough_permile_buildings,reverse=True)
ratio = round(borough_permile_buildings[0]/borough_permile_buildings[1],2)

print("Ratio of the highest to the second-highest value of these New Building per sqMile of each bouroughs is {}".format(ratio))

#%% 5) What proportion of job applications involve an increase from the number of existing dwelling units to the number of proposed dwelling units
# residential Existing Occupancy and job type A1?

q5_data = fil_data.loc[fil_data['Job Type'] == 'A1']

residential_codes = ['RES', 'R-']
res_search_string = '|'.join(residential_codes)

q5_data['Existing Occupancy'].fillna('')
Res_Exsiting_JobA1 = q5_data[q5_data['Existing Occupancy'].str.contains(search_string)]

remove_empty_filter = (Res_Exsiting_JobA1['Proposed Dwelling Units'] != '') & (Res_Exsiting_JobA1['Existing Dwelling Units'] != '')
Res_Exsiting_JobA1 = Res_Exsiting_JobA1.loc[remove_empty_filter]

Res_Exsiting_JobA1 = Res_Exsiting_JobA1.astype({'Proposed Dwelling Units': 'int32', 'Existing Dwelling Units': 'int32'})

filter_Units_Increase = Res_Exsiting_JobA1['Proposed Dwelling Units'] > Res_Exsiting_JobA1['Existing Dwelling Units']
Num_Increase_Units = Res_Exsiting_JobA1[filter_Units_Increase]

Num_units_Increased = Num_Increase_Units.shape[0]
total_units_filter = Res_Exsiting_JobA1.shape[0]

ratio = round(Num_units_Increased/total_units_filter,2)
print("Proportion of job applications involving an increase in the number of existing dwelling units with number of proposed dwelling units for residential Existing Occupancy and job type A1 is {}".format(ratio))

#%% 6) Linear Regression for Number of Days of Approval in Brooklyn

q6_data = fil_data.loc[fil_data['Borough'].str.upper() == 'BROOKLYN']
q6_data = q6_data[q6_data['Fully Permitted'] != '']

q6_data['Fully Permitted'] = pd.to_datetime(q6_data['Fully Permitted'], format='%m/%d/%Y')

q6_data['Approval Time (Days)'] = (q6_data['Fully Permitted'] - q6_data['Pre- Filing Date']).dt.days
q6_data = q6_data[q6_data['Approval Time (Days)'] >= 0]


q6_reg_data = q6_data[['Approval Time (Days)','Pre-Filing_Year']]

#To check if the 2 values have a linear relationship
plt.scatter(q6_reg_data['Approval Time (Days)'],q6_reg_data['Pre-Filing_Year'])

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

X = pd.DataFrame(q6_reg_data['Approval Time (Days)'])
Y = pd.DataFrame(q6_reg_data['Pre-Filing_Year'])

model = LinearRegression()
scores = []

kfold = KFold(n_splits=3, shuffle=True)

for i, (train, test) in enumerate(kfold.split(X, Y)):
    model.fit(X.iloc[train,:], Y.iloc[train,:])
    score = model.score(X.iloc[test,:], Y.iloc[test,:])
    scores.append(score)

avg_r2 = round(sum(scores)/len(scores),4)

print("Average R2 Score of the linear regression of #Approval Time vs Pre_Filing_Year is {}".format(avg_r2))
     