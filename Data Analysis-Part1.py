#Data Analysis Phase
# Main aim is to understand more about the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_columns',None)
dataset = pd.read_csv('C:/Users/DELL/OneDrive/SYLLABUS/Data Science/Advance-House-Price-Prediction/GithubMaterial/train.csv')

#print the shape of the dataset
print(dataset.shape)
print(dataset.head())

# Find out how many missing values are there
# 1: make a list of the features which has more than one missing value in a col
features_with_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]

# Print the feature name and percentage of missing values
print("Missing values column:",len(features_with_na))
for feature in features_with_na:
    print(feature,np.round(dataset[feature].isnull().mean()*100,4),' % missing values')

missing_values = np.round(dataset[features_with_na].isnull().mean()*100,4)
missing_values = missing_values.sort_values(ascending = False)
plt.figure(figsize=(15,8))
sns.barplot(x=missing_values.index,y=missing_values.values,palette="viridis")
plt.title("Percentage of Missing Values by Column",fontsize=16)

plt.xlabel("Column Names",fontsize=14)
plt.ylabel("Percentage of Missing Values",fontsize=14)
plt.xticks(rotation=45,ha='right',fontsize=10)
# plt.show()

# Since there are many missing values we need to find the relationship between missing
#values and sales price

for feature in features_with_na:
    data = dataset.copy()
    data[feature] = np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
#    plt.show()

#Here With the relation between the missing values and the dependent variable is clearly visible So
#We need to replace these nan values with something meaningful which we will do in the Feature
# Engineering section


# Counting numerical features:
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes!='O'] # 'O' denotes the string or object data type
print("Number of numerical variables: ",len(numerical_features))
print(dataset[numerical_features].head())


# Temporal Variables(Eg: Datetime Variables)
# From the Dataset we have 4 year variables. We have extract information from the datetime
# variables like no of years or no of days. One example in this specific scenario can be difference in
# years between the year the house was built and the year the house was sold, We will be
# performing this analysis in the Feature Engineering which is the next video.

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
print(year_feature)

#Printing the year feature
for feature in year_feature:
    print(feature,dataset[feature].unique())


# Let's analyze the temporal datetime variables
# we will check weather there is a relation between year  the house is sold and salePrice

# Group by the year sold and compute median SalePrice
sale_price_year = dataset.groupby('YrSold')['SalePrice'].median()

# Line plot: Median SalePrice vs Year of Sale
plt.figure(figsize=(10, 6))
plt.plot(sale_price_year.index, sale_price_year.values, marker='o', linestyle='-', color='g', label='Median Sale Price')
plt.xlabel('Year Sold', fontsize=12)
plt.ylabel('Median House Price', fontsize=12)
plt.title('House Sale Price vs Year Sold', fontsize=14)
plt.grid()
plt.legend()
# plt.show()

# Now we will compare the difference between all years features with saleprice
for feature in year_feature:
    if(feature!='YrSold'):
        data = dataset.copy()
        data[feature] = data['YrSold']-data[feature]
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        # plt.show()



# Segregating discrete features
discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: ",len(discrete_features))
print(discrete_features)

# Finding relationship between discrete features and salePrice

for feature in discrete_features:
    data=dataset.copy()
    grouped_data= data.groupby(feature)['SalePrice'].median().reset_index()
    sns.barplot(data=grouped_data,x=feature,y='SalePrice',palette="viridis")
    # plt.show()


# Segregating the continuous features
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+['Id']]
print("Continuous feature count: ",len(continuous_features))
print(continuous_features)
for feature in continuous_features:
    data = dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

# Some features have gaussian distribution but others have skewed data, so we will do normalization (non gaussian to gaussian distribution)
# That will be helpful for linear model prediction

