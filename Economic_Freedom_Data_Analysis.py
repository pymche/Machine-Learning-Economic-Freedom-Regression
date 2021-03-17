#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="whitegrid")
from continent import continent

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics


# # 0. Loading Data

df = pd.read_excel('data.xlsx', sheet_name='EFW Panel Data 2019 Report')


df


# # 1. Data Cleaning

df.drop(columns=['Unnamed: 0', 'Unnamed: 2'], inplace=True)


df.columns = df.loc[1]


df.drop(index=[0, 1], inplace=True)


new_col = []
for x in df.columns[:3]:
    new_col.append(x)
for x in df.columns[3:]:
    x = x.split(' ')[2:]
    x = ' '.join(x)
    new_col.append(x)
print(new_col)

df.columns = ['Year', 'Countries', 'EFW', 'Size of Government', 'Legal System & Property Rights', 'Sound Money', 'Freedom to trade internationally', 'Regulation']


df.isna().sum()


df.dropna(axis='index', how='any', inplace=True)
df = df.reset_index()
df.drop(columns='index', inplace=True)
df


# Counting number of countries

len(df['Countries'].unique())


# ### Adding new column - Continent

df['Continent']=""
for length in range(len(df['Countries'])):

    for i in continent:
        if df['Countries'][length] in i['Country_Name']:
            df['Continent'][length]=i['Continent_Name']

# print(df['Continent'])


print(df['Continent'].value_counts())


# look at countries of 'empty' continent, but not many of them so ignore

print(df['Continent'].unique())
go = df['Continent'] == ''
list = df['Countries'].loc[go].unique()

for i in list:
    print(i)


# change order of columns of dataframe

# cols = list(df.columns.values)
# print(cols)

df=df[['Year', 'Countries', 'Continent', 'EFW', 'Size of Government', 'Legal System & Property Rights', 'Sound Money', 'Freedom to trade internationally', 'Regulation']]
df.columns


df.info()


# ### Quick look at Descriptive Statistics

df['Year'] = df['Year'].astype(int)
df[df.columns[3:]] = df[df.columns[3:]].astype(float)
df.info()


stats = []
for x in df.columns[3:]:
    stat = {}
    stat[' '] = x
    stat['Mean'] = df[x].mean()
    stat['Standard Dev'] = df[x].std()
    stat['3rd Quartile'] = df[x].quantile(0.75)
    stat['1st Quartile'] = df[x].quantile(0.25)
    stat['Interquartile Range'] = stat['3rd Quartile'] - stat['1st Quartile']
    stats.append(stat)
stats_df = pd.DataFrame(stats).set_index(' ')
stats_df


# ### Distribution of Attributes

plt.figure(figsize=(15,8))
ax = sns.boxplot(data=df[df.columns[3:]], width=0.5)


df.set_index(["Year", "Continent"], inplace=True)

# Need to sort before we are able to access the values via .loc[x, y]
df.sort_index(inplace=True)


# ### Finalized data format

df


df.loc[2000, 'Europe']['EFW'].mean()
# df.xs((1970, 'Europe'), level=('Year', 'Continent'))


car = df.index.get_level_values('Continent').unique()
car[2]


# ### Creating sub-dataframes of each attribute by continent (time series) for in-depth analysis

df.columns[1:]


# Create a dictionary of dataframes with the yearly average of each attribute (columns) grouped by CONTINENTS

cols = df.columns[1:]

# for c in cols:
#     print(c)
#     values = df[c].unique()
#     print(values)

# create a dictionary for each column by combining lists and names
lists = [[]for c in cols]
names = []
for name in cols:
    name = name.split(' ')[0]+'_dict'
    names.append(name)
# print(names)

# Dictionary no.1 before converting data to dataframe
cont_dict = {name: list for name, list in zip (names, lists)}
# print(cont_dict)
    
# fill empty lists with year average values of each column of each continent
continents = df.index.get_level_values('Continent').unique()
years = [x for x in range(2000, 2018)]

for c in cols:
#     print(c)
    for continent in continents:
        cont={}
        cont['Continent'] = continent
#         print(cont['Continent'])
        if continent != '':
            for year in years:
                cont['Year'] = year
                cont['Mean'] = df.loc[year, continent][c].mean()
#                 print('Mean of {} of {} in year {}: {}'.format(c, continent, year, df.loc[year, continent][c].mean()))
                c_tmp = c.split(' ')[0]+'_dict'
                cont_dict[c_tmp].append(cont.copy())

# print(cont_dict)


# Dictionary no.2 after converting dictionaries inside into dataframes
attributes = {}
for attribute in cont_dict:
    print(attribute)
    attributes[attribute] = pd.DataFrame(cont_dict[attribute])
#     attributes[attribute] = pd.DataFrame(cont_dict[attribute]).set_index(['Continent', 'Year']).sort_index()


attributes['EFW_dict']


# # 2. Exploratory Data Analysis

# ### Top and bottom 20 countries in terms of EFW in 2017
# 
# In 2017, Europe dominates the first twenty countries that have the highest EFW scores in the world, with nine countries from Europe,  followed by five from Asia. Fourteen out of the twenty countries with the lowest EFW scores are in Africa. Interestingly, four of the bottom group are also from Asia. We will have a closer look at the economies of Asia later on, but for now we compare the composition of European and African economies to see how they determine the EFW scores.

df_country = df.loc[2017].sort_values(by='EFW', ascending=False)
df_head = df_country.head(20)
df_tail = df_country.tail(20)

print(df_head[['Countries', 'EFW']])
print(df_head.index.get_level_values('Continent').value_counts())

df_head_count = df_head.index.get_level_values('Continent').value_counts()
ax = df_head_count.plot.barh(x='lab', y='val', figsize=(10,7))
ax.set_title('20 Countries with highest EFW - Continent Count')
ax.set_ylabel('Continent')
ax.set_xlabel('Number of Continent')


print(df_tail[['Countries', 'EFW']])
print(df_tail.index.get_level_values('Continent').value_counts())
df_tail_count = df_tail.index.get_level_values('Continent').value_counts()
ax = df_tail_count.plot.barh(x='lab', y='val', figsize=(10,7))
ax.set_title('20 Countries with lowest EFW - Continent Count')
ax.set_ylabel('Continent')
ax.set_xlabel('Number of Continent')


# ### Exploring correlation between variables
# 
# According to the correlation table below, Freedom to Trade Internationally has the highest correlation with EFW among other features, closely followed by Sound Money and Regulation. A graph representing each of these three features will be presented below. It is worth noting that correlation between features is generally low, hovering between 0.5 and 0.6. Size of the Government has relatively low correlation with all other features,

corr_df = pd.DataFrame(df.iloc[:, 1:]).astype(float)
corr_df.corr()


# ### Closer look into pairings - EFW

sns.jointplot(x="EFW", y="Freedom to trade internationally", data=df, kind="hex", height=7);


sns.jointplot(x="EFW", y="Sound Money", data=df, kind="hex", height=7);


sns.jointplot(x="EFW", y="Regulation", data=df, kind="hex", height=7);


# Interestingly, 'Size of Government' has a relatively low correlation with all other attributes

sns.pairplot(df)


# ### Time Series Analysis

# The following figures represent the changes of the mean value of each feature overtime from year 2000 to 2017.
# 
# South America is the only continent that saw an overall decrease in EFW score, while both Asia and Africa experienced a significant growth in EFW score over the two decades. Size of Government has been extremely volatile among all continents, with the exception of Africa, which has remained relatively stable at around 6 points over the years. Legal System and Property Rights is the most consistently stable feature among all continents over the years. Although it is worth noting that Europe saw a slight dip while Africa has experienced growth in this area. Europe, Asia and Africa saw an improvement in Sound Money. There is a decreasing trend in all continents in Freedom to Trade Internationally, albeit at a very insignificant rate.
# 
# With the highest EFW score, Europe has a relatively small size of government, a robust legal system with property rights, an abundance of sound money and freedom to trade internationally, as well as a moderate level of regulation.
# 
# As a developing continent, Africa has experienced a significant improvement in all areas of economy. Although it still ranked the lowest in said areas, it is without doubt that at the rate that it is growing, it will soon overtake other continents as a powerful economy.

sig_cols = ['EFW_dict', 'Size_dict', 'Legal_dict', 'Sound_dict', 'Freedom_dict', 'Regulation_dict']
sig_cols_val = ['EFW', 'Size of Government', 'Legal System & Property Rights',
       'Sound Money', 'Freedom to trade internationally', 'Regulation']
sig_dict = {name: val for name, val in zip (sig_cols, sig_cols_val)}

for dict in ['EFW_dict', 'Size_dict', 'Legal_dict', 'Sound_dict', 'Freedom_dict', 'Regulation_dict']:
    data = attributes[dict]
    graph = sns.relplot(x="Year", y="Mean", ci=None, kind="line", hue = "Continent", data=data, height=5, aspect=11.7/8.27)
    graph.fig.suptitle(sig_dict[dict])
    graph.fig.subplots_adjust(top=.9)


# ### A Closer Look at Asia

# As noted in previous sections, Asia has countries that score both the highest and lowest scores of EFW. Why is that the case? Let's have a look at the top and the bottom 5 Asian economies in terms of EFW. The following table shows that 4 out of 5 top countries are in Asia, whereas the bottom 5 countries are mostly in the Middle East. It is known that East Asia has experienced drastic growth in the late 20th century during the East Asian Miracle, and has remained strong economies since then. On the contrary, the Middle East has been politically and economically challenged by numerous unrests and upheavels in recent years. This explains the geographical disparity in economic performance, hence the big gap in EFW despite being in the same continent.

asia_df = df.xs(('Asia', 2017), level=('Continent', 'Year')).sort_values(by='EFW', ascending=False)
print(asia_df.head(5)['Countries'])
print(asia_df.tail(5)['Countries'])


# Nevertheless, as shown in the graph below, economies at the bottom group in terms of EFW scores have seen a significant increase in economic freedom since the 1980s. There is a convergence in EFW scores between the countries with the highest and lowest economic freedom.

asia_df2 = df.xs(('Asia'), level=('Continent')).sort_values(by='EFW', ascending=False).reset_index()
# asia_df2 = asia_df2[asia_df2['Year'] >= 2000]

ax = sns.relplot(x="Year", y="EFW", hue="Countries",
                 data=asia_df2, kind="line")
ax.fig.set_size_inches(10,10)


# In fact, most continents apart from South America are experiencing a convergence in EFW scores, as the disparities of economic freedom within each continent are shrinking overtime.

dum = []
for cont in df.index.get_level_values('Continent').unique():
    if cont != "":
        for year in range(2000, 2018):
            tmp = {}
            std = df.xs((year, cont), level=('Year', 'Continent'))['EFW'].std()
            tmp['Continent'] = cont
            tmp['Year'] = year
            tmp['std'] = std
            dum.append(tmp)
        
p_df = pd.DataFrame(dum)
# print(p_df)

p_df.set_index('Year', inplace=True)
p_df.groupby('Continent')['std'].plot(legend=True, figsize=[10, 10])


# # 3. Predictive Analysis - Linear Regression

# To investigate the impacts of the following features on EFW - Size of Government, Legal System & Property Rights, Sound Money, Freedom to trade internationally and Regulation, linear regression models are run with the data. Multiple models will be used, and the model with the best accuracy can help determine how much each feature can influence economic freedom, and provide pointers to policy-makers on which area to direct resources into.

regdf = df.reset_index()[df.columns[1:]]
regdf


# ## Model 1: OLS

# Train Model

x = np.array(regdf.drop(['EFW'], 1))
y = np.array(regdf['EFW'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# Fit data into model and check accuracy

linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test) 
print(acc)


# Printing coefficients

X = regdf.drop(['EFW'], 1)
coeff_df = pd.DataFrame(linear.coef_, X.columns, columns=['Coefficients'])
coeff_df


# predictions

predictions = linear.predict(x_test) 

# for x in range(len(predictions)):
#     print(f'Predicted values: {predictions[x]}, True value: {y_test[x]}')


# Predicted VS True value

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## 2. Ridge

# Ridge

x = np.array(regdf.drop(['EFW'], 1))
y = np.array(regdf['EFW'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

ridge = linear_model.Ridge(alpha=.5)
ridge.fit(x, y)

acc = linear.score(x_test, y_test) 
print(acc)


# Printing coefficients

X = regdf.drop(['EFW'], 1)
coeff_df = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficients'])
coeff_df


predictions = ridge.predict(x_test) 

# for x in range(len(predictions)):
#     print(f'Predicted values: {predictions[x]}, True value: {y_test[x]}')


# Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## 3. Lasso

x = np.array(regdf.drop(['EFW'], 1))
y = np.array(regdf['EFW'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

lasso = linear_model.Lasso(alpha=.5)
lasso.fit(x, y)

acc = lasso.score(x_test, y_test) 
print(acc)


# Printing coefficients

X = regdf.drop(['EFW'], 1)
coeff_df = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficients'])
coeff_df


predictions = lasso.predict(x_test) 

# for x in range(len(predictions)):
#     print(f'Predicted value: {predictions[x]}, True value: {y_test[x]}')


# Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## 4. Elastic Net

x = np.array(regdf.drop(['EFW'], 1))
y = np.array(regdf['EFW'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

elastic = linear_model.ElasticNet(alpha=.5)
elastic.fit(x, y)

acc = elastic.score(x_test, y_test) 
print(acc)


# Printing coefficients

X = regdf.drop(['EFW'], 1)
coeff_df = pd.DataFrame(elastic.coef_, X.columns, columns=['Coefficients'])
coeff_df


predictions = lasso.predict(x_test) 

# for x in range(len(predictions)):
#     print(f'Predicted value: {predictions[x]}, True value: {y_test[x]}')


# Calculating the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ### Model Evaluation and Analysis

# Amongst the four models above, Ridge regression performs with the best accuracy, while Lasso regression returns results with the least precision. Considering that the variables are not highly correlated with one another (as discussed in Exploratory Data Analysis), there is no need for feature selection, which Lasso normally does. Ordinary Least Square performs only marginally worse than Ridge Regression. Both return similar prediction, and suggest that each of the variable has a coefficient of around 0.19 or 0.2. In other words, an increase in 1 point of each variable, say Sound Money, will raise the EFW score by 0.2 points. Considering there are five features in total, their effects on the EFW score are evenly distributed. To rank the importance of each variable according to their miniscule differences, Legal System & Property Rights ranks the highest in influencing EFW, followed closely by Regulation, then Size of Government, and finally Sound Money, as well as Freedom to Trade Internationally.
