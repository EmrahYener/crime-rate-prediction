#!/usr/bin/env python
# coding: utf-8

# # STEP #0: PROBLEM STATEMENT

# ![image.png](attachment:image.png)

# - Image Source: https://commons.wikimedia.org/wiki/File:Chicago_skyline,_viewed_from_John_Hancock_Center.jpg
# - The Chicago Crime dataset contains a summary of the reported crimes occurred in the City of Chicago from 2001 to 2017. 
# - Dataset has been obtained from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system.
# - Dataset contains the following columns: 
#     - ID: Unique identifier for the record.
#     - Case Number: The Chicago Police Department RD Number (Records Division Number), which is unique to the incident.
#     - Date: Date when the incident occurred.
#     - Block: address where the incident occurred
#     - IUCR: The Illinois Unifrom Crime Reporting code.
#     - Primary Type: The primary description of the IUCR code.
#     - Description: The secondary description of the IUCR code, a subcategory of the primary description.
#     - Location Description: Description of the location where the incident occurred.
#     - Arrest: Indicates whether an arrest was made.
#     - Domestic: Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act.
#     - Beat: Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. 
#     - District: Indicates the police district where the incident occurred. 
#     - Ward: The ward (City Council district) where the incident occurred. 
#     - Community Area: Indicates the community area where the incident occurred. Chicago has 77 community areas. 
#     - FBI Code: Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). 
#     - X Coordinate: The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. 
#     - Y Coordinate: The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. 
#     - Year: Year the incident occurred.
#     - Updated On: Date and time the record was last updated.
#     - Latitude: The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
#     - Longitude: The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
#     - Location: The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block.
# - Datasource: https://www.kaggle.com/currie32/crimes-in-chicago

# - You must install fbprophet package as follows: 
#      pip install fbprophet
#      
# - If you encounter an error, try: 
#     conda install -c conda-forge fbprophet
# 
# - Prophet is open source software released by Facebook’s Core Data Science team.
# 
# - Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
# 
# - Prophet works best with time series that have strong seasonal effects and several seasons of historical data. 
# 
# - For more information, please check this out: https://research.fb.com/prophet-forecasting-at-scale/
# https://facebook.github.io/prophet/docs/quick_start.html#python-api
# 

# # STEP #1: IMPORTING DATA

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet


# In[3]:


df_chicago_1=pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
df_chicago_2=pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
df_chicago_3=pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)

#We have errors because of the data quality. we will ignore the errors.


# As you see we have 3 datasets, but we will work on all values between 2005-2017, because of that we can bring together all datasets, which means we will concatinate the datasets:

# In[4]:


df_chicago=pd.concat([df_chicago_1,df_chicago_2,df_chicago_3])


# In[5]:


df_chicago_1.shape


# In[6]:


df_chicago_2.shape


# In[7]:


df_chicago_3.shape


# In[8]:


df_chicago.shape


# # STEP #2: EXPLORING THE DATASET  

# In[9]:


df_chicago.head()


# In[11]:


df_chicago.tail()


# ### Now let's see if we have any missing values:

# In[12]:


plt.figure(figsize=(10,10))  #Here the pyplot figure size is width and height by inches.
sns.heatmap(df_chicago.isnull(), cbar=False, cmap='YlGnBu')  #We do not need color bar, so it is False.


# In[10]:


df_chicago.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)


# In[11]:


df_chicago.drop(['ID'], inplace=True, axis=1)


# In[12]:


df_chicago


# ### We need a proper date time format in order to use the Prophet. Now the Date column is just a string.

# In[13]:


df_chicago.Date=pd.to_datetime(df_chicago.Date, format='%m/%d/%Y %I:%M:%S %p')


# In[14]:


df_chicago   # As you see the Date column has changed:


# ### Now let's see how many count we have for each "Primary Type"

# In[15]:


df_chicago['Primary Type'].value_counts()


# #### If you want to see the TOP XXX Labels, you can add "iloc"

# In[16]:


df_chicago['Primary Type'].value_counts().iloc[:10]  #Top 10 Crimes


# #### Let's get index to each label:

# In[17]:


df_chicago['Primary Type'].value_counts().iloc[:10].index


# ### Let's plot the top 10 as a graphic by using seaborn:

# In[18]:


primary_type_order=df_chicago['Primary Type'].value_counts().iloc[:10].index  #We will get order info to plot in an order


# In[19]:


plt.figure(figsize=(10,10))
sns.countplot(y='Primary Type', data=df_chicago, order=primary_type_order)


# ### This time let's plot the "Location Description" column:

# In[33]:


plt.figure(figsize=(15,10))
sns.countplot(y='Location Description', data=df_chicago, order=df_chicago['Location Description'].value_counts().iloc[:10].index)


# #### Now we will set the "Date" column as the dataframe index:

# In[21]:


df_chicago.index=pd.DatetimeIndex(df_chicago.Date)


# In[22]:


df_chicago  #Index column is shown as bold


# #### We want to look the total incidents for each year, for this we will use "resample":
# #### Resample is a Convenience method for frequency conversion and resampling of time series.

# In[23]:


df_chicago.resample('Y').size()


# #### Let's plot this statistic:

# In[24]:


plt.plot(df_chicago.resample('Y').size())
plt.title('Total Number of Crimes per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')


# In[25]:


plt.plot(df_chicago.resample('M').size())
plt.title('Total Number of Crimes per Month')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')


# In[27]:


plt.plot(df_chicago.resample('Q').size())
plt.title('Total Number of Crimes per Quarter')
plt.xlabel('Quarter')
plt.ylabel('Number of Crimes')


# # STEP #3: PREPARING THE DATA

# ### New we will use Prophet

# In[28]:


chicago_prophet=df_chicago.resample('M').size()


# In[29]:


chicago_prophet


# #### As you see above, for the resample output, we do not have a row index other than Date column. Let's create index:

# In[30]:


chicago_prophet=df_chicago.resample('M').size().reset_index()


# In[31]:


chicago_prophet


# In[34]:


chicago_prophet.columns=['Date','Crime Count']


# In[35]:


chicago_prophet


# In[36]:


df_chicago_prophet=chicago_prophet.rename(columns={'Date':'ds', 'Crime Count':'y'})


# In[37]:


df_chicago_prophet


# # STEP #4: MAKE PREDICTIONS

# In[38]:


model=Prophet()
model.fit(df_chicago_prophet)


# ### Now model is ready. Next, we should tell the prophet model which future should it predict:

# In[48]:


future_one_year=model.make_future_dataframe(periods=365)


# In[49]:


forecast_one_year=model.predict(future)


# In[50]:


forecast_one_year


# In[51]:


figure=model.plot(forecast_one_year, xlabel='Date', ylabel='Crime Rate')


# ### We can also plot as trend, for this, instead of writing "plot", we write "plot_components":

# In[52]:


figure_trend=model.plot_components(forecast_one_year)


# In[53]:


future_two_years=model.make_future_dataframe(periods=730)


# In[54]:


forecast_two_years=model.predict(future_two_years)


# In[59]:


figure_two_years=model.plot(forecast_two_years, xlabel='Date', ylabel='Crime Rate')


# In[58]:


figure_two_years_trend=model.plot_components(forecast_two_years)


# ### Thanks!

# If you have any question please feel free to contact with me:
# * github.com/EmrahYener
# * linkedin.com/in/emrah-yener
# * xing.com/profile/emrah_yener
# 
# Sources:
# * https://www.udemy.com/course/deep-learning-machine-learning-practical/
# 

# In[ ]:




