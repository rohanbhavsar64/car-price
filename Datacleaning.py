#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')


# In[4]:


match.head()


# In[5]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df


# In[6]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df


# In[7]:


match_df['city'].unique()


# In[8]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[9]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df.shape


# In[10]:


match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['match_id','city','winner','total_runs']]
delivery_df = match_df.merge(delivery,on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]
delivery_df


# In[11]:


groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
      last_five.extend(groups.get_group(id).rolling(window=18).sum()['total_runs_y'].values.tolist())


# In[12]:


delivery_df['last_five']=last_five


# In[13]:


delivery_df['city'].value_counts().keys()


# In[14]:


delivery_df.info()


# In[15]:


delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[16]:


delivery_df['current score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']
delivery_df


# In[17]:


delivery_df['runs_left']=delivery_df['total_runs_x']+1-delivery_df['current score']


# In[18]:


delivery_df['ball_left']=120-((delivery_df['over']-1)*6+delivery_df['ball'])
delivery_df


# In[19]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x == 0 else 1)
delivery_df['player_dismissed']=delivery_df['player_dismissed']
wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets_left']=10-wickets
delivery_df.sample(5)


# In[20]:


groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=18).sum()['player_dismissed'].values.tolist())


# In[21]:


delivery_df['last_five_wicket']=last_five


# In[22]:


delivery_df['crr']=(delivery_df['current score']*6)/(120-delivery_df['ball_left'])
delivery_df


# In[23]:


delivery_df['rrr']=(delivery_df['runs_left']*6)/(delivery_df['ball_left'])
delivery_df


# In[24]:


def result(raw):
    return 1 if(raw['batting_team']==raw['winner']) else 0


# In[25]:


delivery_df['result']=delivery_df.apply(result,axis=1)


# In[27]:


delivery_df.info()


# In[28]:


final_df = delivery_df[['batting_team','bowling_team','city','batsman','non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','result','last_five','last_five_wicket']]
final_df = final_df.sample(final_df.shape[0])
final_df.head(20)


# In[29]:


final_df=final_df[final_df['ball_left']!=0]


# In[48]:


final_df['batsman']=final_df['batsman'].str.split(' ').str.get(-1)
final_df['non_striker']=final_df['non_striker'].str.split(' ').str.get(-1)
final_df


# In[49]:


final_df.dropna(inplace=True)


# In[50]:


final_df.to_csv('result.csv')


# In[51]:


x=final_df.drop(columns='result')
y=final_df['result']


# In[52]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)


# In[53]:


xtrain


# In[54]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ohe.fit(x[['batting_team','bowling_team','city','batsman','non_striker']])

trf = ColumnTransformer([
    ('trf',OneHotEncoder(categories=ohe.categories_),['batting_team','bowling_team','city','batsman','non_striker'])
]
,remainder='passthrough')


# In[55]:


ohe.categories_


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
fromm
from sklearn.pipeline import Pipeline


# In[57]:


pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2',LogisticRegression())
    ])


# In[58]:


final_df.isnull().sum()


# In[59]:


pipe.fit(xtrain,ytrain)


# In[60]:


y_pred = pipe.predict(xtest)


# In[61]:


pipe.predict_proba(xtest)[1]


# In[62]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[91]:


n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','city','batsman','non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket'],data=np.array(['Royal Challengers Bangalore','Chennai Super Kings','Indore','Dhoni','Sundar',63,42,7,2,11.23,9.00,33,2.0]).reshape(1,13))).astype(float)


# In[92]:


print("Win Chances of Batting team is:", n[0][1]*100,"%")
print("Win Chances of Bowling team is:", n[0][0]*100,"%")


# In[90]:


final_df['city'].unique()


# In[ ]:




