import streamlit as st
import pickle
import pandas as pd
import numpy as np
#from streamlit_lottie import st_lottie
final_df = pd.read_csv('result.csv')
final_df=final_df.drop(columns=['Unnamed: 0'])
# st.write(df)
x=final_df.drop(columns='result')
y=final_df['result']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
ohe.fit(x[['batting_team','bowling_team','city']])

trf = ColumnTransformer([
    ('trf',OneHotEncoder(categories=ohe.categories_),['batting_team','bowling_team','city'])
]
,remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2', LogisticRegression())
    ])
pipe.fit(xtrain,ytrain)
y_pred = pipe.predict(xtest)
batting=['Kolkata Knight Riders', 'Royal Challengers Bangalore',
       'Delhi Daredevils', 'Mumbai Indians', 'Kings XI Punjab',
       'Deccan Chargers', 'Chennai Super Kings', 'Rajasthan Royals',
       'Sunrisers Hyderabad', 'Delhi Capitals']

shar=['Kolkata', 'Bangalore', 'Hyderabad', 'Chennai', 'Mumbai', 'Jaipur',
       'Delhi', 'Kimberley', 'Dharamsala', 'Chandigarh', 'Port Elizabeth',
       'Bengaluru', 'Ahmedabad', 'Johannesburg', 'Cuttack', 'Durban',
       'Mohali', 'Raipur', 'Pune', 'Centurion', 'Ranchi', 'Indore',
       'Sharjah', 'Cape Town', 'Nagpur', 'Abu Dhabi', 'East London',
       'Visakhapatnam', 'Bloemfontein']
wic1=['0','1','2','3','4','5','6','7','8','9']
wic2=['1','2','3','4','5','6','7','8','9','10']
st.title('IPL Win Predictor')
col1,col2,col3=st.columns(3)
with col1:
    a = st.selectbox('batting_team',sorted(batting))
with col2:
    b = st.selectbox('bowling_team',sorted(batting))
with col3:
    c= st.selectbox('city',sorted(shar))
col1,col2,col3=st.columns(3)
with col1:
    d= int(st.number_input('runs_left'))
with col2:
    f=st.selectbox('wickets_left',wic2)
with col3:
    g=st.number_input('crr')
col1,col2,col3=st.columns(3)
with col1:
    h=st.number_input('Runs in last three overs')
with col2:
    i=st.selectbox('Wickets in last three overs',sorted(wic1))
with col3:
    e= st.number_input('balls left in Inning')
col1,col2=st.columns(2)
with col1:
    k=st.number_input('target')
with col2:
    l=st.number_input('required run rate')
#input_df = pd.DataFrame(
  #  {'name': [name], 'company': [company], 'year': [year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})
n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','city','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket'],data=np.array([a,b,c,d,e,f,k,g,l,h,i]).reshape(1,11))).astype(float)
# result = pipe.predict(input_df)
#st.subheader('SOME IMPORTANT RECORDS:')
#z=str(int((final_df[final_df['batting_team']==a]['result'].sum())/final_df[final_df['batting_team']==a]['result'].shape*100))
#st.write(f'Winning percentage  Team  :orange[{a}]'+" "+f'with 2nd Batting is: :red[{z}]'+'%')
#y = str(int(100-(final_df[final_df['bowling_team'] == b]['result'].sum()) / final_df[final_df['bowling_team'] == b]['result'].shape * 100))
#st.write(f'Winning percentage  Team :orange[{b}]' + ' ' + f'with 2nd Bowling is: :red[{y}]' + '%')
#x= str(int((final_df[final_df['city'] == c]['result'].sum()) / final_df[final_df['city'] == c][ 'result'].shape * 100))
#st.write(f'Win % in :orange[{c}]' + ' ' + f'with 2nd Batting is : :red[{x}]' + '%')
probablity=str(int(n[0][1]*100))
if a!=b:
    if st.button('Predict'):
        st.write(f"Win probablity of Batting Team:  :red[{probablity}]"+"%")
        df=final_df[final_df['batting_team']==a][final_df['bowling_team']==b]
        #i1=sns.barplot(x='result',y='total_runs_x',data=df)
        #st.pyplot(plt.gcf())
        #st.write(df['result'].value_counts())

