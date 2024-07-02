import streamlit as st
import pickle
import pandas as pd
import numpy as np
#from streamlit_lottie import st_lottie
final_df = pd.read_csv('result.csv')
final_df=final_df.drop(columns=['Unnamed: 0'])
delivery_df=pd.read_csv('IPL.csv')
match = pd.read_csv('matches.csv')
# st.write(df)
x=final_df.drop(columns='result')
y=final_df['result']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','city'])],remainder='passthrough')
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
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket']].fillna(0)
    temp_df = temp_df[temp_df['ball_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        return None, None
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
st.title('Analysis of Previous Matches')
batting=['Kolkata Knight Riders', 'Royal Challengers Bangalore',
       'Delhi Daredevils', 'Mumbai Indians', 'Kings XI Punjab',
       'Deccan Chargers', 'Chennai Super Kings', 'Rajasthan Royals',
       'Sunrisers Hyderabad', 'Delhi Capitals']
col1,col2,col3=st.columns(3)
with col1:
    a1 = st.selectbox('batting team',sorted(batting))
with col2:
    b1 = st.selectbox('bowling team',sorted(batting))
with col3:
    c1=st.selectbox('bowling_team',sorted(match['season'].unique()))
match=match[match['team2']==a1]
match=match[match['team1']==b1]
match=match[match['season']==c1]
g=match['id'].unique()
l=st.selectbox('Match_id',g)
temp_df,target = match_progression(delivery_df,l,pipe)
st.subheader(delivery_df[delivery_df['match_id']==l]['batting_team'].unique()+' v/s '+delivery_df[delivery_df['match_id']==l]['bowling_team'].unique())
st.text('City : '+delivery_df[delivery_df['match_id']==l]['city'].unique())
st.text('Season : '+str(match[match['id']==l]['season'].unique()))
import plotly.graph_objects as go

if a1!= b1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers', name='Wickets in Over', marker=dict(color='yellow')))
    runs=fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over', marker=dict(color='purple'))) 
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', name='Batting Team Probability', line=dict(color='#00a65a', width=4)))
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', name='Bowling Team Probability', line=dict(color='red', width=4)))

    fig.update_layout(title='Target-' + str(target), legend_title='Legend')

    st.plotly_chart(fig)
