import streamlit as st
import pickle
import pandas as pd
import numpy as np
#from streamlit_lottie import st_lottie
final_df = pd.read_csv('result.csv')
final_df=final_df.drop(columns=['Unnamed: 0'])
delivery_df=pd.read_csv('IPL.csv')
match = pd.read_csv('matches.csv')
x=final_df.drop(columns='result') 
y=final_df['result']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder()
trf = ColumnTransformer([ ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','city'])],remainder='passthrough') 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.pipeline import Pipeline 
pipe=Pipeline( steps=[ ('step1',trf), 
                       ('step2', LogisticRegression()) ]) 
pipe.fit(xtrain,ytrain)
y_pred = pipe.predict(xtest)
st.title('IPL Match Predication & Analysis')
# Add a radio button to the sidebar
part = st.radio(" ", ["Prediction", "Analysis"],horizontal=True)

if part == "Prediction":
    # Prediction part
    st.title('IPL Win Predictor')
    batting=final_df['batting_team'].unique()
    shar=final_df['city'].unique()
    wic2=[1,2,3,4,5,6,7,8,9,10]
    wic1=[0,1,2,3,4,5,6,7,8,9,10]
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

    n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','city','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket'],data=np.array([a,b,c,d,e,f,k,g,l,h,i]).reshape(1,11))).astype(float)
    probablity=str(int(n[0][1]*100))
    if a!=b:
        if st.button('Predict'):
            st.write(f"Win probablity of Batting Team:  :red[{probablity}]"+"%")

elif part == "Analysis":
    st.title('Analysis of Previous Matches')

    batting = ['Kolkata Knight Riders', 'Royal Challengers Bangalore',
               'Delhi Daredevils', 'Mumbai Indians', 'Kings XI Punjab',
               'Deccan Chargers', 'Chennai Super Kings', 'Rajasthan Royals',
               'Sunrisers Hyderabad', 'Delhi Capitals']

    col1, col2, col3 = st.columns(3)
    with col1:
      a1 = st.selectbox('batting team', sorted(batting))
    with col2:
        b1 = st.selectbox('bowling team', sorted(batting))
    with col3:
        c1 = st.selectbox('Season', sorted(match['season'].unique()))

    match = match[match['team2'] == a1]
    match = match[match['team1'] == b1]
    match = match[match['season'] == c1]
    g = match['id'].unique()
    l = st.selectbox('Match_id', g)

    def match_progression(x_df, match_id, pipe):
        match = x_df[x_df['match_id'] == match_id]
        match = match[(match['ball'] == 6)]
        temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'ball_left', 'wickets_left', 'total_runs_x', 'crr', 'rrr', 'last_five', 'last_five_wicket']].fillna(0)
        temp_df = temp_df[temp_df['ball_left']!= 0]
        if temp_df.empty:
            print("Error: Match is not Existed")
            return None, None
        result = pipe.predict_proba(temp_df)
        temp_df['lose'] = np.round(result.T[0]*100, 1)
        temp_df['win'] = np.round(result.T[1]*100, 1)
        temp_df['end_of_over'] = range(1, temp_df.shape[0]+1)
        target = temp_df['total_runs_x'].values[0]
        runs = list(temp_df['runs_left'].values)
        new_runs = runs[:]
        runs.insert(0, target)
        temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
        wickets = list(temp_df['wickets_left'].values)
        new_wickets = wickets[:]
        new_wickets.insert(0, 10)
        wickets.append(0)
        w = np.array(wickets)
        nw = np.array(new_wickets)
        temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
        temp_df['score']=temp_df['runs_after_over'].cumsum()
        print("Target-", target)
        temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'score','lose', 'win']]
        return temp_df, target

    temp_df, target = match_progression(delivery_df, l, pipe)
import plotly.graph_objects as go
if a1 == b1:
    st.write('No match Available')
else:
    if temp_df is None:
        st.write("Error: Match is not Existed")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers', name='Wickets in Over', marker=dict(color='yellow')))
        runs = fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over', marker=dict(color='purple')))
        fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', name='Probability of ' +a1, line=dict(color='#00a65a', width=4)))
        fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', name='Probability of '+ b1, line=dict(color='red', width=4)))
        fig.update_layout(title='Target-' + str(target), legend_title='Legend')
        st.plotly_chart(fig)
        st.write(delivery_df[delivery_df['match_id']==l])
        st.subheader('Summary')
        r1 = match[match['id'] == l]['player_of_match'].unique()
        r2 = match[match['id'] == l]['winner'].unique()
        r3 = match[match['id'] == l]['venue'].unique()
        r4 = match[match['id'] == l]['city'].unique()
        r5 = match[match['id'] == l]['toss_winner'].unique()
        r6 = match[match['id'] == l]['dl_applied'].unique()
        data = {'Field': ['Vanue', 'City', 'Toss Winner', 'DLS Methode', 'POM', 'Winner'],
                'Name': [r3[0], r4[0], r5[0], r6[0], r1[0], r2[0]]}
        fg = pd.DataFrame(data)
        st.table(fg)
        
