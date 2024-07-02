import streamlit as st
import pickle
import pandas as pd
import numpy as np
#from streamlit_lottie import st_lottie
final_df = pd.read_csv('result.csv')
final_df=final_df.drop(columns=['Unnamed: 0'])
delivery_df=pd.read_csv('IPL.csv')
match = pd.read_csv('matches.csv')

st.title('IPL Win Predictor')

# Add a radio button to the sidebar
part = st.sidebar.radio("Select Part", ["Prediction", "Analysis"])

if part == "Prediction":
    # Prediction part
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
    # Analysis part
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
        c1=st.selectbox('Season',sorted(match['season'].unique()))
    match=match[match['team2']==a1]
    match=match[match['team1']==b1]
    match=match[match['season']==c1]
    g=match['id'].unique()
    l=st.selectbox('Match_id',g)
    temp_df,target = match_progression(delivery_df,l,pipe)
    st.subheader(delivery_df[delivery_df['match_id']==l]['batting_team'].unique()+' v/s '+delivery_df[delivery_df['match_id']==l]['bowling_team'].unique())
    st.text('City : '+delivery_df[delivery_df['match_id']==l]['city'].unique())
    st.text('Season : '+str(match[match['id']==l]['season'].unique()))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers', name='Wickets in Over', marker=dict(color='yellow')))
    runs=fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over', marker=dict(color='purple'))) 
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', name='Batting Team Probability', line=dict(color='#00a65a', width=4)))
    fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', name='Bowling Team Probability', line=dict(color='red', width=4)))
    fig.update_layout(title='Target-' + str(target), legend_title='Legend')
    st.plotly_chart(fig)
