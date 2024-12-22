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
    probablity1=int(n[0][1]*100)
    probablity2=int(n[0][0]*100)
    data=[probablity1,probablity2]
    data1=[a,b]
    if a!=b:
        if st.button('Predict'):
          import plotly.graph_objects as go
          fig = go.Figure(data=[go.Pie(labels=data1, values=data, hole=.5)])
          st.write(fig)

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
sf=pd.read_csv('flags_iso.csv')
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Session Distribution', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Define the function for Score Comparison
o=st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 50
h = st.text_input('URL( ESPN CRICINFO >Select Match > Click On Overs )') or 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'
if (h=='https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'):
    st.write('Enter Your URL')
r = requests.get(h)
#r1=requests.get('https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/india-vs-new-zealand-1st-semi-final-1384437/full-scorecard')
b=BeautifulSoup(r.text,'html')
venue=b.find(class_='ds-flex ds-items-center').text.split(',')[1]
list=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
list8=[]
list9=[]
list10=[]
#print(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[49].text)
elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
for i, element in enumerate(elements):
    if not element.text.split('/'):
        print(' ')
    else:
        if i % 2 != 0:
            list.append(element.text.split('/')[0])
            list1.append(element.text.split('/')[1].split('(')[0])
for i, element in enumerate(elements):
    if element.text.split('/') is None:
        print(' ')
    else:
        if i % 2 == 0:
            list8.append(element.text.split('/')[0])
            list9.append(i/2+1)
            list10.append(element.text.split('/')[1].split('(')[0])
            
dict1={'inng1':list8,'over':list9,'wickets':list10}
df1=pd.DataFrame(dict1)
for i in range(len(list)):
    list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
    list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
    list4.append(b.find_all('th',class_='ds-min-w-max')[1].text)
    list5.append(b.find_all('th',class_='ds-min-w-max')[2].text)
    list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])
    if o==50:
        list7.append(b.find(class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo').text.split(' ')[0])
if o==50:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3,'winner':list7} 
else:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3} 
df=pd.DataFrame(dict)

df['score']=df['score'].astype('int')
df1['inng1']=df1['inng1'].astype('int')
df1['over']=df1['over'].astype('int')
df['over']=df['over'].astype('int')
df['wickets']=df['wickets'].astype('int')
df['target']=df['target'].astype('int')
df['runs_left']=df['target']-df['score']
df=df[df['score']<df['target']]
df['crr']=(df['score']/df['over'])
df['rrr']=((df['target']-df['score'])/(50-df['over']))
df['balls_left']=300-(df['over']*6)
df['runs'] = df['score'].diff()
df['last_10']=df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket']=df['wickets_in_over'].rolling(window=10).sum()
df=df.fillna(50)
#st.write(df)
df['match_id']=100001
neg_idx = df1[df1['inng1']<0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf=df
lf=lf[:int(o)]
st.subheader('Scorecard')
o=int(o)
if o != 50:
    # Create a single row with two columns
    col1, col2 = st.columns([1, 1])  # Equal width columns

    with col1:
        bowling_team = df['bowling_team'].unique()[0]
        batting_team = df['batting_team'].unique()[0]

        # Get the URL for the bowling team
        bowling_team_url = sf[sf['Country'] == bowling_team]['URL']
        if not bowling_team_url.empty:
            # Display the bowling team flag and name in the same line
            col_bowling, col_bowling_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_bowling:
                st.image(bowling_team_url.values[0], width=50)  # Adjust width as needed
            with col_bowling_name:
                st.write(f"**{bowling_team}**")

        # Get the URL for the batting team
        batting_team_url = sf[sf['Country'] == batting_team]['URL']
        if not batting_team_url.empty:
            # Display the batting team flag and name in the same line
            col_batting, col_batting_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_batting:
                st.image(batting_team_url.values[0], width=50)  # Adjust width as needed
            with col_batting_name:
                st.write(f"**{batting_team}**")

    with col2:
        # Adjust the layout of col2 to be left-aligned
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)  # Ensure left alignment
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '50)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 8].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 9].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 7]) + ' runs in ' + str(df.iloc[o - 1, 10]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)  # Close the div for left alignment

    # Display teams and results
else:
  col1, col2 = st.columns(2)
  with col1:
    st.write(f"**{df['bowling_team'].unique()[0]}**")
    st.write(f"**{df['batting_team'].unique()[0]}**")
  with col2:
    st.write(str(df['target'].unique()[0]))
    st.write('(' + str(df.iloc[-1, 5]) + '/' + '50)   ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))

  if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
      st.write(winner[0] + ' Won')
    else:
      st.write("Winner information not available.")
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Scatter(x=df1['over'], y=df1['inng1'],line_width=3,line_color='red',name=df['bowling_team'].unique()[0]),
    go.Scatter(x=lf['over'], y=lf['score'],line_width=3,line_color='green',name=df['batting_team'].unique()[0])
])
fig.update_layout(title='Score Comperison',
                  xaxis_title='Over',
                  yaxis_title='Score')
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
            runs = fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker=dict(color='purple')))
            wicket_text = temp_df['wickets_in_over'].astype(str)
            wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over'] * 1  # adjust y-position based on wickets
            wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets

            wicket = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                               mode='markers', name='Wickets in Over',
                                               marker=dict(color='orange', size=10),
                                               text=wicket_text, textposition='top center'))

# Line plots for batting and bowling teams
            batting_team = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', name=a1,
                                              line=dict(color='#00a65a', width=3)))
            bowling_team = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', name=b1,
                                              line=dict(color='red', width=4)))

            fig.update_layout(title='Target-' + str(target))
            st.write(fig)

            
        
