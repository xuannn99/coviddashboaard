# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:28:42 2021

@author: Lim Zi Xuan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
import plotly.graph_objects as go
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from datetime import timedelta


st.cache(persist=True)
def load_data():
    covid=pd.read_csv("coviddata0330.csv")
    latest=covid[covid["Date"] == "2021-03-30"][["Country","Population","Confirmed","Deaths","Recovered","Active"]]
    confirmed_df = pd.read_csv("confirmed_0330.csv")
    death_df = pd.read_csv("deaths_0330.csv")
    recovered_df = pd.read_csv("recovered_0330.csv")
    return covid, latest, confirmed_df, death_df, recovered_df
covid, latest, confirmed_df, death_df, recovered_df= load_data()

covid1 = covid.copy()
covid1["Date"]=pd.to_datetime(covid1["Date"], errors='coerce')


def breakline():
    return st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: red;'> 🦠 Covid-19 Dashboard 🦠</h1>", unsafe_allow_html=True)

confirmed_total = int(latest['Confirmed'].sum())
active_total = int(latest['Active'].sum())
deaths_total = int(latest['Deaths'].sum())
recovered_total = int(latest['Recovered'].sum())

st.title("World Data Summary: ")
st.markdown(
        """
        Total Confirmed Cases | Total Active Cases | Total Recovered Cases | Total Deaths Cases
        ----------------|--------------|------------|----------
        {0}             | {1}          | {2}        | {3} 
        
        """.format(confirmed_total, active_total, recovered_total, deaths_total)
    )

breakline()


st.markdown("<h2 style='text-align: center; color: black; background-color:lightcoral'>Covid pandemic across the world</h2>",
            unsafe_allow_html=True)

df_list = []
labels = []
colors = []
colors_dict =  {
        'Confirmed' : 'lightblue',
        'Deaths' : 'red',
        'Recovered' : 'green'
    }
features = st.multiselect("Select Confirmed/Deaths/Cases : ", 
                          ['Confirmed', 'Deaths', 'Recovered'],
                          default = ['Confirmed','Recovered','Deaths'],
                          key = 'world')
for feature in features:
    if feature == 'Confirmed':
        labels.append('Confirmed')
        colors.append(colors_dict['Confirmed'])
        df_list.append(confirmed_df)
    if feature == 'Deaths':
        labels.append('Deaths')
        colors.append(colors_dict['Deaths'])
        df_list.append(death_df)
    if feature == 'Recovered':
        labels.append('Recovered')
        colors.append(colors_dict['Recovered'])
        df_list.append(recovered_df)
        
    line_size = [4, 5, 6]
    
fig_world = go.Figure();
    
for i, df in enumerate(df_list):
        x_data = np.array(list(df.iloc[:, 4:].columns))
        y_data = np.sum(np.asarray(df.iloc[:,4:]),axis = 0)
            
        fig_world.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers',
        name=labels[i],
        line=dict(color=colors[i], width=line_size[i]),
        connectgaps=True,
        text = "Total " + str(labels[i]) +": "+ str(y_data[-1])
        ));
    
        fig_world.update_layout(
        title="COVID-19 cases of World",
        xaxis_title='Date',
        yaxis_title='No. of Cases',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        width = 800,
        
        );

fig_world.update_yaxes(type="linear")
st.plotly_chart(fig_world);

 
     
st.header("Select Country to visualize")
city = st.selectbox("Country",covid["Country"][:190])

st.markdown("Info: ")
filtered_dict = covid[covid['Country']== city].to_dict('records')[0]
filtered_dict1 = latest[latest['Country']== city].to_dict('records')[0]

fig_dc = go.Figure()
fig_dc.add_trace(
        go.Indicator(mode = "number",
        value = filtered_dict['Population'],
        #delta = {'position': "top", 'reference': 320},
        domain = {'row': 0, 'column': 0},
        title = {'text': 'Population <br> '}
        ))
fig_dc.add_trace(
        go.Indicator(mode = "number",
        value = filtered_dict1['Confirmed'],
        #delta = {'position': "top", 'reference': 320},
        domain = {'row': 0, 'column': 1},
        title = {'text': 'Confirmed <br> Case', 'font.color': "lightblue"}
        ))
fig_dc.add_trace(
        go.Indicator(mode = "number",
        value = filtered_dict1['Deaths'],
        #delta = {'position': "top", 'reference': 320},
        domain = {'row': 0, 'column': 2},
        title = {'text': 'Deaths <br> Case', 'font.color': "red"}
        ))
fig_dc.add_trace(
        go.Indicator(mode = "number",
        value = filtered_dict1['Recovered'],
        #delta = {'position': "top", 'reference': 320},
        domain = {'row': 0, 'column': 3},
        title = {'text': 'Recovered <br> Case', 'font.color': "green"}
        ))
fig_dc.update_layout(
        grid = {'rows': 1, 'columns': 4, 'pattern': "independent"},
        width = 700,
        height = 200,
        )
st.plotly_chart(fig_dc)


def dataf(latest):
    total_d = pd.DataFrame({
        'Status':['Confirmed', 'Deaths', 'Recovered', 'Active'],
        'Cases':(latest['Confirmed'].iloc[0],
                 latest['Deaths'].iloc[0],
                 latest['Recovered'].iloc[0],
                 latest['Active'].iloc[0])})
    return total_d


tt = dataf(latest[latest["Country"]== city])
if st.checkbox("View Bar chart",False):
    bar = px.bar(tt, x= 'Status', y= 'Cases',color = 'Status')
    st.plotly_chart(bar)

st.header(f"View Daily Cases for {city}")
chartopt = st.radio("Select type of Chart",("Line Chart","Scatter Chart"))
daily_c = st.selectbox("Select option",('Daily New Cases', 'Daily New Recoveries','Daily New Deaths'))

city1= covid[covid["Country"]== city]   
new_c  = px.line(city1, x = city1["Date"], y = city1["New cases"], title= "New Cases in {}".format(city), width= 700, height= 450)
new_cs  = px.scatter(city1, x = city1["Date"], y = city1["New cases"], title= "New Cases in {}".format(city), width= 700, height= 450)

new_d  = px.line(city1, x = city1["Date"], y = city1["New deaths"], title= "New Deaths in {}".format(city), width= 700, height= 450)
new_ds  = px.scatter(city1, x = city1["Date"], y = city1["New deaths"], title= "New Deaths in {}".format(city), width= 700, height= 450)

new_r  = px.line(city1, x = city1["Date"], y = city1["New recovered"], title= "New Recovered in {}".format(city), width= 700, height= 450)
new_rs  = px.scatter(city1, x = city1["Date"], y = city1["New recovered"], title= "New Recovered in {}".format(city), width= 700, height= 450)


if daily_c =='Daily New Cases':
    if chartopt == "Line Chart":
        st.plotly_chart(new_c)
    else:
        st.plotly_chart(new_cs)
elif daily_c =='Daily New Recoveries':
    if chartopt == "Line Chart":
        st.plotly_chart(new_r)
    else:
        st.plotly_chart(new_rs)
elif daily_c =='Daily New Deaths':
    if chartopt == "Line Chart":
        st.plotly_chart(new_d)
    else:
        st.plotly_chart(new_ds)
       
        
st.header(f" View Cases for {city} : ")    
case_type = st.selectbox("Select case type : ", ['Confirmed', 'Deaths', 'Recovered', 'Confirmed, Deaths and Recovered'])    

case_c  = go.Line(x = pd.to_datetime(city1["Date"]), y = city1["Confirmed"], name = 'TOTAL CONFIRMED CASES')
case_d  = go.Line( x = pd.to_datetime(city1["Date"]), y = city1["Deaths"], name = 'TOTAL DEATHS')
case_r  = go.Line(x = pd.to_datetime(city1["Date"]), y = city1["Recovered"], name = 'TOTAL RECOVERED')
    
if case_type == 'Confirmed':
    figc = go.Figure(case_c)  
    figc.update_layout(title = "Confirmed Cases in {}".format(city))
    st.plotly_chart(figc)    
elif case_type == 'Deaths':      
    figd = go.Figure(case_d)  
    figd.update_layout(title = "Deaths Cases in {}".format(city))
    st.plotly_chart(figd)     
elif case_type == 'Recovered':        
    figr = go.Figure(case_r) 
    figr.update_layout(title = "Recovered Cases in {}".format(city))
    st.plotly_chart(figr)
elif case_type == 'Confirmed, Deaths and Recovered':        
    fign = go.Figure([case_c, case_d, case_r])
    fign.update_layout(legend_orientation='h', title = "Confirmed, Deaths and Recovered Cases in {}".format(city))    
    st.plotly_chart(fign) 
    

st.header(f"View Cases for {city} by Month/Day/Date")    
    
a= alt.Chart(covid1[covid1["Country"]==city],width=500,height=400).mark_bar().encode(
    x="day(Date):O",
    y="month(Date):O",
    color=alt.Color('sum(New deaths)',scale=alt.Scale(range=['ivory', 'red'])),
    tooltip="sum(New deaths)"
)

b= alt.Chart(covid1[covid1["Country"]==city],width=500,height=400).mark_text().encode(
    x="day(Date):O",
    y="month(Date):O",
    text="sum(New deaths)" 
)


c= alt.Chart(covid1[covid1["Country"]==city],width=900,height=300).mark_bar().encode(
    x="date(Date):O",
    y="month(Date):O",
    color=alt.Color('sum(New deaths)',scale=alt.Scale(range=['ivory', 'red'])),
    tooltip="sum(New deaths)"
)

d= alt.Chart(covid1[covid1["Country"]==city],width=900,height=300).mark_text(angle=270).encode(
    x="date(Date):O",
    y="month(Date):O",
    text="sum(New deaths)" 
)

an= alt.Chart(covid1[covid1["Country"]==city],width=500,height=400).mark_bar().encode(
    x="day(Date):O",
    y="month(Date):O",
    color=alt.Color('sum(New cases)',scale=alt.Scale(range=['azure', 'DodgerBlue'])),
    tooltip="sum(New cases)"
)

bn=alt.Chart(covid1[covid1["Country"]==city],width=500,height=400).mark_text().encode(
    x="day(Date):O",
    y="month(Date):O",
    text="sum(New cases)" 
)


cn= alt.Chart(covid1[covid1["Country"]==city],width=900,height=300).mark_bar().encode(
    x="date(Date):O",
    y="month(Date):O",
    color=alt.Color('sum(New cases)',scale=alt.Scale(range=['azure', 'DodgerBlue'])),
    tooltip="sum(New cases)"
)

dn=alt.Chart(covid1[covid1["Country"]==city],width=900,height=300).mark_text(angle=270).encode(
    x="date(Date):O",
    y="month(Date):O",
    text="sum(New cases)" 
)

opc = st.radio("Select the option",('New case', 'New deaths'))
op = st.radio("Select the option",('Day and Month','Date and Month'))

if op == 'Day and Month':
    if opc == "New deaths":
        st.altair_chart(a+b)
    else:
        st.altair_chart(an+bn)
elif op == 'Date and Month':
    if opc == "New deaths":
        st.altair_chart(c+d)
    else:
        st.altair_chart(cn+dn)


st.header("View Countries with Highest Cases: ")
opt_selected = st.selectbox("", options=['Top Confirmed Cases', 'Top Death Cases', 'Top Recovered Cases', 'Highest Death Percentage', 'Highest Confirmed Percentage', 'Highest Recovered Percentage'])
country_count = st.slider('No. of countries :', 
                          min_value=2, max_value=10, 
                          value=5, key='incident_count')

#confirmed
table_data_c = latest.sort_values("Confirmed",ascending=False)[["Country","Confirmed"]].head(country_count)
table_data_c.reset_index(inplace = True,drop = True)
    
fig1_c = ff.create_table(table_data_c, height_constant=30)
fig2_c = px.bar(table_data_c, y='Confirmed',x='Country',color='Country',height=400)
fig2_c.update_layout(title='Top Confirmed Cases Country',xaxis_title='Country',yaxis_title='Total Confirmed Case',template="plotly_dark")

#death
table_data_d = latest.sort_values("Deaths",ascending=False)[["Country","Deaths"]].head(country_count)
table_data_d.reset_index(inplace = True,drop = True)
    
fig1_d = ff.create_table(table_data_d, height_constant=30)
fig2_d = px.bar(table_data_d, y='Deaths',x='Country',color='Country',height=400)
fig2_d.update_layout(title='Top Death Cases Country',xaxis_title='Country',yaxis_title='Total Death Case',template="plotly_dark")

#Recovered
table_data_r = latest.sort_values("Recovered",ascending=False)[["Country","Recovered"]].head(country_count)
table_data_r.reset_index(inplace = True,drop = True)
    
fig1_r = ff.create_table(table_data_r, height_constant=30)
fig2_r = px.bar(table_data_r, y='Recovered',x='Country',color='Country',height=400)
fig2_r.update_layout(title='Top Recovered Cases Country',xaxis_title='Country',yaxis_title='Total Recovered Case',template="plotly_dark")

#confirmedp
conp= latest["Confirmed"] / latest["Population"] * 100
latest_p = latest.copy()
latest_p["Confirmed Percentage"] = conp
con_p = latest_p.sort_values("Confirmed Percentage",ascending=False)[["Country","Confirmed Percentage"]].head(country_count)
con_p.reset_index(inplace = True,drop = True)

fig1_cp = ff.create_table(con_p, height_constant=30)
fig2_cp = px.bar(con_p, y='Confirmed Percentage',x='Country',color='Country',height=400)
fig2_cp.update_layout(title='Top Confirmed Percentage Country',xaxis_title='Country',yaxis_title='Total Confirmed Percentage',template="plotly_dark")


#death per
deathp= latest["Deaths"] / latest["Confirmed"] * 100
latest_p = latest.copy()
latest_p["Death Percentage"] = deathp
death_p = latest_p.sort_values("Death Percentage",ascending=False)[["Country","Death Percentage"]].head(country_count)
death_p.reset_index(inplace = True,drop = True)

fig1_dp = ff.create_table(death_p, height_constant=30)
fig2_dp = px.bar(death_p, y='Death Percentage',x='Country',color='Country',height=400)
fig2_dp.update_layout(title='Top Death Percentage Country',xaxis_title='Country',yaxis_title='Total Death Percentage',template="plotly_dark")

#recoveredp
recp= latest["Recovered"] / latest["Confirmed"] * 100
latest_p = latest.copy()
latest_p["Recovered Percentage"] = recp
rec_p = latest_p.sort_values("Recovered Percentage",ascending=False)[["Country","Recovered Percentage"]].head(country_count)
rec_p.reset_index(inplace = True,drop = True)

fig1_rp = ff.create_table(rec_p, height_constant=30)
fig2_rp = px.bar(rec_p, y='Recovered Percentage',x='Country',color='Country',height=400)
fig2_rp.update_layout(title='Top Recovered Percentage Country',xaxis_title='Country',yaxis_title='Total Recovered Percentage',template="plotly_dark")

if opt_selected == 'Top Confirmed Cases':
    st.plotly_chart(fig2_c)
    st.plotly_chart(fig1_c)
elif  opt_selected == 'Top Death Cases':
    st.plotly_chart(fig2_d)
    st.plotly_chart(fig1_d)
elif  opt_selected == 'Top Recovered Cases':
    st.plotly_chart(fig2_r)
    st.plotly_chart(fig1_r)
elif  opt_selected == 'Highest Confirmed Percentage':
    st.plotly_chart(fig2_cp)
    st.plotly_chart(fig1_cp)
elif  opt_selected == 'Highest Death Percentage':
    st.plotly_chart(fig2_dp)
    st.plotly_chart(fig1_dp)
else:
    st.plotly_chart(fig2_rp)
    st.plotly_chart(fig1_rp)
      
breakline()
st.header("View Countries with Least Cases: ")
type_of_case = st.selectbox('Select type of case : ', 
                            ['Confirmed', 'Active', 'Deaths', 'Recovered'],
                            key = 'least_cases')
selected_count = st.slider('No. of countries :', 
                           min_value=1, max_value=10, 
                           value=5, key = 'least_cases')
sorted_country_df = latest[latest[type_of_case] > 0].sort_values(type_of_case, ascending= True)
def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="Country", y=type_of_case, size=type_of_case, color="Country",
               hover_name="Country", size_max=60)
    fig.update_layout(
    title=str(n) +" Countries with least " + type_of_case.lower() + " cases",
    xaxis_title="Countries",
    yaxis_title= type_of_case + " Cases",
    width = 800
    )   
    st.plotly_chart(fig);
bubble_chart(selected_count)


st.header("View Covid-19 details by Date")
cty = st.selectbox(" Select Country",covid1["Country"][:191], key="cities")
ddd = st.date_input(
     "Select the Date",value = datetime.date(2020, 2, 1),min_value = datetime.date(2020, 2, 1), max_value= datetime.date(2021, 3, 13))
st.write('The selected date is:', ddd)


city_v = covid1[(covid1['Date'] == pd.Timestamp(ddd)) & (covid1['Country'] == cty)][["Country","Date","Confirmed", "Deaths", "Recovered", "Active"]].reset_index(drop=True)
st.dataframe(city_v)


st.markdown("<h2 style='text-align: center; color: black; background-color:lightgreen'>Countries with zero cases</h2>",
            unsafe_allow_html=True)
case_type = st.selectbox("Select case type : ", 
                         ['Confirmed', 'Active', 'Deaths'], 1,
                         key= 'zero_cases')
temp_df = latest[latest[case_type] == 0]
st.write(' Countries with zero ' + case_type.lower() + ' cases :')
if len(temp_df) == 0:
    st.error('There are no records present where ' + case_type.lower() + ' cases are zero!')
else:
    temp_df = temp_df[['Country', 'Confirmed', 'Deaths', 'Recovered', 'Active']].reset_index(drop=True)
    st.write(temp_df)
    
breakline()


st.title("Prediction for Confirmed cases")
covid3 =covid1.groupby(["Date"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum',"New cases":"sum"})

covid3["Days Since"]=covid3.index-covid3.index[0]
covid3["Days Since"]=covid3["Days Since"].dt.days


st.header('Linear Regression')

train_ml=covid3.iloc[:int(covid3.shape[0]*0.80)]
valid_ml=covid3.iloc[int(covid3.shape[0]*0.80):]
model_scores=[]
r2_scores=[]

lin_reg=LinearRegression(normalize=True)
lin_reg.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
accuracy = lin_reg.score(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

prediction_valid_linreg=lin_reg.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))


model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))


prediction_linreg=lin_reg.predict(np.array(covid3["Days Since"]).reshape(-1,1))
linreg_output=[]
for i in range(prediction_linreg.shape[0]):
    linreg_output.append(prediction_linreg[i][0])

figp=go.Figure()
figp.add_trace(go.Scatter(x=covid3.index, y=covid3["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
figp.add_trace(go.Scatter(x=covid3.index, y=linreg_output,
                    mode='lines',name="Linear Regression Best Fit Line",
                    line=dict(color='black', dash='dot')))
figp.update_layout(title="Confirmed Cases Linear Regression Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
st.plotly_chart(figp)

st.write("Root Mean Square Error for Linear Regression: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_linreg)))
st.write(f'Training Accuracy: {round(accuracy*100,3)} %')



st.header('Polynomial Regression')
train_ml=covid3.iloc[:int(covid3.shape[0]*0.90)]
valid_ml=covid3.iloc[int(covid3.shape[0]*0.90):]



degree_num = st.slider('No. of Degree :', 
                          min_value=1, max_value=10, 
                          value=6, key='degree')


poly = PolynomialFeatures(degree = degree_num)

train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["Confirmed"]

linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)
accuracy_poly = linreg.score(train_poly,y)


prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_poly))
model_scores.append(rmse_poly)


comp_data=poly.fit_transform(np.array(covid3["Days Since"]).reshape(-1,1))

predictions_poly=linreg.predict(comp_data)

figpp=go.Figure()
figpp.add_trace(go.Scatter(x=covid3.index, y=covid3["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
figpp.add_trace(go.Scatter(x=covid3.index, y=predictions_poly,
                    mode='lines',name="Polynomial Regression Best Fit",
                    line=dict(color='black', dash='dot')))
figpp.update_layout(title="Confirmed Cases Polynomial Regression Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))

st.plotly_chart(figpp)
st.write("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)
st.write("R2 score: ",r2_score(valid_ml["Confirmed"],prediction_poly))
st.write(f' Training Accuracy: {round(accuracy_poly*100,3)} %')


st.header('SVM Model')
c_num = st.slider('Penalty Parameter, C :', 
                          min_value=1, max_value=10, 
                          value=3, key='C')

degrees_num = st.slider('No. of degree :', 
                          min_value=1, max_value=10, 
                          value=5, key='degreesvm')
#Intializing SVR Model
svm=SVR(C=c_num,degree=degrees_num,kernel='poly',epsilon=0.01)

#Fitting model on the training data
svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))
accuracy_svm = svm.score(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["Confirmed"]).reshape(-1,1))

prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))


model_scores.append(np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))


prediction_svm=svm.predict(np.array(covid3["Days Since"]).reshape(-1,1))
figs=go.Figure()
figs.add_trace(go.Scatter(x=covid3.index, y=covid3["Confirmed"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
figs.add_trace(go.Scatter(x=covid3.index, y=prediction_svm,
                    mode='lines',name="Support Vector Machine Best fit Kernel",
                    line=dict(color='black', dash='dot')))
figs.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))

st.plotly_chart(figs)
st.write("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["Confirmed"],prediction_valid_svm)))
st.write(f'Training Accuracy: {round(accuracy_svm*100,3)} %')

breakline()

new_date=[]
new_prediction_lr=[]
new_prediction_poly=[]
for i in range(1,18):
    new_date_poly=poly.fit_transform(np.array(covid3["Days Since"].max()+i).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])
new_prediction_svm=[]
for i in range(1,18):
    new_date.append(covid3.index[-1]+timedelta(days=i))
    new_prediction_lr.append(lin_reg.predict(np.array(covid3["Days Since"].max()+i).reshape(-1,1))[0][0])
    new_prediction_svm.append(svm.predict(np.array(covid3["Days Since"].max()+i).reshape(-1,1))[0])

pd.set_option('display.float_format', lambda x: '%.6f' % x)
model_predictions=pd.DataFrame(zip(new_date,new_prediction_lr,new_prediction_poly,new_prediction_svm),
                               columns=["Dates","Linear Regression Prediction","Polynonmial Regression Prediction","SVM Prediction"])

breakline()

st.header('Prediction for future cases')
model_predictions.head()
st.dataframe(model_predictions)



















