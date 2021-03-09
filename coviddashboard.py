# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:04:18 2021

@author: Lim Zi Xuan
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt
import plotly.graph_objects as go
import datetime




st.cache(persist=True)
def load_data():
    covid=pd.read_csv("D:/kdu/Final Year Project/Dataset/COVID-19 dataset.csv")
    covid1=pd.read_csv("D:/kdu/Final Year Project/Dataset/COVID-19 testing2.csv")
    covid1["Date"]=pd.to_datetime(covid1["Date"], errors='coerce')
    latest=covid[covid["Date"] == "2/27/2021"][["Country","Population","Confirmed","Recovered","Deaths","Active"]]
    confirmed_df = pd.read_csv("D:/kdu/Final Year Project/Dataset/time_series_covid19_confirmed_global.csv")
    death_df = pd.read_csv("D:/kdu/Final Year Project/Dataset/time_series_covid19_deaths_global.csv")
    recovered_df = pd.read_csv("D:/kdu/Final Year Project/Dataset/time_series_covid19_recovered_global.csv")
    return covid, latest, confirmed_df, death_df, recovered_df, covid1
covid, latest, confirmed_df, death_df, recovered_df, covid1 = load_data()



def breakline():
    return st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: red;'> 🦠 Covid-19 Dashboard 🦠</h1>", unsafe_allow_html=True)

confirmed_total = int(latest['Confirmed'].sum())
active_total = int(latest['Active'].sum())
deaths_total = int(latest['Deaths'].sum())
recovered_total = int(latest['Recovered'].sum())

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

#temp = covid2[covid2['Date']==max(covid['Date'])]
#m = folium.Map(location=[0, 0], title="Map for Recovery", min_zoom = 1, max_zoom = 4, zoom_start = 1)
#for i in range(0, len(temp)):    
#    folium.Circle(
#        location = [temp.iloc[i]['Lat'], temp.iloc[i]['Long']], color ='red', fill ='crimson',
#       radius = int(temp.iloc[i]['Confirmed'])**0.05
#        ).add_to(m)   
#generate_map(m)

st.markdown("<h2 style='text-align: center; color: black; background-color:Beige'>View Cases on Map</h2>",
            unsafe_allow_html=True)
fig_con = px.choropleth(covid, locations="Country", 
	                        locationmode='country names', color="Confirmed", 
	                        animation_frame = 'Date', 
                    
	                        title='Countries with Confirmed Cases')

fig_d = px.choropleth(covid, locations="Country", 
	                        locationmode='country names', color="Deaths", 
	                        animation_frame = 'Date', 
	                        title='Countries with Death Cases')
	                       
fig_r = px.choropleth(covid, locations="Country", 
	                        locationmode='country names', color="Recovered", 
	                        animation_frame = 'Date', 
	                        title='Countries with Recovered Cases')

fig_a = px.choropleth(covid, locations="Country", 
	                        locationmode='country names', color="Active", 
	                        animation_frame = 'Date', 
	                        title='Countries with Active Cases')
opt = st.radio(
     "Select option",
     ('Confirmed Cases', 'Recovered Cases','Deaths Cases', 'Active Cases'))

if opt == 'Confirmed Cases':
     st.plotly_chart(fig_con)
elif opt == 'Recovered Cases':
    st.plotly_chart(fig_r)
elif opt == 'Deaths Cases':
    st.plotly_chart(fig_d)
else:
     st.plotly_chart(fig_a)
        
st.header("Select Country")
city = st.selectbox("Country",covid["Country"][:191])

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

#covid_c = latest.iloc[0]['Confirmed']
#covid_d = latest.iloc[0]['Deaths']
#covid_r = latest.iloc[0]['Recovered']

#if st.checkbox("View Bar chart",False):
#    bar = px.bar(latest, x= ['Confirmed', 'Deaths', 'Recovered'], y= ['covid_c', 'covid_d', 'covid_r'], color = ['Confirmed', 'Deaths', 'Recovered'])
 #   st.plotly_chart(bar)
    
def dataf(latest):
    total_d = pd.DataFrame({
        'Status':['Confirmed', 'Deaths', 'Recovered'],
        'Cases':(latest['Confirmed'].iloc[0],
                 latest['Deaths'].iloc[0],
                 latest['Recovered'].iloc[0])})
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
case_d  = go.Line( x = pd.to_datetime(city1["Date"]), y = city1["Recovered"], name = 'TOTAL DEATHS')
case_r  = go.Line(x = pd.to_datetime(city1["Date"]), y = city1["Deaths"], name = 'TOTAL RECOVERED')
    
if case_type == 'Confirmed':
    figc = go.Figure(case_c)  
    figc.update_layout(title = "Confirmed Cases in {}".format(city))
    st.plotly_chart(figc)    
elif case_type == 'Deaths':      
    figd = go.Figure(case_d)  
    figc.update_layout(title = "Deaths Cases in {}".format(city))
    st.plotly_chart(figd)     
elif case_type == 'Recovered':        
    figr = go.Figure(case_r) 
    figc.update_layout(title = "Recovered Cases in {}".format(city))
    st.plotly_chart(figr)
elif case_type == 'Confirmed, Deaths and Recovered':        
    fign = go.Figure([case_c, case_d, case_r])
    fign.update_layout(legend_orientation='h', title = "Confirmed, Deaths and Recovered Cases in {}".format(city))    
    st.plotly_chart(fign)        


st.header(f"View deaths for {city} by Month/Day/Date")    
    
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

        

st.header("View Top 5 Countries : ")
opt_selected = st.selectbox("", options=['Top Confirmed Cases', 'Top Death Cases', 'Top Recovered Cases', 'Death Percentage', 'Confirmed Percentage', 'Recovered Percentage'])
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
elif  opt_selected == 'Confirmed Percentage':
    st.plotly_chart(fig2_cp)
    st.plotly_chart(fig1_cp)
elif  opt_selected == 'Death Percentage':
    st.plotly_chart(fig2_dp)
    st.plotly_chart(fig1_dp)
else:
    st.plotly_chart(fig2_rp)
    st.plotly_chart(fig1_rp)
    
#bar2 = alt.Chart(table_data).mark_bar().encode(
#    x="Confirmed",
#   y=alt.Y("Country",sort="-x"),
#    color=alt.Color("Country"),
 #   tooltip = "Confirmed"
#).interactive()

#fig1.add_trace([fig2])
#st.altair_chart(bar2)

#st.markdown("### View Cases for selected country : ")

#country_s = st.multiselect('Select Countries',covid["Country"][:191])
#View_c = st.selectbox("Select option", ('Confirmed case', 'Death case', 'Recovered case'))

#country2 = covid[covid["Country"].isin(country_s)]

#country_c  = px.line(country2, x = "Date", y = "Confirmed", width= 700, height= 450)
#country_c.update_yaxes(type="linear")


#st.plotly_chart(country_c)

st.header("View the Dataset by Month")

if st.checkbox("Click to View the Dataset",False):
    "Select the Month from Slider"
    nc = st.slider("Month",2,11,2,1)
    covid1 = covid1[covid1["Date"].dt.month ==nc]
    "data", covid1

#options = st.multiselect(
#    'Select Multiple Countries',
#    covid1["Country"][:191])


#fire=alt.Chart(covid1[covid1["Country"].isin(options)],width=500,height=300).mark_line().encode(
#    x="Date",
#    y="Confirmed",
#    tooltip=["Date","Country","Confirmed"],
 #   color="Country",
#    size="Confirmed"
#).interactive()

#bar1 = alt.Chart(covid1[covid1["Country"].isin(options)]).mark_bar().encode(
#    y="sum(New cases)",
#    x=alt.X("Country",sort="-y"),
 #   color="Country",
#    tooltip = "sum(New cases)"
#).interactive()

#st.altair_chart(fire)

st.header("View Covid-19 details by Date")
cty = st.selectbox("Country",covid1["Country"][:191])
ddd = st.date_input(
     "Select the Date",value = datetime.date(2020, 2, 1),min_value = datetime.date(2020, 2, 1), max_value= datetime.date(2020, 2, 27))
st.write('The selected date is:', ddd)
#st.dataframe(covid1.iloc[covid1[covid1["Date"] == ddd][covid1["Country"]==cty]][["Country","Date","New cases"]].reset_index(drop=True))
#st.dataframe(covid1.loc[[covid1[covid1["Date"] == pd.Timestamp(ddd)]["Country"]==cty]][["Country","Date","New cases"]].reset_index(drop=True))
#st.dataframe(covid1[[covid1[covid1["Date"] == ddd]["Country"]==cty]][["Country","Date","New cases"]].reset_index(drop=True))

siti = covid1[covid1['Date'] == pd.Timestamp(ddd)][["Country","Date","Confirmed", "Deaths", "Recovered"]].reset_index(drop=True)
st.dataframe(siti)






