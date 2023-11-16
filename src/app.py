import time 

import numpy as np  
import pandas as pd  
import plotly.express as px  
import streamlit as st 
import random
import tensorflow as tf

load_model = tf.keras.models.load_model('/Users/hrishikeshhpillai/Documents/hrishi/deeplearning/gridai/forecasting/checkpoints/model_load1_100.h5')
gen_model = tf.keras.models.load_model('/Users/hrishikeshhpillai/Documents/hrishi/deeplearning/gridai/forecasting/checkpoints/model_gen1_10_sigmoid_with_output_sigmoid.h5')




st.set_page_config(
    page_title="Energy Forecasting Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
dataset_url ="../datasets/SolarPrediction2.csv"


# read csv from a URL
# @st.cache_data
# def get_data() -> pd.DataFrame:
#     return pd.read_csv(dataset_url, parse_dates=['datetime'])

# df1 = get_data()

load_X_test = pd.read_csv('/Users/hrishikeshhpillai/Documents/hrishi/deeplearning/gridai/forecasting/checkpoints/TestSets/load_X_test.csv', parse_dates=['Datetime'])
gen_X_test = pd.read_csv('/Users/hrishikeshhpillai/Documents/hrishi/deeplearning/gridai/forecasting/checkpoints/TestSets/gen_X_test.csv', parse_dates=['datetime'])
DEMAND_MAX = 44736.048650000004
RAD_MAX = 1601.26
GRID = 0

messages = ["The demand is completely met by solar generation, relaxation in grid output", "The demand is not completely met by the solar generation, moving output to the grid", "The demand is not completely met by grid and solar generation, initiating steps for load shedding"]

# dashboard title
st.title(":red[GridNet AI]")

# top-level filters
# state = st.selectbox("Select the district", ['Ernankulam', 'Kottayam', 'Trivandrum', 'Kollam', 'Thrissur', 'Palakkad', 'Kozhikode', 'Malappuram', 'Kannur', 'Kasaragod'])

# creating a single-element container
placeholder = st.empty()





i = random.randint(0, len(load_X_test)-7)

#demand
df_load_X = load_X_test[i:i+7]
X_load = df_load_X[['Temperature', 'Humidity', 'WindSpeed', 'powerconsumption']]
X_load = np.array(X_load)
load_pred = load_model.predict(np.reshape(X_load, (1, 7, 4)))
j = random.randint(0, len(gen_X_test)-38-12) #check this - 12 ?
#generation
df_gen_X = gen_X_test[j:j+38]
X_gen = df_gen_X[['Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']]
X_gen = np.array(X_gen)
gen_pred = gen_model.predict(np.reshape(X_gen, (1, 38, 6)))



if (load_pred < gen_pred):
    message_info = 0
elif((load_pred > gen_pred) and ((load_pred - gen_pred) < GRID)):
    message_info = 1
else:
    message_info = 2

# df["Radiation_new"] = df["Radiation"] * np.random.choice(range(1, 5))
# df["Temperature_new"] = df["Temperature"] * np.random.choice(range(1, 5))
# df["Pressure_new"] = df["Pressure"] * np.random.choice(range(1, 5))
# df["Humidity_new"] = df["Humidity"] * np.random.choice(range(1, 5))
# df["WindDirection(Degrees)_new"] = df["WindDirection(Degrees)"] * np.random.choice(range(1, 5))
# df["Speed_new"] = df["Speed"] * np.random.choice(range(1, 5))


avg_temperature_load = np.mean(df_load_X["Temperature"])*40.01
avg_humidity_load = np.mean(df_load_X["Humidity"])*94.8
avg_windspeed_load = np.mean(df_load_X["WindSpeed"])*6.483
# avg_humidity = np.mean(df_load_X["Humidity_new"])
# avg_winddirection = np.mean(df_load_X["WindDirection(Degrees)_new"])
# avg_speed = np.mean(df_load_X["Speed_new"])


avg_temperature_gen = np.mean(df_gen_X["Temperature"])*71
avg_humidity_gen = np.mean(df_gen_X["Humidity"])*103
avg_windspeed_gen = np.mean(df_gen_X["Speed"])*40.5
avg_radiation_gen = np.mean(df_gen_X["Radiation"])*1601.26
avg_pressure_gen = np.mean(df_gen_X["Pressure"])*30.56
avg_wind_gen = np.mean(df_gen_X["WindDirection(Degrees)"])*359.95




with placeholder.container():
    

    st.markdown("## Generation Station Conditions")
    kpi1, kpi2, kpi3 = st.columns(3)


    kpi1.metric(
        label="Temperature",
        value=f"{round(avg_temperature_load)} W/m\u00b2",
        delta=round(avg_temperature_load) - 10, ## change this as datafranes inputs i+7 th value - i+6th value
    )

    kpi2.metric(
        label="Humidity",
        value=f"{round(avg_humidity_load)} °F",
        delta=round(avg_humidity_load) - 10,
    )

    kpi3.metric(
        label="Windspeed",
        value=f"{round(avg_windspeed_load)} Pa",
        delta=round(avg_windspeed_load) - 10,
    )


    st.markdown("## Demand Area Conditions")
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)


    kpi1.metric(
        label="Temperature",
        value=f"{round(avg_temperature_gen)} °F",
        delta=round(avg_temperature_gen) - 10,
    )

    kpi2.metric(
        label="Humidity",
        value=f"{round(avg_humidity_gen)} °F",
        delta=round(avg_humidity_gen) - 10,
    )

    kpi3.metric(
        label="Windspeed",
        value=f"{round(avg_windspeed_gen)} ",
        delta=round(avg_windspeed_gen) - 10,
    )

    kpi4.metric(
        label="Radiation",
        value=f"{round(avg_radiation_gen)} W/m\u00b2",
        delta=round(avg_radiation_gen) - 10,
    )

    kpi5.metric(
        label="Pressure",
        value=f"{round(avg_pressure_gen)} Pa",
        delta=round(avg_pressure_gen) - 10,
    )
    kpi6.metric(
        label="WindDirection(Degrees)",
        value=f"{round(avg_wind_gen)} °",
        delta=round(avg_wind_gen) - 10,
    )



    st.button("Update Values")

    st.markdown("## Forecasted Values")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Forecasted Generation")
        st.title(f":red[{round((gen_pred[0][0])*RAD_MAX)}]")

        


    with c2:

        st.markdown("#### Forecasted Load")

        st.title(f":green[{round((load_pred[0][0])*DEMAND_MAX)}]")

    st.markdown("## Insights")

    

    st.text_area(
        label="### Radiation",
        value=f"{messages[message_info]}"
    )
   

    # create two columns for charts
    fig_col1, fig_col2 = st.columns(2)
   
    with fig_col1:
        st.markdown("### Radiation Chart")
        fig = px.line(
            data_frame=gen_X_test, y="Radiation", x="datetime"
        )
        st.write(fig)
        
    #time series chart for temperature

    with fig_col2:
        st.markdown("### Temperature Chart")
        fig = px.line(
            data_frame=gen_X_test, y="Temperature", x="datetime"
        )
        st.write(fig)

    

    st.markdown("### Detailed Data View")
    df_col1, df_col2 = st.columns(2)
    with df_col1:
        st.dataframe(gen_X_test)
    with df_col2:
        st.dataframe(load_X_test)

    # st.button("Load Shedding")

    
    
    time.sleep(1)
