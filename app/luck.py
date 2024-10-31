import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import json

# Define function to prepare data for Prophet
def prepare_prophet_dataframe(json_data, y_option='blue', red_position=None):
    dates = []
    y_values = []

    for item in json_data:
        # Extract year and day from the issue number
        year = int('20' + item['issue'][:2])
        day = int(item['issue'][2:]) * 2

        # Calculate the date
        date = datetime(year, 1, 1) + timedelta(days=day - 1)
        
        # Calculate y based on the chosen option
        if y_option == 'blue':
            y = item['blue']
        elif y_option == 'red':
            if red_position is None:
                raise ValueError("red_position must be specified for 'red' option.")
            y = item['reds'][red_position - 1]
        else:
            raise ValueError("Invalid y_option. Choose 'blue' or 'red' with a specific red_position.")

        dates.append(date.strftime('%Y-%m-%d'))
        y_values.append(y)

    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': y_values
    })

    # Sort DataFrame by date in ascending order
    df = df.sort_values('ds').reset_index(drop=True)

    return df

# Read data from JSON file
with open('smalldata.json', 'r') as file:
    json_data = json.load(file)

# Prepare data for blue ball prediction
prophet_df_blue = prepare_prophet_dataframe(json_data, y_option='blue')

# Train and predict for blue ball with optimized parameters
m_blue = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False) # type: ignore
m_blue.fit(prophet_df_blue)
future_blue = m_blue.make_future_dataframe(periods=1)
forecast_blue = m_blue.predict(future_blue)

# Print blue ball prediction results
print("Blue Ball Prediction Results:")
print(forecast_blue[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1))

# Prepare and predict for each red ball position
for i in range(1, 7):
    # Prepare data for red ball position i
    prophet_df_red = prepare_prophet_dataframe(json_data, y_option='red', red_position=i)

    # Train and predict for red ball position i with optimized parameters
    m_red =  Prophet()
    m_red.fit(prophet_df_red)
    future_red = m_red.make_future_dataframe(periods=1)
    forecast_red = m_red.predict(future_red)

    # Print red ball position i prediction results
    print(f"\nRed Ball Position {i} Prediction Results:")
    print(forecast_red[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1))
 

#  {"issue":"24124","reds":[2,14,15,17,25,30],"blue":11},
# {"issue":"24123","reds":[2,15,22,26,30,33],"blue":4},,  4 - 10 - 14 - 18 - 24 - 28 - 8
# 1 - 4 - 8 - 12 - 17 - 23 - 2   9 - 16 - 21 - 25 - 30 - 33 - 14    2 - 8 - 14 - 4/1/9/10/4/16
# 5 - 10 - 15 - 21 - 25 - 29 - 8   1 - 5 - 8 - 13 - 19 - 24 - 3   10 - 16 - 22 - 26 - 32 - 33 - 14