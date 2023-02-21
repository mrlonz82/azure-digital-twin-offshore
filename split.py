import pandas as pd
import datetime

df = pd.read_csv("power.csv")

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])


q1_min = datetime.datetime(2018, 1, 1, 0, 10)
q1_max = datetime.datetime(2018, 3, 31, 23, 50)

q2_min = datetime.datetime(2018, 4, 1, 0, 0)
q2_max = datetime.datetime(2018, 6, 30, 23, 50)

q3_min = datetime.datetime(2018, 7, 1, 0, 0)
q3_max = datetime.datetime(2018, 9, 28, 21, 20)

q4_min = datetime.datetime(2018, 10, 2, 16, 30)
q4_max = datetime.datetime(2018, 12, 31, 23, 50)

# extract for Q1
q1_df = pd.DataFrame(data=df.loc[df["Date"] <= q1_max])
q1_df.to_csv("power/Q1.csv", index=False)

df = pd.DataFrame(data=df.loc[df["Date"] >= q1_max])

# extract for Q2
q2_df = pd.DataFrame(data=df.loc[df["Date"] <= q2_max])
q2_df.to_csv("power/Q2.csv", index=False)

df = pd.DataFrame(data=df.loc[df["Date"] >= q2_max])

# extract for Q3
q3_df = pd.DataFrame(data=df.loc[df["Date"] <= q3_max])
q3_df.to_csv("power/Q3.csv", index=False)

df = pd.DataFrame(data=df.loc[df["Date"] >= q3_max])

# extract for Q4
q4_df = pd.DataFrame(data=df.loc[df["Date"] <= q4_max])
q4_df.to_csv("power/Q4.csv", index=False)

