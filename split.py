import pandas as pd
import datetime

df = pd.read_csv("datasets/offshore wind farm data.csv")

# Convert the 'Date' column to datetime format
df['date'] = pd.to_datetime(df['date'])


q1_min = datetime.datetime(2010, 1, 1, 0, 0)
q1_max = datetime.datetime(2010, 3, 31, 7, 0)

q2_min = datetime.datetime(2010, 1, 1, 0, 0)
q2_max = datetime.datetime(2010, 6, 24, 3, 50)

q3_min = datetime.datetime(2010, 7, 20, 7, 10)
q3_max = datetime.datetime(2010, 9, 30, 19, 30)

q4_min = datetime.datetime(2010, 10, 1, 00, 00)
q4_max = datetime.datetime(2010, 12, 11, 9, 30)

# extract for Q1
q1_df = pd.DataFrame(data=df.loc[df["date"] <= q1_max])
q1_df.to_csv("splits/2010/Q1.csv", index=False)

df = pd.DataFrame(data=df.loc[df["date"] >= q1_max])

# extract for Q2
q2_df = pd.DataFrame(data=df.loc[df["date"] <= q2_max])
q2_df.to_csv("splits/2010/Q2.csv", index=False)

df = pd.DataFrame(data=df.loc[df["date"] >= q2_max])

# extract for Q3
q3_df = pd.DataFrame(data=df.loc[df["date"] <= q3_max])
q3_df.to_csv("splits/2010/Q3.csv", index=False)

df = pd.DataFrame(data=df.loc[df["date"] >= q3_max])

# extract for Q4
q4_df = pd.DataFrame(data=df.loc[df["date"] <= q4_max])
q4_df.to_csv("splits/2010/Q4.csv", index=False)

