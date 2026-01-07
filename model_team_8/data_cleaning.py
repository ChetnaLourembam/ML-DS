import pandas as pd
# import numpy as np

df = pd.read_csv("data_prep\data_full.csv")

## organize the table in order of date
df = df.sort_values(by=["Date"])

df["Date"] = pd.to_datetime(df["Date"])

df["month"] = df["Date"].dt.month
df["day_of_week"] = df["Date"].dt.dayofweek
print(df.info())
# reduce the table between the working dates 01-07-2013 to 31-07-2019
start_date = "2013-07-01"
end_date = "2019-07-31"
mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
df = df.loc[mask]
print(df.info())

### create a new column with daily total sales for all products
df["daily_total_sales"] = df.groupby("Date")["Sales Volume"].transform("sum")
df.head()

print(df.head(1))
print(df.tail(1))

# code for specific columns handling missing values
### check for missing values on all columns
print(df.isna().sum())
# ### handling missing values in Temperature column ###
# # 1) ensure datetime
# df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# # 2) month-day key (ignores year)
# df["month_day"] = df["Date"].dt.strftime("%m-%d")

# # 3) mean temp per month-day across all years
# md_mean_temp = df.groupby("month_day")["Temperature"].mean()

# # 4) fill missing Temperature using that mean
# df["Temperature"] = df["Temperature"].fillna(df["month_day"].map(md_mean_temp))
# ### check for missing values again ###
# print(df.isna().sum())


def fill_by_same_day_mean(df, date_col, cols, fallback="month"):
    """
    Fill NaNs in `cols` by mean of same month-day across years.

    df: dataframe
    date_col: name of date column
    cols: list of column names to fill
    fallback: None | "month" | "overall"
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # key ignoring year - add to dataframe for proper alignment
    out["month_day"] = out[date_col].dt.strftime("%m-%d")

    for c in cols:
        md_mean = out.groupby("month_day")[c].mean()
        out[c] = out[c].fillna(out["month_day"].map(md_mean))

        if fallback == "month":
            out["month_key"] = out[date_col].dt.month
            m_mean = out.groupby("month_key")[c].mean()
            out[c] = out[c].fillna(out["month_key"].map(m_mean))
            out.drop("month_key", axis=1, inplace=True)
        elif fallback == "overall":
            out[c] = out[c].fillna(out[c].mean())

    # Remove temporary column
    out.drop("month_day", axis=1, inplace=True)
    return out


df = fill_by_same_day_mean(
    df,
    date_col="Date",
    cols=[
        "Temperature",
        "Wind Speed",
        "Cloud Cover",
        "Temperature_H",
        "Wind Speed_H",
        "Cloud Cover_H",
        "Precipitation_H",
    ],
    fallback="month",  # optional
)

### check for missing values on all columns
print(df.isna().sum())

### data_org has all columns + date features from 01-07-2013 to 31-07-2019
df.to_csv("data_prep\data_org.csv", index=False)
