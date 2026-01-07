import os
import glob
import pandas as pd
from datetime import timedelta


def copy_raw_to_work():
    """
    Loads each CSV from raw_data folder and saves it individually
    into work_data folder.
    """

    input_folder = "data_raw"
    output_folder = "data_prep"

    # make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # find all CSV files in raw_data
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in {input_folder}/")

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        # keep the original filename
        filename = os.path.basename(file_path)
        save_path = os.path.join(output_folder, filename)

        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    print("\nAll files copied to work_data!")


# copy_raw_to_work()

###     Load train and test datasets     ###

train = pd.read_csv("data_prep/train.csv")
test = pd.read_csv("data_prep/test.csv")
train.rename(
    columns={"Datum": "Date", "Warengruppe": "Product Group", "Umsatz": "Sales Volume"},
    inplace=True,
)
test.rename(columns={"Datum": "Date", "Warengruppe": "Product Group"}, inplace=True)
print(train.head(), train.min(), train.max())
print(train.info())
print(test.head(), test.min(), test.max())
print(test.info())

####     Check for overlapping IDs  ####

# len(set(train) & set(test))
print(f"Number of overlapping IDs: {len(set(train['id']) & set(test['id']))}")
# check for overlapping columns
overlapping_columns = set(train.columns) & set(test.columns) - {"id"}
print(f"Overlapping columns (excluding 'id'): {overlapping_columns}")
print(f"Number of overlapping dates:  {len(set(train['Date']) & set(test['Date']))}")

####    weather data    ####

wetter_og = pd.read_csv("data_prep\wetter.csv")
wetter_og.rename(
    columns={
        "Datum": "Date",
        "Bewoelkung": "Cloud Cover",
        "Temperatur": "Temperature",
        "Windgeschwindigkeit": "Wind Speed",
        "Wettercode": "Weather Code",
    },
    inplace=True,
)
print(wetter_og.head(), wetter_og.min(0), wetter_og.max(0))
print(wetter_og.info())
wetter_holt = pd.read_csv("data_prep\weather_holtenau.csv")
wetter_holt.rename(
    columns={
        "Temperature": "Temperature_H",
        "Wind Speed": "Wind Speed_H",
        "Cloud Cover": "Cloud Cover_H",
        "Precipitation": "Precipitation_H",
    },
    inplace=True,
)
print(wetter_holt.head(), wetter_holt.min(0), wetter_holt.max(0))
print(wetter_holt.info())

####   Kiwo data   ###
kiwo_data = pd.read_csv("data_prep/kiwo.csv")
kiwo_data.rename(columns={"Datum": "Date"}, inplace=True)
print(kiwo_data.head(), kiwo_data.min(0), kiwo_data.max(0))
print(kiwo_data.info())

### holiday data ###
holidays = pd.read_csv("data_prep/kiel_holidays.csv")
print(holidays.head(), holidays.min(0), holidays.max(0))
print(holidays.info())


# Function to calculate Easter Sunday using Meeus/Jones/Butcher algorithm
def calculate_easter(year):
    """Calculate Easter Sunday for a given year"""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1

    return pd.Timestamp(year=year, month=month, day=day)


# Function to get Good Friday
def get_good_friday(year):
    """Get Good Friday (2 days before Easter Sunday)"""
    easter = calculate_easter(year)
    good_friday = easter - timedelta(days=2)
    return good_friday


# Load existing holidays
holidays = pd.read_csv("data_prep/kiel_holidays.csv")
print("Current holidays shape:", holidays.shape)
print("\nCurrent holidays:")
print(holidays.head(10))

# Determine year range from your data
# You can adjust these based on your analysis period
start_year = 2013
end_year = 2019

# Generate Good Friday dates
good_fridays = []
for year in range(start_year, end_year + 1):
    gf_date = get_good_friday(year)
    good_fridays.append(
        {
            "Date": gf_date.strftime("%d.%m.%Y"),  # Format to match your existing data
            "Day of Week": gf_date.strftime("%A"),
            "Holiday Name (English)": "Good Friday",
            "Holiday Name (German)": "Karfreitag",
        }
    )

good_fridays_df = pd.DataFrame(good_fridays)
print("\n\nGood Friday dates to add:")
print(good_fridays_df)

# Check if Good Friday already exists
existing_good_fridays = holidays[
    holidays["Holiday Name (English)"].str.contains("Good Friday", na=False)
]
print(f"\n\nExisting Good Friday entries: {len(existing_good_fridays)}")

# Combine with existing holidays
if len(existing_good_fridays) == 0:
    holidays_updated = pd.concat([holidays, good_fridays_df], ignore_index=True)

    # Sort by date
    holidays_updated["Date_temp"] = pd.to_datetime(
        holidays_updated["Date"], format="%d.%m.%Y"
    )
    holidays_updated = holidays_updated.sort_values("Date_temp")
    holidays_updated = holidays_updated.drop("Date_temp", axis=1)

    print(f"\n\nUpdated holidays shape: {holidays_updated.shape}")
    print("\nUpdated holidays (showing all):")
    print(holidays_updated)

    # Save updated holidays
    holidays_updated.to_csv("data_prep/kiel_holidays.csv", index=False)
    print("\n✓ Updated kiel_holidays.csv saved!")

    # Reload the saved file to use for merging
    holidays = pd.read_csv("data_prep/kiel_holidays.csv")

    # Verify the save
    print(f"\nVerification - Total holidays: {len(holidays)}")
    print(
        f"Good Friday entries: {holidays['Holiday Name (English)'].str.contains('Good Friday', na=False).sum()}"
    )
else:
    print("Good Friday already exists in the dataset. Skipping...")


# Standardize date format for holidays (DD.MM.YYYY -> YYYY-MM-DD)
holidays["Date"] = pd.to_datetime(holidays["Date"], format="%d.%m.%Y").dt.strftime(
    "%Y-%m-%d"
)
print(f"Converted holidays Date format. Sample: {holidays['Date'].head()}")

####   Merging train and test datasets   ####

# Merge train and test by id
data = pd.concat([train, test], ignore_index=True)

# DO NOT add is_closed - just keep original data
data.to_csv("data_prep\data.csv", index=False)
print(data.head, data.info())

####    Merging weather data     ####
data_weather = pd.merge(data, wetter_og, on="Date", how="left")
data_weather = pd.merge(data_weather, wetter_holt, on="Date", how="left")
print(data_weather.head(), data_weather.min(), data_weather.max())
print(data_weather.info())

data_weather_kiwo = pd.merge(data_weather, kiwo_data, on="Date", how="left")
# Fill NaN values in KielerWoche with 0 (0 = not Kieler Woche, 1 = is Kieler Woche)
data_weather_kiwo["KielerWoche"] = (
    data_weather_kiwo["KielerWoche"].fillna(0).astype(int)
)
print(data_weather_kiwo.head())
print(data_weather_kiwo.info())
print(f"KielerWoche values: {data_weather_kiwo['KielerWoche'].value_counts()}")

data_weather_kiwo.to_csv("data_prep\data_weather_kiwo.csv", index=False)
print("Merged data saved to data_weather_kiwo.csv")

####   Merging holiday data   ####
print(f"\nBefore holiday merge - data shape: {data_weather_kiwo.shape}")
print(f"Unique dates in data: {data_weather_kiwo['Date'].nunique()}")
print(f"Unique dates in holidays: {holidays['Date'].nunique()}")
print(f"Sample dates from data: {data_weather_kiwo['Date'].head().tolist()}")
print(f"Sample dates from holidays: {holidays['Date'].head().tolist()}")

all_dataset = pd.merge(data_weather_kiwo, holidays, on="Date", how="left")

# Drop the German holiday name column (keep only English)
if "Holiday Name (German)" in all_dataset.columns:
    all_dataset = all_dataset.drop(columns=["Holiday Name (German)"])
    print("Dropped 'Holiday Name (German)' column")

# Drop Day of Week column if it exists (will be created in data_cleaning.py)
if "Day of Week" in all_dataset.columns:
    all_dataset = all_dataset.drop(columns=["Day of Week"])

print(f"\nAfter holiday merge - data shape: {all_dataset.shape}")
print(f"Holiday columns non-null counts:")
print(f"  Holiday Name (English): {all_dataset['Holiday Name (English)'].notna().sum()}")

print(all_dataset.head(20))
print(all_dataset.info())
all_dataset.to_csv("data_prep\data_full.csv", index=False)
print("\ndata_full.csv saved successfully!")
