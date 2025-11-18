import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############ LOADING DATASETS ##########################
sales = pd.read_csv("umsatzdaten_gekuerzt.csv")         
weather = pd.read_csv("wetter.csv")     
kieler = pd.read_csv("kiwo.csv")      



############ CONVERT DATUM INTO DATETIME ##################
sales['Datum'] = pd.to_datetime(sales['Datum'])
weather['Datum'] = pd.to_datetime(weather['Datum'])
kieler['Datum'] = pd.to_datetime(kieler['Datum'])



############# MERGING DATA ##################
# Merge sales + weather
merged = pd.merge(sales, weather, on='Datum', how='left')
# Merge Kieler Woche
merged = pd.merge(merged, kieler, on='Datum', how='left')



############# FILLING MISSING KIELER WOCHE WITH 0 ####################
merged['KielerWoche'] = merged['KielerWoche'].fillna(0)



############ DESCRIPTIVE STATISTICS ###########################
# Basic descriptive statistics for numeric columns
print(merged.describe())

# statistics for Umsatz (sales)
print("Sales (Umsatz) stats:")
print(merged['Umsatz'].describe())

# statistics for KielerWoche (0/1)
print("Kieler Woche frequency:")
print(merged['KielerWoche'].value_counts())


# mean sales on Kieler Woche vs non-Kieler Woche
print("Average sales on Kieler Woche vs non-Kieler Woche:")
print(merged.groupby('KielerWoche')['Umsatz'].mean())
# Calculate average sales by Kieler Woche
avg_sales = merged.groupby('KielerWoche')['Umsatz'].mean().reset_index()
# Replace 0/1 with labels for better readability
avg_sales['KielerWoche'] = avg_sales['KielerWoche'].map({0: 'No', 1: 'Yes'})
# Plot
sns.barplot(x='KielerWoche', y='Umsatz', data=avg_sales, palette='pastel')
plt.ylabel("Average Sales (€)")
plt.xlabel("Kieler Woche")
plt.title("Average Sales: Kieler Woche vs Non-Kieler Woche")
plt.show()




# Scatter plot: Temperature vs Sales
sns.scatterplot(x='Temperatur', y='Umsatz', data=merged)
plt.title("Sales vs Temperature")
plt.show()

# Scatter plot: Cloudiness vs Sales
sns.scatterplot(x='Bewoelkung', y='Umsatz', data=merged)
plt.title("Sales vs Cloudiness")
plt.show()

# Scatter plot: Wind speed vs Sales
sns.scatterplot(x='Windgeschwindigkeit', y='Umsatz', data=merged)
plt.title("Sales vs Wind Speed")
plt.show()
