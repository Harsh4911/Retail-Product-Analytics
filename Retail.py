import pandas as pd
import numpy as np

print("--- Starting Project 4: Retail E-commerce Analytics ---")

# 1. Dono saal ke dataset ko load aur combine karna
print("Loading 2009-2010 data...")
df_09_10 = pd.read_csv('online_retail_09_10.csv')

print("Loading 2010-2011 data...")
df_10_11 = pd.read_csv('online_retail_10_11.csv')

# Dono dataframes ko ek ke neeche ek jodna (Concatenate)
print("Merging datasets...")
df = pd.concat([df_09_10, df_10_11], ignore_index=True)
print(f"Total rows after merging: {len(df)}")

# 2. Data Cleaning
print("\nCleaning data...")

# Missing CustomerIDs wale rows ko hatana (kyunki unka cohort ya RFM nahi ban sakta)
df = df.dropna(subset=['CustomerID'])

# Data types sahi karna
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Quantity aur UnitPrice string format se numeric check karna aur safe rakhna
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# Cancelled orders ko identify karna (Invoice numbers starting with 'C')
df['Is_Cancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')

# Sahi orders filter karna (Quantity > 0 aur Price > 0 aur jo cancelled nahi hain)
df_cleaned = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['Is_Cancelled'] == False)].copy()

# Total Sales (Revenue) column banana
df_cleaned['Total_Spend'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

print("\n--- Initial Cleaning Summary ---")
print(f"Cleaned Dataset Rows: {len(df_cleaned)}")
print(f"Unique Customers: {df_cleaned['CustomerID'].nunique()}")
print(f"Unique Products Sold: {df_cleaned['StockCode'].nunique()}")
print(f"Total Revenue Generated: ${df_cleaned['Total_Spend'].sum():,.2f}")

# Cleaned data ko save kar lena takki aage isi se processing ho
df_cleaned.to_csv('online_retail_cleaned.csv', index=False)
print("\nCleaned data successfully saved as 'online_retail_cleaned.csv'")



import pandas as pd
import numpy as np

print("--- Starting Step 2: RFM Generation & Scoring ---")

# 1. Cleaned data ko load karna aur dates ko parse karna
df_09_10 = pd.read_csv('online_retail_09_10.csv')
df_10_11 = pd.read_csv('online_retail_10_11.csv')
df = pd.concat([df_09_10, df_10_11], ignore_index=True)

# Quick background cleaning to sync numbers
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df['Is_Cancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
df_cleaned = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['Is_Cancelled'] == False)].copy()
df_cleaned['Total_Spend'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# 2. Base Date set karna Recency calculate karne ke liye (Dataset ke aakhri date ke 1 din baad ki date)
snapshot_date = df_cleaned['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Snapshot Date for Recency calculation: {snapshot_date.strftime('%Y-%m-%d')}")

# 3. Aggregating RFM Metrics per Customer
rfm = df_cleaned.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Total_Spend', 'sum')
).reset_index()

print("\nRFM Base Metrics Calculated. Scoring Now...")

# 4. Scoring (1 se 5 ka score dena quintiles ke basis par)
# Recency mein jitne kam din honge, utna achha score hoga (so labels=[5,4,3,2,1])
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])

# Frequency aur Monetary mein jitna bada number, utna achha score (so labels=[1,2,3,4,5])
# rank(method='first') handle karta hai duplicate values ko quintiles mein break karne ke liye
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

# Combined Score String banana (e.g., '555' ya '111')
rfm['RFM_Cell'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# 5. Core Segments Define Karna Based on R and F Scores
def assign_segment(row):
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    
    if r >= 4 and f >= 4:
        return 'Champions / Power Users'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'Recent / New Customers'
    elif r <= 2 and f >= 4:
        return 'Cant Lose Them / High-Value At Risk'
    elif r <= 2 and f <= 2:
        return 'Lost / Hibernating'
    else:
        return 'Regular / Needs Attention'

rfm['Customer_Segment'] = rfm.apply(assign_segment, axis=1)

# Summary of Segments
segment_summary = rfm.groupby('Customer_Segment').agg(
    Customer_Count=('CustomerID', 'count'),
    Avg_Recency_Days=('Recency', 'mean'),
    Avg_Frequency_Orders=('Frequency', 'mean'),
    Avg_Total_Spend=('Monetary', 'mean')
).round(1).reset_index()

print("\n" + "="*50)
print(" RETAIL CUSTOMER RFM SEGMENT SUMMARY ")
print("="*50)
print(segment_summary.to_string(index=False))

# Export processing for next stages
rfm.to_csv('online_retail_rfm_segments.csv', index=False)
print("\nRFM Segments exported successfully to 'online_retail_rfm_segments.csv'")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- Starting Step 3: Visualizing RFM Segments ---")

# 1. Saved RFM data ko load karna
rfm = pd.read_csv('online_retail_rfm_segments.csv')

# Aggregation for plots
plot_data = rfm.groupby('Customer_Segment').agg(
    Total_Customers=('CustomerID', 'count'),
    Total_Revenue=('Monetary', 'sum')
).reset_index().sort_values(by='Total_Revenue', ascending=False)

# 2. Charts ki Plotting
plt.figure(figsize=(15, 6))

# Chart 1: Customer Count by Segment
plt.subplot(1, 2, 1)
sns.barplot(
    data=plot_data, 
    y='Customer_Segment', 
    x='Total_Customers', 
    hue='Customer_Segment',
    palette='viridis', 
    legend=False
)
plt.title('Customer Count by Strategic Segment', fontsize=13, fontweight='bold')
plt.xlabel('Number of Customers')
plt.ylabel('')

# Chart 2: Total Revenue Share by Segment
plt.subplot(1, 2, 2)
sns.barplot(
    data=plot_data, 
    y='Customer_Segment', 
    x='Total_Revenue', 
    hue='Customer_Segment',
    palette='magma', 
    legend=False
)
plt.title('Total Revenue Contribution ($) by Segment', fontsize=13, fontweight='bold')
plt.xlabel('Total Revenue Generated ($)')
plt.ylabel('')

plt.tight_layout()
plt.show()

print("\nCharts displayed successfully. Ready for the next core data component!")


import pandas as pd
from itertools import combinations
from collections import Counter

print("--- Starting Step 4: Market Basket Analysis & Product Association ---")

# 1. Base Data Load karna (To check transactions)
df_09_10 = pd.read_csv('online_retail_09_10.csv')
df_10_11 = pd.read_csv('online_retail_10_11.csv')
df = pd.concat([df_09_10, df_10_11], ignore_index=True)

# Basic cleaning optimization
df = df.dropna(subset=['CustomerID', 'Description'])
df['Is_Cancelled'] = df['InvoiceNo'].astype(str).str.startswith('C')
df_valid = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['Is_Cancelled'] == False)]

print("Grouping items by Invoice to find baskets...")
# Har invoice par bikne wale unique items ki list banana
baskets = df_valid.groupby('InvoiceNo')['Description'].apply(lambda x: list(set(x))).reset_index()

# Sirf woh invoices rakhna jisme 1 se zyada items khareede gaye hon (Bundles banane ke liye)
baskets = baskets[baskets['Description'].transform(len) > 1]

# 2. Item Pairs Generate karna aur unka count nikalna
print("Generating item combinations and counting pairs...")
pair_counter = Counter()

for items in baskets['Description']:
    # Har basket ke items ko alphabetically sort karke combinations banana takki duplicate pairs na banein
    items.sort()
    for pair in combinations(items, 2):
        pair_counter[pair] += 1

# Dataframe banana results ka
pairs_df = pd.DataFrame(pair_counter.most_common(15), columns=['Product_Pair', 'Transaction_Count'])

# Total transactions count calculate karna
total_baskets = len(baskets)

# 3. Association Metrics Calculate karna
pairs_df['Item_A'] = pairs_df['Product_Pair'].apply(lambda x: x[0])
pairs_df['Item_B'] = pairs_df['Product_Pair'].apply(lambda x: x[1])

# Support nikalna
pairs_df['Support (%)'] = ((pairs_df['Transaction_Count'] / total_baskets) * 100).round(2)

print("\n" + "="*70)
print(" TOP 10 RETAIL PRODUCT BUNDLES / ASSOCIATIONS (MARKET BASKET) ")
print("="*70)
print(pairs_df[['Item_A', 'Item_B', 'Transaction_Count', 'Support (%)']].head(10).to_string(index=False))

# Excel/Power BI dashboard ke liye final associations save karna
pairs_df.to_csv('market_basket_recommendations.csv', index=False)
print("\nMarket Basket recommendations saved as 'market_basket_recommendations.csv'")
