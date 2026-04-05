import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("amz_ca_total_products_data_processed.csv")

# -------------------------------
# 1. FIX DATA TYPES
# -------------------------------

df['stars'] = df['stars'].astype(float)
df['reviews'] = df['reviews'].astype(int)
df['price'] = df['price'].astype(float)
df['listPrice'] = df['listPrice'].astype(float)
df['boughtInLastMonth'] = df['boughtInLastMonth'].astype(int)

# Convert boolean → int (for ML)
df['isBestSeller'] = df['isBestSeller'].astype(int)

# -------------------------------
# 2. HANDLE MISSING VALUES
# -------------------------------

# Replace 0 listPrice with NaN (invalid value)
df['listPrice'].replace(0, np.nan, inplace=True)

# Fill missing listPrice with price
df['listPrice'].fillna(df['price'], inplace=True)

# Fill missing ratings/reviews with safe defaults
df['stars'].fillna(df['stars'].mean(), inplace=True)
df['reviews'].fillna(0, inplace=True)

# -------------------------------
# 3. CLEAN TEXT DATA
# -------------------------------

# Remove extra spaces in category
df['categoryName'] = df['categoryName'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Clean title (optional but good)
df['title'] = df['title'].str.strip()

# -------------------------------
# 4. REMOVE DUPLICATES
# -------------------------------

df.drop_duplicates(subset='asin', inplace=True)

# -------------------------------
# 5. FEATURE ENGINEERING
# -------------------------------

# Discount percentage
df['discount_percent'] = ((df['listPrice'] - df['price']) / df['listPrice']) * 100

# Popularity score
df['popularity_score'] = df['reviews'] * df['stars']

# Price difference
df['price_diff'] = df['listPrice'] - df['price']

# -------------------------------
# 6. OPTIONAL: DROP UNUSED COLUMNS
# -------------------------------

df.drop(columns=['imgUrl', 'productURL'], inplace=True)

# -------------------------------
# 7. FINAL CHECK
# -------------------------------

print(df.info())
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_amazon_data.csv", index=False)