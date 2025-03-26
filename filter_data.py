import pandas as pd

# Read the original CSV file
print("Reading the original CSV file...")
df = pd.read_csv('Amazon Beauty Recommendation system.csv')

# Get initial statistics
print("\nInitial dataset statistics:")
print(f"Total rows: {len(df)}")
print(f"Unique products: {df['ProductId'].nunique()}")
print(f"Unique users: {df['UserId'].nunique()}")

# Filter products with at least 15 ratings
product_rating_counts = df['ProductId'].value_counts()
products_with_min_ratings = product_rating_counts[product_rating_counts >= 15].index
df_filtered = df[df['ProductId'].isin(products_with_min_ratings)]

# Filter users who have given at least 15 ratings
user_rating_counts = df_filtered['UserId'].value_counts()
users_with_min_ratings = user_rating_counts[user_rating_counts >= 15].index
df_filtered = df_filtered[df_filtered['UserId'].isin(users_with_min_ratings)]

# Get final statistics
print("\nFiltered dataset statistics:")
print(f"Total rows: {len(df_filtered)}")
print(f"Unique products: {df_filtered['ProductId'].nunique()}")
print(f"Unique users: {df_filtered['UserId'].nunique()}")

# Save the filtered data
output_file = 'filtered_amazon_beauty.csv'
df_filtered.to_csv(output_file, index=False)
print(f"\nFiltered data saved to {output_file}")

# Print some sample product IDs for testing
print("\nSample product IDs for testing:")
print(df_filtered['ProductId'].sample(5).tolist()) 