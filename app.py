from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Global variable to store the model
item_similarity_df = None

def create_model():
    global item_similarity_df
    # Create model from scratch using filtered dataset
    df = pd.read_csv('filtered_amazon_beauty.csv')
    
    # Create pivot table
    pivot_table = df.pivot_table(
        index='UserId', 
        columns='ProductId', 
        values='Rating', 
        aggfunc='mean', 
        fill_value=0
    )
    
    # Calculate similarity matrix
    item_similarity = cosine_similarity(pivot_table.T)
    item_similarity_df = pd.DataFrame(
        item_similarity, 
        index=pivot_table.columns, 
        columns=pivot_table.columns
    )
    
    print(f"Model created with {len(item_similarity_df)} products")
    return item_similarity_df

def get_similar_items(product_id, top_n=5):
    global item_similarity_df
    if item_similarity_df is None:
        item_similarity_df = create_model()
    
    if product_id not in item_similarity_df.index:
        return None
    similar_items = item_similarity_df[product_id].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_items

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    product_id = request.form['product_id']
    similar_items = get_similar_items(product_id)
    
    if similar_items is None:
        return render_template('result.html', 
                             error="Product ID not found. Please try another product ID.",
                             product_id=product_id)
    
    recommendations = []
    for idx, similarity in similar_items.items():
        recommendations.append({
            'product_id': idx,
            'rating': round(similarity, 2)
        })
    
    return render_template('result.html', 
                         recommendations=recommendations,
                         product_id=product_id)

if __name__ == '__main__':
    app.run(debug=True) 