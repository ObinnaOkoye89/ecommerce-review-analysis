## Loand and Clean dataset

import os
import pandas as pd

# Load the dataset
reviews = pd.read_csv("womens_clothing_e-commerce_reviews.csv")

# Filter out rows where 'Review Text' is null
valid_reviews = reviews[reviews['Review Text'].notnull()].copy()

# Extract the review texts into a list
texts = valid_reviews['Review Text'].tolist()


## Create and Store Embeddings

import openai
import numpy as np

# Initialize the OpenAI API key
openai_api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = openai_api_key

def get_embeddings(texts, batch_size=100):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        # Create embeddings for a batch of texts
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=texts[i:i+batch_size]
        )
        embeddings.extend([record['embedding'] for record in response['data']])
    return embeddings

# Generate embeddings for all review texts
embeddings = get_embeddings(texts)


## Dimensionality Reduction and Visualization

import umap.umap_ as umap
import matplotlib.pyplot as plt

# Perform dimensionality reduction to 2D using UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot the 2D visualization
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7)
plt.title("2D Visualization of Review Embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.show()


## Feedback Categorization

categories = ['quality', 'fit', 'style', 'comfort']
categorized_reviews = {}

for category in categories:
    # Find reviews that contain the category keyword (case-insensitive)
    matches = valid_reviews[valid_reviews['Review Text'].str.contains(category, case=False)]
    categorized_reviews[category] = matches['Review Text'].head(3).tolist()  # Top 3 examples

# Print the categorized reviews for inspection
print(categorized_reviews)


## Similarity Search Function

from sklearn.metrics.pairwise import cosine_similarity

def find_similar_reviews(input_text, texts, embeddings, top_k=3):
    # Get embedding for input text
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=[input_text]
    )
    input_embedding = response['data'][0]['embedding']
    
    # Compute cosine similarities between the input embedding and all review embeddings
    sims = cosine_similarity([input_embedding], embeddings)[0]
    # Retrieve indices of the top k most similar reviews
    top_indices = sims.argsort()[-top_k:][::-1]
    
    # Return the corresponding review texts
    return [texts[i] for i in top_indices]

# Apply the similarity search function to an example review
input_review = "Absolutely wonderful - silky and sexy and comfortable"
most_similar_reviews = find_similar_reviews(input_review, texts, embeddings, top_k=3)

# Output the most similar reviews
print("Most similar reviews:")
for review in most_similar_reviews:
    print("- ", review)

