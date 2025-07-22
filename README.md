# Movie Recommendation System

## Overview

This project implements a hybrid recommendation system combining:

- **PyTorch-based Collaborative Filtering model:**  
  Learns user and item embeddings and incorporates side information (genres as multi-hot vectors, and scaled movie metadata like vote averages and popularity) to predict user ratings.

- **LLM-powered Content-based Recommendation:**  
  Uses SentenceTransformer embeddings (`all-mpnet-base-v2`) on movie overviews to perform semantic similarity search for arbitrary text prompts.

These two components are integrated via a FastAPI service providing user-based, item-based, and content-based recommendations.

---

## Architecture

### PyTorch Recommender Model

- Embedding layers for users and movies.  
- Numerical features (scaled vote average, vote count, popularity) processed via MLP.  
- Genre multi-hot vectors processed via separate MLP.  
- Concatenation of embeddings + numerical + genre features into final MLP to predict ratings.  
- Trained with MSE loss and optimized with Adam.

### LLM Content-based Search

- SentenceTransformer model generates fixed-size embeddings of movie overviews.  
- Query embedding generated on-the-fly for input prompts.  
- Cosine similarity ranking of query against all movie embeddings to find top semantically related movies.

---

## API Endpoints

- `/user_recommend/{user_id}`  
  Personalized recommendations based on the PyTorch collaborative filtering model.

- `/title-or-prompt?input_text=...`  
  Returns:  
  - Item-based collaborative filtering recommendations if `input_text` matches a movie title.  
  - Otherwise, returns semantic content-based recommendations using the LLM embeddings.

---
### File description
####  1. Item_based_rec.ipynb

- An item-based CF system: find movies similar to any single movie.

- A multi-item weighted CF system: aggregate across multiple rated movies to generate personalized recommendations.

#### 2. TF-IDF.ipynb

- Suggest movies based on the movies' overview + genres and user prompt using TF-IDF

#### 3. SentenceTransformer.ipynb

- Suggest movies based on the movies' overview + genres and user prompt using HuggingFace (A pretrained MPNet-based sentence embedding model)

#### 4. PyTorch.ipynb

- A semantic movie recommendation system using pytorch

#### 5. /app/app.py
- Deployed by FastAPI
https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjQxc2FkdXIyNm12NWZrcjE0eWlwcTAzNzRocXBjb3p2OThqYzV3NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Lfa4lEFfGbgOubOCem/giphy.gif

----