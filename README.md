# 🎬 Movie Recommendation System

**Live Demo (currently working on):** []

**Video Demo:** 

![Link to a short Loom/YouTube video showcasing the API calls and responses](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjQxc2FkdXIyNm12NWZrcjE0eWlwcTAzNzRocXBjb3p2OThqYzV3NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Lfa4lEFfGbgOubOCem/giphy.gif)

## Overview

This project develops a sophisticated **hybrid movie recommendation system** designed to enhance user experience by providing personalized and relevant movie suggestions. It leverages a combination of **collaborative filtering**, **content-based filtering (semantic search)**, and **deep learning techniques**, all deployed via a **FastAPI microservice**.

**Key Features:**

* **Personalized User-Based Recommendations:** Utilizes a custom PyTorch deep learning model to predict user ratings based on their past interactions and rich movie features.
* **Semantic Content-Based Search:** Allows users to find movies based on natural language prompts, leveraging state-of-the-art `SentenceTransformer` embeddings (MPNet-based).
* **Item-to-Item Recommendations:** Provides suggestions for similar movies based on item-level correlations, useful for "more like this" functionality.
* **Robust Data Preprocessing:** Handling and transforming raw movie metadata (genres, overviews, numerical features) for machine learning models.
* **API Deployment:** Packaged as a FastAPI service for easy integration and scalability, exposing various recommendation endpoints.

## Problem Solved / Business Value

This system aims to:
* **Improve User Engagement:** By suggesting highly relevant movies, keeping users on the platform longer.
* **Enhance Content Discovery:** Helping users find new movies based on other users' behavior and prompt, going beyond traditional static recommendations by incorporating interaction and behavioral influence.
* **Showcase Advanced ML/Deployment Skills:** Provides a comprehensive example of building, training, and deploying a complex machine learning system.

## Technical Architecture

This project is built around three core recommendation approaches, integrated into a unified API:

### 1. PyTorch-based Collaborative Filtering Model (for Personalized User Recommendations)
* **Objective:** Predict a user's rating for a given movie based on their historical preferences and movie attributes.
* **Model:** A custom neural network built with PyTorch.
    * **Input Features:**
        * **User Embeddings:** Learned latent representations for each unique user ID (`nn.Embedding`).
        * **Movie Embeddings:** Learned latent representations for each unique movie title (`nn.Embedding`).
        * **Numerical Features:** Scaled `vote_average`, `vote_count`, and `popularity` (processed through a small MLP).
        * **Genre Features:** Multi-hot encoded genre vectors (processed through a separate MLP).
    * **Architecture:** Concatenates user and movie embeddings with processed numerical and genre features, feeding them into a multi-layered perceptron (MLP) for final rating prediction.
    * **Training:** Optimized using Adam with Mean Squared Error (MSE) loss. Achieved **RMSE of 0.87 and MAE of 0.67** on the validation set. (Rating is from 0.5 to 5.0)

### 2. LLM-Powered Content-Based Recommendation (for Semantic Search)
* **Objective:** Recommend movies based on semantic similarity to a user's free-form text query (e.g., "bleak dystopian sci-fi").
* **Technique:** Leverages pre-trained transformer models for natural language understanding.
* **Model:** `SentenceTransformer` model (`all-mpnet-base-v2`) to generate dense, fixed-size **embeddings** for movie overviews and genres.
* **Similarity:** Cosine similarity is used to find the most semantically similar movie embeddings to the input query embedding.

### 3. Item-Based Collaborative Filtering (for "More Like This")
* **Objective:** Suggest movies similar to a given movie based on co-ratings from users.
* **Technique:** Calculates Pearson correlation coefficients between movie rating vectors.
* **Filtering:** Recommendations are filtered to ensure high `vote_count` and `vote_average` for quality.

## 🔗 API Endpoints

The FastAPI service exposes the following endpoints:

* **`GET /user_recommend/{user_id}?top_k={number}`**
    * **Description:** Provides personalized movie recommendations for a specific user ID using the PyTorch collaborative filtering model.
    * **Example Request:** `curl "http://localhost:8000/user_recommend/209?top_k=5"`
    * **Example Response (JSON):**
        ```json
      {
        "user_id": 123,
        "recommendations": [
          {
            "title": "The Godfather",
            "predicted_rating": 4.245197772979736,
            "vote_average": 8.5,
            "vote_count": 6024,
            "popularity": 41.109264,
            "genres": [
              "Drama",
              "Crime"
            ]
          },
          ... more recommendations
        ```

* **`GET /title-or-prompt?input_text={query}`**
    * **Description:** A versatile endpoint that intelligently switches between recommendation types:
        * If `input_text` exactly matches an existing movie title, it returns **item-based collaborative filtering** recommendations.
        * Otherwise, it performs a **semantic content-based search** using the LLM embeddings to find movies related to the text prompt.
    * **Example Request (Title Match):** `curl "http://127.0.0.1:8000/title-or-prompt?input_text=Star%20Wars"`
    * **Example Request (Semantic Prompt):** `curl "http://127.0.0.1:8000/title-or-prompt?input_text=bleak%20dystopian%20sci-fi"`
    * **Example Response (JSON - Semantic Prompt):**
        ```json
        [
          {
            "title": "Return of the Jedi",
            "similarity": 0.7477742283527742,
            "vote_count": 4763,
            "vote_average": 7.9,
            "poster_path": "/jx5p0aHlbPXqe3AH9G15NvmWaqQ.jpg"
          },
          // ... more recommendations
        ]
        ```

## 🚀 Getting Started

This section will guide you through setting up and running the FastAPI application locally to interact with the movie recommendation system. **No GPU is required to run the API,** as all machine learning models and embeddings are pre-trained and included in the repository.

1.  **Clone the repository:**
    Begin by cloning the project from GitHub to your local machine:
    ```bash
    git clone https://github.com/kensuke0529/Rec-sys.git
    cd Rec-
    sys
    ```

2.  **Create and activate a Python virtual environment:**
    It's recommended to use a virtual environment to manage project dependencies cleanly.
    ```bash
    python -m venv venv_recommender
    source venv_recommender/bin/activate  # On Windows: .\venv_recommender\Scripts\activate
    ```

3.  **Install dependencies:**
    Install all required Python libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the FastAPI application:**
    Navigate to the `app` directory and start the FastAPI server using Uvicorn. The API automatically loads all necessary pre-processed data, trained models, and embeddings from the `data/` and `model/` directories.
    ```bash
    cd app
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will now be running locally. You can access the interactive API documentation (Swagger UI) in your web browser at:
    `http://localhost:8000/docs`


