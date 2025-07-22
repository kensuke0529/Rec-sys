import os
import pickle
import ast

from fastapi import FastAPI, HTTPException, Query
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util

# --- Model class ---

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_items, n_genres, embedding_dim=32):
        super(RecSysModel, self).__init__()
        self.user_embed = nn.Embedding(n_users, embedding_dim)
        self.item_embed = nn.Embedding(n_items, embedding_dim)

        self.extra_features = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.genre_features = nn.Sequential(
            nn.Linear(n_genres, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 32 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, user, item, vote_avg, count, popularity, genre):
        user_emb = self.user_embed(user)
        item_emb = self.item_embed(item)
        x_embed = torch.cat([user_emb, item_emb], dim=1)

        x_num = torch.cat([vote_avg.unsqueeze(1), count.unsqueeze(1), popularity.unsqueeze(1)], dim=1)
        x_num = self.extra_features(x_num)

        x_genre = self.genre_features(genre)
        x = torch.cat([x_embed, x_num, x_genre], dim=1)

        return self.mlp(x).squeeze(1)


# --- Utility to load pickle files ---

def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[Error] Loading {filepath}: {e}")
        raise


# --- Paths and loading encoders/scaler ---

base_dir = os.path.dirname(os.path.abspath(__file__))

user_enc = load_pickle(os.path.join(base_dir, '../model/user_enc.pkl'))
item_enc = load_pickle(os.path.join(base_dir, '../model/item_enc.pkl'))
scaler = load_pickle(os.path.join(base_dir, '../model/scaler.pkl'))
genre2idx = load_pickle(os.path.join(base_dir, '../model/genre2idx.pkl'))

# --- Load movie_df and preprocess ---

movie_df = pd.read_parquet(os.path.join(base_dir, '../data/movie_data.parquet'))

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []

movie_df['genres_list'] = movie_df['genres'].apply(parse_genres)

def multi_hot_encode(genres, genre2idx):
    vector = [0] * len(genre2idx)
    for g in genres:
        if g in genre2idx:
            vector[genre2idx[g]] = 1
    return vector

movie_df['genre_vector'] = movie_df['genres_list'].apply(lambda g: multi_hot_encode(g, genre2idx))
movie_df = movie_df.drop_duplicates(subset='title')

scaled_vals = scaler.transform(movie_df[['vote_average', 'vote_count', 'popularity']])
movie_df[['vote_avg', 'vote_cnt', 'popularity_scaled']] = scaled_vals

movie_df['title_enc'] = item_enc.transform(movie_df['title'])

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)
n_genres = len(genre2idx)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RecSysModel(n_users, n_items, n_genres)
model_path = os.path.join(base_dir, '../model/best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

corr = pd.read_parquet('../data/item_based_corr_full.parquet')
df = pd.read_parquet('../data/movie_data_processed.parquet')

# --- Item based recommendation function ---

def item_based_rec(title, top_n=9):
    try:
        if title not in corr.columns:
            raise ValueError(f"Title '{title}' not found in correlation matrix.")

        similar_scores = corr[title].drop(title)

        similar_df = pd.DataFrame({
            'title': similar_scores.index,
            'similarity': similar_scores.values
        })

        movie_info = df[['title', 'vote_count', 'vote_average', 'poster_path']].drop_duplicates(subset='title')
        similar_df = similar_df.merge(movie_info, on='title', how='left')

        filtered_df = similar_df[similar_df['vote_count'] > 150]
        topn_df = filtered_df.sort_values(by='similarity', ascending=False).head(top_n).reset_index(drop=True)

        return topn_df

    except Exception as e:
        print(f"[ERROR] item_based_rec failed for title '{title}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Content-based recommendation setup ---

df1 = df[['title','overview','genres','poster_path','vote_average','vote_count']].copy()
df1['genres'] = df1['genres'].astype(str)
df1 = df1.drop_duplicates().reset_index(drop=True)

def load_movie_embeddings(path):
    movie_embeddings_np = np.load(path).astype(np.float32)
    return torch.from_numpy(movie_embeddings_np)

movie_embeddings = load_movie_embeddings('../model/movie_embeddings.npy')
movie_embeddings = movie_embeddings.to(device)

embedding_model = SentenceTransformer('all-mpnet-base-v2')

# Check alignment between embeddings and df1 on startup
if movie_embeddings.shape[0] != len(df1):
    raise RuntimeError(f"[FATAL] movie_embeddings length {movie_embeddings.shape[0]} != df1 rows {len(df1)}. "
                       "Regenerate embeddings to match the dataset.")

def recommend_movies(prompt, top_k=10, vote_threshold=150):
    try:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt is empty")

        query_embedding = embedding_model.encode(prompt, convert_to_tensor=True).to(device)

        cos_scores = util.cos_sim(query_embedding, movie_embeddings)[0]
        top_results = torch.topk(cos_scores, k=len(cos_scores))
        indices = top_results.indices.cpu().numpy()
        scores = top_results.values.cpu().numpy()

        recommended = df1.iloc[indices].copy()
        recommended['similarity'] = scores

        recommended = (
            recommended.drop_duplicates(subset='title')
            .query('vote_count > @vote_threshold')
            .sort_values(by='similarity', ascending=False)
            .reset_index(drop=True)
        )

        if recommended.empty:
            raise ValueError("No recommendations found after filtering.")

        return recommended[['title', 'similarity', 'overview', 'vote_average', 'vote_count', 'poster_path']].head(top_k)

    except Exception as e:
        print(f"[ERROR] recommend_movies failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")


# --- FastAPI app and routes ---

app = FastAPI()

@app.get("/user_recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10):
    if user_id not in user_enc.classes_:
        raise HTTPException(status_code=404, detail="User not found")

    user_idx = user_enc.transform([user_id])[0]
    item_indices = torch.tensor(movie_df['title_enc'].values, dtype=torch.long).to(device)
    user_indices = torch.tensor([user_idx] * len(movie_df), dtype=torch.long).to(device)

    vote_avg_tensor = torch.tensor(movie_df['vote_avg'].values, dtype=torch.float32).to(device)
    count_tensor = torch.tensor(movie_df['vote_cnt'].values, dtype=torch.float32).to(device)
    popularity_tensor = torch.tensor(movie_df['popularity_scaled'].values, dtype=torch.float32).to(device)
    genre_tensor = torch.tensor(np.stack(movie_df['genre_vector'].values), dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(user_indices, item_indices, vote_avg_tensor, count_tensor, popularity_tensor, genre_tensor)
        preds = preds.cpu().numpy()

    movie_df['predicted_rating'] = preds
    filtered_df = movie_df[(movie_df['vote_count'] > 200) & (movie_df['vote_average'] > 5)]

    top_recommendations = filtered_df.sort_values('predicted_rating', ascending=False).head(top_k)

    return {
        "user_id": user_id,
        "recommendations": [
            {
                'title': row['title'],
                'predicted_rating': float(row['predicted_rating']),
                'vote_average': float(row['vote_average']),
                'vote_count': int(row['vote_count']),
                'popularity': float(row['popularity']),
                'genres': row['genres_list']
            }
            for _, row in top_recommendations.iterrows()
        ]
    }


@app.get("/title-or-prompt")
def combined_recommendation(input_text: str = Query(..., description="Movie title or search prompt")):
    input_text_clean = input_text.strip()
    if not input_text_clean:
        raise HTTPException(status_code=400, detail="Input text is empty")
    try:
        if input_text_clean in df['title'].values:
            print(f"[INFO] Item-based rec for title: '{input_text_clean}'")
            result = item_based_rec(input_text_clean)
            print(f"[INFO] Item-based rec result count: {len(result)}")
            return result.to_dict(orient='records')  # Return as JSON serializable dict
        else:
            print(f"[INFO] Content-based rec for prompt: '{input_text_clean}'")
            result = recommend_movies(input_text_clean)
            print(f"[INFO] Content-based rec result count: {len(result)}")
            return result.to_dict(orient='records')
    except Exception as e:
        print(f"[ERROR] combined_recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

