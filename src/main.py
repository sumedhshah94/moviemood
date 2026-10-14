from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, ValidationError, validator, root_validator#, field_serializer, model_serializer

from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


from typing import List

from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from sklearn.preprocessing import MinMaxScaler
from redis import asyncio as aioredis

import joblib
import pickle
import numpy as np
import pandas as pd
import json
from pprint import pprint
from io import StringIO
from scipy.stats import boxcox

class Song(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    type: str
    id: str
    uri: str
    track_href: str
    analysis_url: str
    duration_ms: float
    time_signature: int

'''
class Music(BaseModel):
    music_list: List[Song]
'''

class Music(BaseModel):
    music_list: List[List[str]]
    drop_movies: List[dict] | None = None


class Movie(BaseModel):
    omdb_title: str
    genres: str
    omdb_plot: str
    omdb_director:  str
    omdb_actors: str
    imdb_score: float
    omdb_poster: str
    omdb_runtime: str

class Movies(BaseModel):
    movies_list: List[Movie]

app = FastAPI()

#model = joblib.load("trainer/music_to_mood_model.pickle")
loaded_rb = pickle.load(open("trainer/rb_scaler.pkl", "rb"))
loaded_model = pickle.load(open("trainer/music_to_mood_model.pickle", "rb"))

load_movie_data = pd.read_csv('data/cleaned/wiki_movies_mood_label_by_gpt-3.5-turbo_final.csv')
emotions_mapping = {0: 'sad', 1: 'happy', 2:'energetic', 3:'calm'}

def getIndexOfMovie(movie_title, movie_director):
    movie = load_movie_data.loc[(load_movie_data['omdb_title']==movie_title) & (load_movie_data['omdb_director']==movie_director)]
    return movie.index[0]

@cache()
async def get_cache():
    return 1

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://redis:6379", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

@app.get("/", status_code=501)
async def root():
    raise HTTPException(status_code=501)

@app.get("/hello")
async def hello(name: str = ''):
    if name:
        return {"message": "Hello " + name}
    else:
        raise HTTPException(status_code=400, detail="No value passed")
    
@app.get("/health")
async def health():
    today = datetime.now()
    return {"time":today.isoformat()}

@cache(expire=60)
@app.post("/predict")#, response_model=Movies)
async def predict(music: Music):
    movies_to_drop = []
    if music.drop_movies:
        movies_to_drop = [getIndexOfMovie(movie['omdb_title'], movie['omdb_director']) for movie in music.drop_movies]
    arr = music.music_list
    arr[0] = [x.lower() for x in arr[0]]
    df = pd.DataFrame(data = arr[1:], columns=arr[0])
    #csv_text = StringIO(music)
    #df = pd.read_csv(csv_text)
    #jsondata = music.model_dump(mode='json')
    #df = pd.json_normalize(jsondata['music_list'])
    X = df[['danceability','acousticness','energy','instrumentalness','liveness','valence','loudness','speechiness','tempo']]
    X_transformed = loaded_rb.transform(X)
    y_pred = loaded_model.predict(X_transformed)
    df['mood'] = y_pred
    df['mood'] = df['mood'].map(emotions_mapping)
    df[['sad','happy','energetic','calm']] = loaded_model.predict_proba(X_transformed)
    
    # Extract var3 and var4 columns
    var3 = df['energetic']
    var4 = df['calm']
    # Apply Box-Cox transformation
    transformed_var3, lambda_var3 = boxcox(var3 + 1)  # Adding 1 to handle zero values
    transformed_var4, lambda_var4 = boxcox(var4 + 1)
    # Scale the transformed variables to the range [0, 1]
    scaler = MinMaxScaler()
    scaled_var3 = scaler.fit_transform(transformed_var3.reshape(-1, 1))
    scaled_var4 = scaler.fit_transform(transformed_var4.reshape(-1, 1))
    # Update the DataFrame with scaled variables
    df['energetic_scaled'] = scaled_var3
    df['calm_scaled'] = scaled_var4

    user_mood_vector = df[['happy', 'sad', 'energetic_scaled', 'calm_scaled']].mean()
    user_mood_reshaped = user_mood_vector.values.reshape(1, -1)
    movie_df = load_movie_data.drop_duplicates(subset=['title', 'omdb_director'], keep=False)
    movie_mood_vectors = movie_df[['happy','sad','energetic','calm']].values
    similarities = cosine_similarity(user_mood_reshaped, movie_mood_vectors)
    sim_scores = list(enumerate(similarities[0]))
    eligible_movies = sorted(sim_scores, key=lambda x: (x[1], movie_df.loc[x[0], 'rotten_tomatoes_score']), reverse=True)
    top_movies = [i for i in eligible_movies if not pd.isna(movie_df.loc[i[0], 'rotten_tomatoes_score']) or not pd.isna(movie_df.loc[i[0], 'metacritic_score'])]
    # iterate through top_movies and remove a movie if it is in the movies_to_drop
    new_top_movies = []
    for tup in top_movies:
        if tup[0] in movies_to_drop:
            continue
        else:
            new_top_movies.append(tup)
    new_top_movies = new_top_movies[:5]
    top_movie_indices = [i[0] for i in new_top_movies]
    recommended_movies = movie_df.iloc[top_movie_indices]
    recommended_movies = recommended_movies[['omdb_title','genres','omdb_plot','omdb_director', \
                                            'omdb_actors', 'imdb_score', 'omdb_poster', 'omdb_runtime']]
    recommended_movies_json = recommended_movies.to_json(orient='records')

    return {"movies_list":json.loads(recommended_movies_json)}