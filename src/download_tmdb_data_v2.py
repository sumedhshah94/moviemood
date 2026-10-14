"""
Pull movie metadata from the https://www.themoviedb.org API.
Requires an API key stored in a .config file
The code is currently restricted to the movie category. To get it to run with
other categories, update the constants
    (CATEGORY_SPECIFIC_CALLS, JSON_COLUMNS, KEYS_TO_DROP)
and delete the movie specific section of the export_data() function.
Please see attribution requirements here before posting data:
    https://www.themoviedb.org/faq/api
API documentation:
    https://developers.themoviedb.org/3/getting-started
    https://www.themoviedb.org/documentation/api
"""

# 1st Run started at ~ 2:30 PM
# 2nd Run started at ~ 1:00 AM (next day) and downloaded 91,000 movies in 10 hrs 30 min
import gzip
import json
import os
import sys
import pandas as pd
import requests
from datetime import date
import itertools
import numpy as np

from io import BytesIO
from time import sleep


BASE_API_CALL = 'https://api.themoviedb.org/3/{category}/{entry_id}?api_key={api_key}{category_specifics}'
CATEGORIES = ['movie']
DOWNLOADS_PER_DISK_WRITE = 40
MAX_DOWNLOADS_PER_SECOND = 4
MAX_ATTEMPTS = 3
RATE_LIMITER_DELAY_SECONDS = 10
RATE_LIMIT_EXCEEDED_STATUS_CODE = 429
SUCCESSFUL_CALL_STATUS_CODE = 200

CATEGORY_SPECIFIC_CALLS = {
    'movie': '&append_to_response=credits,keywords',
                            }

JSON_COLUMNS = {
    'genres',
    'keywords',
    'production_countries',
    'production_companies',
    'spoken_languages',
    'overview',
    'adult',
    'backdrop_path',
    'belongs_to_collection',
    'imdb_id',
    'poster_path',
    'video'
}

KEYS_TO_DROP = {}


def was_successful(response):
    return response.status_code == SUCCESSFUL_CALL_STATUS_CODE


def was_rate_limited(response):
    return response.status_code == RATE_LIMIT_EXCEEDED_STATUS_CODE


def make_request(call_url, prior_attempts=0):
    if prior_attempts >= MAX_ATTEMPTS:
        return None
    response = requests.get(call_url)
    if was_rate_limited(response):
        sleep(RATE_LIMITER_DELAY_SECONDS)
    sleep(1 / MAX_DOWNLOADS_PER_SECOND)
    if was_successful(response):
        return response.json()
    else:
        sleep(1)  # attempt to sleep through any intermittent issues
        return make_request(call_url, prior_attempts + 1)


def make_detail_request(category, entry_id):
    category_specifics = ''
    if category in CATEGORY_SPECIFIC_CALLS:
        category_specifics = CATEGORY_SPECIFIC_CALLS[category]
    call_url = BASE_API_CALL.format(
        category=category,
        entry_id=entry_id,
        api_key=API_KEY,
        category_specifics=category_specifics,
                                    )
    return make_request(call_url)


def load_api_key():
    return json.load(open('.config'))['api_key']


def make_category_id_url_suffix(category, extension='json'):
    year = str(pd.to_datetime(date.today()).today().year)
    print(year)
    month = str(pd.to_datetime(date.today()).today().month).zfill(2)
    print(month)
    day = str(pd.to_datetime(date.today()).today().day - 1).zfill(2)
    print(day)
    #return '_'.join([category, 'ids', month, day, year]) + '.' + extension
    return '_'.join([category, 'ids', '10', '01', year]) + '.' + extension

def download_id_list_as_csv(category):
    # see daily file export list docs at:
    # https://developers.themoviedb.org/3/getting-started/daily-file-exports
    print(f'Downloading list of ids for {category}')
    id_list_name = make_category_id_url_suffix(category)
    ID_LISTS_RAW_URL = 'http://files.tmdb.org/p/exports/{0}.gz'.format(id_list_name)
    print(ID_LISTS_RAW_URL)
    with gzip.open(BytesIO(requests.get(ID_LISTS_RAW_URL).content), 'r') as f_open:
        id_list = f_open.readlines()
    # original 'json' is malformed, is actually one dict per line
    ids = pd.DataFrame([json.loads(x) for x in id_list])
    # some entries in the movie id list appear to be collections rather than movies
    if 'original_title' in ids.columns:
        ids.original_title = ids.original_title.apply(str)
        ids = ids[~ids.original_title.str.endswith(' Collection')].copy()
    # You have to drop adult films if you want to post any new data to Kaggle.
    if 'adult' in ids.columns:
        ids = ids[~ids['adult']].copy()
    ids.to_csv(category + '_ids.csv', index=False)


def load_id_list(category):
    if not os.path.exists(category + '_ids.csv'):
        download_id_list_as_csv(category)
    df = pd.read_csv(category + '_ids.csv')
    print(len(df.id.values.tolist()))
    return df.id.values.tolist()


def unpack_credits(df):
    # credits were downloaded with the movie details to cut down on the
    # total number of requests, but it should probably be stored separately
    credits = pd.DataFrame(df[['credits', 'id', 'title']])
    credits.rename(columns={'id': 'movie_id'}, inplace=True)
    new_columns = ['cast', 'crew']
    for column in new_columns:
        try:
            credits[column] = credits['credits'].apply(
                lambda x: x[column] if column in x else [])
        except Exception as e:
            continue
        credits[column] = credits[column].apply(lambda x:
            [{k: v for k, v in i.items() if k not in {'profile_path'}} for i in x])
        credits[column] = credits[column].apply(json.dumps)
    del credits['credits']
    del df['credits']
    return df, credits

"""    try:
        s.encode(encoding='utf-8').decode('ascii')
        d = enchant.Dict("en_US")
        sentence = s.split(" ")
        for letter in sentence:
            if letter:
                if not d.check(letter):
                    return False

    except Exception as e:
        return False
    else:
        return True"""


def export_data(category, all_entries):
    if not all_entries:
        return None
    df = pd.DataFrame(all_entries)
    df = df[[x for x in df.columns if x not in KEYS_TO_DROP]].copy()
    if len(df[df.id.isnull()]) > 0:
        print(f'Dropping {len(df[df.id.isnull()])} entries without ids')
        df = df[~df.id.isnull()]
    df = df[df.id.apply(lambda x: str(x).isnumeric())]
    # this section about credits is specific to the movie category
    df, credits = unpack_credits(df)
    if not df['keywords'].empty:
        try:
            df['keywords'] = df['keywords'].apply(lambda x:
                x['keywords'] if 'keywords' in x else [])
        except Exception as e:
            pass
    for column in JSON_COLUMNS:
        df[column] = df[column].apply(json.dumps)
    needs_header = not(os.path.exists(category + '_data.csv'))
    df.to_csv(category + '_data.csv', index=False, mode='a+', header=needs_header)
    credits.to_csv(category + '_credits.csv', index=False, mode='a+', header=needs_header)


def download_ids(category, id_list):
    #if os.path.exists('movie_data.csv'):
    if os.path.exists(category + '_data.csv'):
        print("PATH EXISTS... exiting...")
        print("length of total ids: " + str(len(id_list)))
        existing_ids = pd.read_csv(category + '_data.csv')
        existing_ids['id'] = pd.to_numeric(existing_ids['id'])
        print("length of existing ids: " + str(len(existing_ids)))
        existing_ids = set(existing_ids.id.values.tolist())
        id_set = set(id_list)
        remaining_ids = id_set.symmetric_difference(existing_ids)
        print("length of remaining: " + str(len(remaining_ids)))
        id_list = list(remaining_ids)
    counter = 0
    all_entries = []
    print(f'Downloading details for {category}')
    for movie_id in id_list:
        current_data = make_detail_request(category, movie_id)
        if not current_data:
            print(f'Failed on id # {movie_id}')
            continue
        counter += 1
        all_entries.append(current_data)
        if counter % DOWNLOADS_PER_DISK_WRITE == 0:
            print(f'Finished downloading {counter} entries for {category}')
            export_data(category, all_entries)
            all_entries = []
    export_data(category, all_entries)


def download_all_data():
    for category in CATEGORIES:
        download_ids(category, load_id_list(category))


if __name__ == '__main__':
    API_KEY = load_api_key()
    download_all_data()