import numpy as np
import pandas as pd

reviews_df = pd.read_csv('user_reviews.csv')
# remove any rows with missing values
reviews_df = reviews_df.drop(['Unnamed: 0'])
# ensure first column is userId
reviews_df = reviews_df.rename(columns={reviews_df.columns[0]: 'userId'})
# convert dataframe from wide to long
reviews = reviews_df.melt(id_vars=['userId'], var_name='movieId', value_name='rating')
all_reviews = reviews.copy()

print(reviews.head(5))

train_reviews = reviews.dropna(subset=['rating'])
train_reviews = train_reviews[train_reviews['rating'] != 0].copy()

train_reviews['userId'] = train_reviews['userId'].astype(str)
train_reviews['movieId'] = train_reviews['movieId'].astype(str)
train_reviews['rating'] = train_reviews['rating'].astype(float)

from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(train_reviews[['userId','movieId','rating']], reader)

# split data into train and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# use SVD algorithm for collaborative filtering
algo = SVD()

# train the algorithm on the trainset
algo.fit(trainset)

# predict ratings on the testset
predictions = algo.test(testset)

from surprise import rmse

# Evaluate performance using RMSE
print("RMSE: ", rmse(predictions))

def rec_5_movies(user_id):
    user_id = str(user_id)

    # get all movie names
    all_movies = all_reviews['movieId'].unique()
    
    # movies that have been rated by the user (score must be 1-5)
    rated_movies = reviews_df[reviews_df['userId'] == user_id]['movieId'].values

     # unseen candidates are those that exist in all_movies but not seen
    candidates = [m for m in all_movies if m not in seen]
    preds = [algo.predict(user_id, m) for m in candidates]
    preds_sorted = sorted(preds, key=lambda p: p.est, reverse=True)
    # return DataFrame with movieId and predicted rating
    top = preds_sorted[:n]
    return pd.DataFrame([(p.iid, p.est) for p in top], columns=['movieId','predicted_rating'])

# --- 6) Test for the specific users ---
test_users = ['Vincent', 'Edgar', 'Addilyn', 'Marlee', 'Javier']
for u in test_users:
    if str(u) not in all_reviews['userId'].unique():
        print(f"User {u} not found in reviews.csv — skipping")
        continue
    recs = rec_5_movies(u)
    print(f"\nTop 5 recommended movies for user {u}:")
    print(recs)