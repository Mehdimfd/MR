from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

app = Flask(__name__)

 #Load data
permalink_url = "https://grouplens.org/datasets/movielens/latest/"
def telecharger_et_stocker(url, nom_fichier):
    try:
        # Effectuer une requête GET pour obtenir le contenu du fichier
        reponse = requests.get(url)

        # Vérifier si la requête a réussi (statut code 200)
        if reponse.status_code == 200:
            # Écrire le contenu dans un fichier local
            with open(nom_fichier, 'wb') as fichier_local:
                fichier_local.write(reponse.content)
            print(f"Le fichier {nom_fichier} a été téléchargé et stocké avec succès.")
        else:
            print(f"Impossible de télécharger le fichier. Code de statut : {reponse.status_code}")
    except requests.RequestException as e:
        print(f"Une erreur s'est produite : {e}")

# Télécharger et stocker le fichier ratings.csv
telecharger_et_stocker(permalink_url + "/ratings.csv", "chemin/vers/le/dossier/ratings.csv")

# Télécharger et stocker le fichier movies.csv
telecharger_et_stocker(permalink_url + "/movies.csv", "chemin/vers/le/dossier/movies.csv")
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")


def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

movies["clean_title"] = movies["title"].apply(clean_title)


vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])


def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        movie_title = request.form["movie_title"]
        if len(movie_title) > 5:
            results = search(movie_title)
            movie_id = results.iloc[0]["movieId"]
            similar_movies = find_similar_movies(movie_id)
            return render_template("index.html", movies=similar_movies.to_dict(orient="records"))
    return render_template("index.html", movies=[])

if __name__ == "__main__":
    app.run(debug=True)
