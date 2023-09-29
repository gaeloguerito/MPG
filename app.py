import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import base64

# Charger les données
@st.cache_resource
def load_data(file):
    df = pd.read_excel(file, skiprows=2)
    df.set_index("Joueur", inplace=True)
    st.write(f"Nombre de lignes après le chargement des données : {len(df)}")
    return df

# Nettoyer les données
def preprocess_data(df):
    columns_to_keep = ["Cote", "Var cote", "Enchère moy", "Note M11", "Nb match", "But", "%Titu", "Temps", "Tps moy", "Min/But", "Min note/but", "Prix/but", "j6", "j5", "j4", "j3", "j2", "j1", "j38", "j37", "j36", "j35", "j34", "j33", "j32", "j31", "j30", "j29", "j28", "j27", "j26", "j25", "j24", "j23", "j22", "j21", "j20", "j19", "j18", "j17", "j16", "j15", "j14", "j13","j12","j11","j10","j9","j8","j7","Cleansheet","But/Peno","But/Coup-franc","But/surface","Pass decis.","Occas° créée","Tirs","Tirs cadrés","Corner gagné","%Passes","Ballons","Interceptions","Tacles","%Duel","Fautes","But évité","Action stoppée","Poss Def","Poss Mil","Centres","Centres ratés","Dégagements","But concédé","Ballon perdu","Passe>Tir","Passe perfo","Dépossédé","Plonge&stop","Nb matchs gagnés","Erreur>But","Diff de buts","Grosse occas manquée","Balle non rattrapée","Bonus moy","Malus moy","Index MPGStats","DMI"]
    df = df[columns_to_keep]

    # Remplacer les valeurs nulles
    df["Cote"].fillna(1, inplace=True)
    df["Enchère moy"].fillna(df["Cote"], inplace=True)
    df["DMI"].fillna(5, inplace=True)
    df.fillna(0, inplace=True)

    # Créer une colonne de moyenne roulante
    rolling_mean = df.iloc[:, 14:18].mean(axis=1)
    df["Moyenne roulante"] = rolling_mean

    
    st.write(f"Nombre de lignes après le prétraitement : {len(df)}")
    return df

# Fonction pour arrondir à l'entier le plus proche ou à .5
def round_half(number):
    return np.round(number * 2) / 2

# Entraîner le modèle
def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, 13]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Créer une instance de StandardScaler
    scaler = StandardScaler()

    # Appliquer la mise à l'échelle aux colonnes de caractéristiques
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Créer et entraîner le modèle XGBoost sur l'ensemble d'entraînement
    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Prédire la prochaine note pour tous les exemples du jeu de données complet
    next_match_predictions = model.predict(X)

    st.write(f"Nombre de prédictions : {len(next_match_predictions)}")

    # Arrondir les prédictions et ajouter à une nouvelle colonne dans le DataFrame original
    df["Prochaine note"] = np.vectorize(round_half)(next_match_predictions)

    return df

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('MPG.png')



def main():
    st.title("Prédiction de la note MPG")

    uploaded_file = st.sidebar.file_uploader("Uploader votre fichier Excel (.xlsx) obtenu de https://www.mpgstats.fr/", type=["xlsx"])

    if uploaded_file is not None:
        df_input = load_data(uploaded_file)
        df_input = preprocess_data(df_input)
        df_with_predictions = train_model(df_input)

        # Afficher uniquement les colonnes "Joueur" et "Prochaine note"
        st.write("Données avec la colonne 'Prochaine note':")
        st.dataframe(df_with_predictions[["Prochaine note"]])

        # Ajouter un filtre pour le nom des joueurs
        player_filter = st.text_input("Filtrer par nom de joueur")
        if player_filter:
            filtered_data = df_with_predictions[df_with_predictions.index.str.contains(player_filter)]
            st.dataframe(filtered_data[["Prochaine note"]])

if __name__ == "__main__":
    main()