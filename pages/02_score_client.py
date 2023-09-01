import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgbm
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import streamlit.components.v1 as components # pour affichage explainer LIME
from lime import lime_tabular
import joblib
from PIL import Image

# Configuration de la page 
###########################
st.set_page_config(
        page_title='Score du Client',
        page_icon = "🎱",
        layout="wide" )


# Centrage de l'image du logo dans la sidebar
#############################################
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('logo p_a_d.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")


# Lecture des jeux de données 
##############################

# décorateur mise en cache de la fonction pour exécution unique
################################################################
@st.cache_data 
def load_X_test_raw():
    X_test_raw_data_path = Path() / 'data/X_test_raw.csv'
    X_test_raw = pd.read_csv(X_test_raw_data_path)
    return X_test_raw


# décorateur mise en cache de la fonction pour exécution unique
################################################################
@st.cache_data 
def load_X_test_mod():
    X_test_mod_data_path=Path() / 'data/X_test_mod.csv'
    X_test_mod = pd.read_csv(X_test_mod_data_path)
    return X_test_mod



if __name__ == "__main__":

    load_X_test_raw()
    load_X_test_mod()

# Evaluation de la demande
##########################
    st.markdown("""
                <p style="color:#772b58;text-align:center;font-size:2.10em;font-style:;font-weight:600;font-family:'helvetica Condensed';margin:0px;">
                Evaluation de la demande du client </h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    
# Création liste déroulante identifiant client
##############################################
    liste_ID = list(load_X_test_raw()['sk_id_curr'])
    col1, col2,col3 = st.columns(3) # division de la largeur de la page en 3 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Sélectionnez ou entrez l'identifiant du client*", 
                                (liste_ID))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")
        st.write("")
    with col3:
        st.write("")    

    
# Lecture du modèle de prédiction et des scores 
###############################################
model_LGBM_path=Path() / 'modeles/LGBM5_saved.joblib'
model_LGBM = joblib.load(open(model_LGBM_path,'rb'))
target_pred = model_LGBM.predict(load_X_test_raw().drop(labels="sk_id_curr", axis=1))

# Prédiction de la classe 0 ou 1
target_proba = model_LGBM.predict_proba(load_X_test_raw().drop(labels="sk_id_curr", axis=1))# Prédiction du % de risque
#st.write(load_X_test_raw().loc[load_X_test_raw()["sk_id_curr"]==ID_client])
#st.write(target_proba[4])



#score du client
##################
df_target_proba = pd.DataFrame(target_proba, columns=['classe_0_proba', 'classe_1_proba'])
df_target_proba = pd.concat([df_target_proba['classe_1_proba'],load_X_test_raw()['sk_id_curr']], axis=1)

score = df_target_proba[df_target_proba['sk_id_curr']==ID_client]
score_value = round(score.classe_1_proba.iloc[0]*100, 2)


# Récupération de la prédiction et mise en forme de la décision
###############################################################
df_target_pred = pd.DataFrame(target_pred, columns=['predictions'])
df_target_pred = pd.concat([df_target_pred, load_X_test_raw()['sk_id_curr']], axis=1)
df_target_pred['client'] = np.where(df_target_pred.predictions == 1, "insolvable", "solvable")
df_target_pred['decision'] = np.where(df_target_pred.predictions == 1, "refuser", "accorder")
solvabilite = df_target_pred.loc[df_target_pred['sk_id_curr']==ID_client, "client"].values
decision = df_target_pred.loc[df_target_pred['sk_id_curr']==ID_client, "decision"].values

# graphique de gauge pour afficher le score 
############################################
col1, col2 = st.columns(2)
with col2:
    st.markdown(""" <br> <br> """, unsafe_allow_html=True)
    st.write(f"Le client dont l'identifiant est **{ID_client}** a  un score de **{score_value:.1f}%**.")
    st.write(f"**Il y a un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
    st.write(f"Selon ces critères , le client est **{solvabilite[0]}** \
                et la décision de lui **{decision[0]}** le crédit doit lui être communiquée. ")

# graphique de gauge pour affichage du score 
############################################
with col1:
    fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 30, 'decreasing': {'color': "green"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'black'},
                                'bar': {'color': 'gray', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': 'lightgreen'},
                                        {'range': [20, 30], 'color': 'lightgreen'},
                                        {'range': [30, 50], 'color': 'red'},
                                        {'range': [50, 100], 'color': 'red'},
                                        ],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 1,
                                            'value': 35.2 }}))

    fig.update_layout(paper_bgcolor='white',
                        height=400, width=400,
                        font={'color': 'black', 'family': 'helvetica Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
    st.plotly_chart(fig, use_container_width=True)


# Explication de la prédiction 
################################
   
st.markdown("""
                <p style="color:#772b58;text-align:center;font-size:2.10em;font-style:
                ;font-weight:600;font-family:'helvetica Condensed';margin:0px;">
                Caractéristiques importantes pour le calcul du score client </h1>
                """, 
                unsafe_allow_html=True)
st.write("")

# Calcul des valeurs Shap
###########################
explainer_shap = shap.TreeExplainer(model_LGBM)
shap_values = explainer_shap.shap_values(load_X_test_raw().drop(labels="sk_id_curr", axis=1))

# récupération de l'index correspondant à l'identifiant du client
#################################################################
idx = int(load_X_test_raw()[load_X_test_raw()['sk_id_curr']==ID_client].index[0])

# Graphique force_plot
#######################
st.write("un graphique`force-plot`situe la prédiction par rapport à la `base value`.") 
st.write("Il indique les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que la grandeur de cet impact.")
st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            load_X_test_raw().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))
# Graphique decision_plot
#########################
st.write("Le graphique`decision_plot` est une autre façon d'expliquer la prédiction.\
            Il donne la valeur et l’impact sur le score de chaque variable par ordre d'importance \
            Il indique la direction prise par la prédiction  pour chacune des valeurs des variables affichées. ")
st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            load_X_test_raw().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            feature_names=load_X_test_raw().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -20, -1), # affichage des 15 variables les + importantes
                            link='logit'))

# Lime
######
st.markdown("""
                <p style="color:#772b58;text-align:center;font-size:2.10em;font-style:
                ;font-weight:600;font-family:'helvetica Condensed';margin:0px;">
                 Autre méthode d'interprétation de modèle : LIME</h1>
                """, 
                unsafe_allow_html=True)
st.write("")

st.write("Local Interpretable Modele agnostic Explanations (LIME)est une technique qui permet de créer un modèle simple autour de la prédiction \
que nous voulons expliquer et d'utiliser ce modele pour expliquer la prédiction. En pratique, pour une prédiction donnée , LIME va pertuber \
les entrées de la prédiction et regarder comment la pertubation a affécté la sortie du modèle. Les attributs qui affectent le plus la sortie \
lorsqu'ils sont pertubés sont considérés comme ayant une importance élevée pour cette prédiction")

st.write("En bleu les caractéristiques qui renforcent la classe 0 avec leur importance")

explainer_lime=lime_tabular.LimeTabularExplainer(load_X_test_raw().iloc[:,1:].values,feature_names=load_X_test_raw().iloc[:,1:].columns.values.tolist(),
                                            class_names=["pas_faillite_paiement","en_faillite_paiement"],
                                            verbose=True,
                                            mode="classification")

# choisir un individu
#####################


expl=explainer_lime.explain_instance(load_X_test_raw().iloc[:,1:].iloc[idx,:],model_LGBM.predict_proba,num_features=15)
fig=expl.as_pyplot_figure(label=1)
# afficher l'explication
#######################
st.write("identifiant client :", load_X_test_raw().iloc[idx, 0])
components.html(expl.as_html(), height=1000)
