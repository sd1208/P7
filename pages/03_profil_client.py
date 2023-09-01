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
import joblib
from PIL import Image


# Configuration de la page 
##########################
st.set_page_config(
        page_title='Profil du Client',
        page_icon = "",
        layout="wide" )



# Centrage de l'image du logo dans la sidebar
#############################################
col1, col2, col3= st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('logo p_a_d.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")




# Lecture des jeux de données 
#############################

# décorateur mise en cache de la fonction pour exécution unique
###############################################################
@st.cache_data
def read_X_test_raw():
    X_test_raw = pd.read_csv(r"C:\Users\serge\OneDrive\Documents\Documents\DATA SCIENTIST\P7\data\preproc\X_test_raw.csv")
    return X_test_raw

# mise en cache de la fonction pour exécution unique
#####################################################
@st.cache_data 
def read_X_test():
    X_test = pd.read_csv(r"C:\Users\serge\OneDrive\Documents\Documents\DATA SCIENTIST\P7\data\X_test.csv")
    X_test = X_test.rename(columns=str.lower)
    return X_test

# mise en cache de la fonction pour exécution unique
####################################################
@st.cache_data 
def read_description_variables():
    description_variables = pd.read_csv(r"C:\Users\serge\OneDrive\Documents\Documents\DATA SCIENTIST\P7\data\preproc\description_variable.csv", sep=";")
    return description_variables


if __name__ == "__main__":

    read_X_test_raw()
    read_description_variables()

# Reponse client
################    
    st.markdown("""
                <p style="color:#772b58;text-align:center;font-size:2.10em;font-style:;font-weight:600;font-family:'helvetica Condensed';margin:0px;">
                Réponse à la demande de prêt du client </h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

   
    # Liste déroulante sélection Identifiant du client 
    ##################################################
    liste_ID = list(read_X_test_raw()['sk_id_curr'])
    col1, col2,col3 = st.columns(3) # division de la largeur de la page en 3 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Sélectionnez ou entrez l'identifiant du client*", 
                                (liste_ID))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")
    with col3:
        st.write("")    

    #Chargement du modèle pour prédiction et score
    ##############################################
    model_LGBM = joblib.load(open("LGBM4_saved.joblib","rb"))
    target_pred = model_LGBM.predict(read_X_test_raw().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
    target_proba = model_LGBM.predict_proba(read_X_test_raw().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

    # Récupération du score du client
    df_target_proba = pd.DataFrame(target_proba, columns=['classe_0_proba', 'classe_1_proba'])
    df_target_proba = pd.concat([read_X_test_raw()['sk_id_curr'],df_target_proba['classe_1_proba']], axis=1)
    score = df_target_proba[df_target_proba['sk_id_curr']==ID_client]
    score_value = round(score.classe_1_proba.iloc[0], 2)

    #st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1%}**.")
    st.write(f"Le risque  que le client portant l'identifiant **{ID_client}** ait des difficultés de paiement est de **{score_value:.1%}**.")
    
   
    

    # informations du client 
    ########################
    data_client=read_X_test()[read_X_test().sk_id_curr == ID_client]
    
    col1, col2 = st.columns(2)
    with col1:
        # Titre H2
        st.markdown("""
                    <h2 style="color:#418b85;text-align:center;font-size:1.8em;font-style:italic;font-weight:600;margin:0px;">
                    Profil du client</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Genre : **{data_client['code_gender'].values[0]}**")
        st.write(f"Tranche d'âge : **{data_client['age_client'].values[0]}**")
        st.write(f"Ancienneté de la pièce d'identité : **{data_client['anciennete_cid'].values[0]}**")
        st.write(f"Situation familiale : **{data_client['name_family_status'].values[0]}**")
        st.write(f"Taille de la famille : **{data_client['nb_famille'].values[0]}**")
        st.write(f"Nombre d'enfants : **{data_client['nb_enfants'].values[0]}**")
        st.write(f"Niveau d'études : **{data_client['name_education_type'].values[0]}**")
        st.write(f"Revenu Total Annuel : **{data_client['total_revenus'].values[0]} $**")
        st.write(f"Type d'emploi : **{data_client['name_income_type'].values[0]}**")
        st.write(f"Ancienneté dans l'entreprise actuelle : **{data_client['anciennete_entreprise'].values[0]}**")
        st.write(f"Type d'habitation : **{data_client['name_housing_type'].values[0]}**")
        st.write(f"Densité de la Population de la région où vit le client : **{data_client['population_region'].values[0]}**")
        st.write(f"Evaluation de *'Prêt à dépenser'* de la région où vit le client : \
                   **{data_client['region_rating_client'].values[0]}**")
    
    with col2:
        # Titre H2
        st.markdown("""
                    <h2 style="color:#418b85;text-align:center;font-size:1.8em;font-style:italic;font-weight:600;margin:0px;">
                    Données financières de l' emprunteur</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Type de Crédit demandé par le client : **{data_client['name_contract_type'].values[0]}**")
        st.write(f"Montant du Crédit demandé par le client : **{data_client['montant_credit'].values[0]} $**")
        st.write(f"Durée de remboursement du crédit : **{data_client['nb_mensualites'].values[0]}**")
        st.write(f"Taux d'endettement : **{data_client['taux_endettement'].values[0]}**")
        st.write(f"Score normalisé du client à partir d'une source de données externe : \
                  **{data_client['ext_source_2'].values[0]:.1%}**")
        st.write(f"Nombre de demande de prêt réalisée par le client : \
                   **{data_client['nb_demande_pret_precedent'].values[0]:.0f}**")
        st.write(f"Montant des demandes de prêt précédentes du client : \
                  **{data_client['montant_demande_pret_precedent'].values[0]} $**")
        st.write(f"Montant payé vs Montant attendu en % : **{data_client['montant_paye_vs_du'].values[0]:.1f}%**")
        st.write(f"Durée mensuelle moyenne des crédits précédents : **{data_client['cnt_instalment'].values[0]:.1f} mois**")
        st.write(f"Nombre de Crédit à la Consommation précédent du client : \
                  **{data_client['prev_contrat_type_consumer_loans'].values[0]:.0f}**")
        st.write(f"Nombre de Crédit Revolving précédent du client : \
                  **{data_client['prev_contrat_type_revolving_loans'].values[0]:.0f}**")
        st.write(f"Nombre de Crédit précédent refusé : \
                  **{data_client['prev_contrat_statut_refused'].values[0]:.0f}**")
        st.write(f"Nombre de crédits cloturés enregistrés au bureau du crédit : \
                  **{data_client['bureau_credit_actif_closed'].values[0]:.0f}**")
        st.write(f"Nombre de crédits de type *'carte de crédit'* enregistrés au bureau du crédit : \
                  **{data_client['bureau_credit_type_credit_card'].values[0]:.0f}**")
        st.write(f"Nombre d'années écoulées depuis la décision précédente : \
                  **{data_client['nb_year_depuis_decision_precedente'].values[0]:.0f} ans**")


    
  
    
    # Calcul des valeurs Shap
    #########################
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(read_X_test_raw().drop(labels="sk_id_curr", axis=1))
    df_shap_values = pd.DataFrame(data=shap_values[1], columns=read_X_test_raw().drop(labels="sk_id_curr", axis=1).columns)
    
    df_work = pd.concat([df_target_proba['classe_1_proba'], df_shap_values], axis=1)
    labels = ['20% ou moins',"de_21_à_30%","de_31_à_50%","51%_ou plus"]
    df_work.insert(0, 'segmentation_risque'," ")
    df_work['segmentation_risque']= pd.qcut(df_work.classe_1_proba, q=4, labels=labels)
    
    
    st.write("")

    # Moyenne des variables par classe
    ##################################
    df_work_mean = df_work.groupby(['segmentation_risque']).mean()
    
    
    df_work_mean=df_work_mean.reset_index(drop=False, inplace=False)
    st.subheader(" probabilité moyenne de défaut de paiement par groupe segmentation du risque")
    fig_moy,ax_moy=plt.subplots()
    ax_moy=round(df_work.groupby('segmentation_risque')['classe_1_proba'].mean(),2).plot(kind='bar')            
    ax_moy.bar_label(ax_moy.containers[0])
    #plt.title('Probabilité moyenne de défaut de paiement par segmentation du risque')
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel('segmentation_risque')
    st.pyplot(fig_moy)

    st.subheader(" Distribution de la probabilité de défaut de paiement par Segmentation du risque")
    fig_dis,ax_dist=plt.subplots() 
    for m in df_work['segmentation_risque'].unique():
              
       # Propriétés graphiques     
       medianprops = {'color':"black"}
       meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'yellow'}
       ax_dist=sns.boxplot( x=df_work.loc[df_work['segmentation_risque']==m]['segmentation_risque'],y=df_work['classe_1_proba'], showfliers=False,medianprops=medianprops, showmeans=True, meanprops=meanprops)
       #plt.title("Distribution de défaut de paiement par Segmentation du risque\n",
          #loc="center", fontsize=14, fontstyle='italic')
       #plt.ylabel(m)
    st.pyplot(fig_dis)
    

  # Comparaison du profil du client par rapport aux différents groupes créés
    ###########################################################################

    st.markdown("""
                <br>
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
               Comparaison du profil du client aux différents groupes de segmentation du risque</h1>
                """, 
                unsafe_allow_html=True)
    

    
    # récupération de l'index correspondant à l'identifiant du client
    #################################################################
    idx = int(read_X_test_raw()[read_X_test_raw()['sk_id_curr']==ID_client].index[0])

    # dataframe avec shap values du client et des 4 groupes de clients
    ##################################################################
    comparaison_client_groupe = pd.concat([df_work[df_work.index == idx], 
                                            df_work_mean],
                                            axis = 0)
    comparaison_client_groupe['segmentation_risque'] = np.where(comparaison_client_groupe.index == idx, 
                                                          read_X_test_raw().iloc[idx, 0],
                                                          comparaison_client_groupe['segmentation_risque'])
    # transformation en array
    #########################
    nmp = comparaison_client_groupe.drop(
                      labels=['segmentation_risque', "classe_1_proba"], axis=1).to_numpy()

    fig = plt.figure(figsize=(8, 12))
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                                nmp, 
                                feature_names=comparaison_client_groupe.drop(
                                              labels=['segmentation_risque', "classe_1_proba"], axis=1).columns.to_list(),
                                feature_order='importance',
                                highlight=0,
                                legend_labels=[ID_client, '20%_et_moins', 'de_21_à_30%', 'de_31_à_50%', '51%_et_plus'],
                                plot_color='inferno_r',
                                legend_location='center right',
                                feature_display_range=slice(None, -30, -1),
                                link='logit'))

    
    
