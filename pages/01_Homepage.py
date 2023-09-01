
import streamlit as st
from PIL import Image

st.set_page_config(
        page_title='Modèle de scoring ',
        page_icon = "🏡",
        layout="wide"
    )

st.sidebar.header("Menu ")

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('logo p_a_d.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")


# Titre affiché sur la page d'accueil
st.markdown("""
            <p style="color:#772b58;text-align:center;font-size:2.10em;font-style:;font-weight:600;font-family:'helvetica Condensed';margin:0px;">
            Tableau de bord de scoring développé pour la société Prêt à Dépenser</p>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            <p style="color::Gray;font-size:0.80em;text-align: center">
            Serge Davister- OpenClassrooms Projet n°7 - Data Scientist</p>
            """, 
            unsafe_allow_html=True)
    
# Description du projet
st.markdown("""
            <p style="color:Black;font-family:'Inter', sans-serif;";font-style:">
            <br><br>
            Ce Dashboard a été conçu pour une utilisation en agence. Il permet d'une part d'apporter au client la réponse à sa demande de prêt et d'autre
            part en cas de réponse négative ,de justifier notre décision en mettant en évidence les éléments concrets qui ont motivé le refus d'octroyer le crédit.</strong>
            """, unsafe_allow_html=True)

st.write("""
            <p style="color:Black;font-family:'Inter', sans-serif;">
            Chaque demande de crédit est soumise à notre modèle de prédiction.Ce modèle a été entrainé avec un jeu de données de 307.511 clients
            pour lesquels nous savions déjà s'ils avaient ou pas rencontré des difficultés à rembourser leur crédit.<br>
            En réponse à chaque demande de crédit ,ce modèle calcule un score qui représente le pourcentage de risque que le client rencontre des difficultés
            pour rembourser son crédit. En fonction de la valeur de ce score, le prêt sera accordé ou refusé.
            <br></p>
            """, unsafe_allow_html=True)



st.markdown("""
            <p style="color:Black;font-size:2.10em;font-style:;font-weight:600;font-family:;">
            <strong> Comment l'utiliser ?<strong><p>""", unsafe_allow_html=True)

st.write("""
            <p style="color:Black;font-family:'Inter', sans-serif;"> <strong>👈 Cliquez sur l'onglet "Score client"</strong> et 
            sélectionnez l'identifiant du client dans la liste déroulante. Le score obtenu par le client s'affichera sur la jauge ainsi que la réponse à communiquer au client .Les variables les plus importantes avec leur poids sont mentionnées sous forme de graphiques
            <br></p>
            """, unsafe_allow_html=True)

st.write("""
            <p style="color:Black;font-family:'Inter', sans-serif;"> <strong>👈 Cliquez sur l'onglet "Profil client"</strong> et 
            sélectionnez l'identifiant du client dans la liste déroulante. Les données du profil géneral du client ainsi que certaines données financières relatives
            aux précédents crédits contractés par le client. 

            <br></p>
            """, unsafe_allow_html=True)



st.write("")


    
