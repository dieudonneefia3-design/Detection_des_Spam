import streamlit as st
import joblib
import re
import string
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Détecteur de SMS Spam",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS ciblé : masque le menu GitHub/Streamlit tout en gardant la barre latérale fonctionnelle
st.markdown("""
    <style>
    /* Masquer le menu 3 points et le pied de page */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Masquer l'icône de profil GitHub et le bouton Fork/View Source */
    a[href*="github.com"] {display: none !important;}
    button[title*="View code"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    
    /* Style des boutons de la page */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. PRÉTRAITEMENT NLP
# ------------------------------------------------------------------
@st.cache_resource
def get_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception:
        # Liste de secours si nltk pose problème
        return set(["to", "you", "a", "the", "in", "for", "is", "on", "that", "by",
                     "this", "with", "i", "it", "not", "or", "be", "are"])

STOP_WORDS = get_stopwords()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)

# ------------------------------------------------------------------
# 3. CHARGEMENT DU MODÈLE ET DU VECTORIZER
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

# ------------------------------------------------------------------
# 4. HISTORIQUE DE SESSION
# ------------------------------------------------------------------
if 'historique' not in st.session_state:
    st.session_state.historique = []

# ------------------------------------------------------------------
# 5. BARRE LATÉRALE (SIDEBAR) - INFOS SOUTENANCE
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Détecteur de SMS Spam")
    st.caption("Système de classification automatique basé sur le Machine Learning")

    st.divider()

    st.subheader("Performances du modèle")
    col_a, col_b = st.columns(2)
    col_a.metric("Exactitude", "98.16 %")
    col_b.metric("Précision", "98.28 %")

    col_c, col_d = st.columns(2)
    col_c.metric("Rappel", "87.02 %")
    col_d.metric("F1-Score", "92.31 %")

    st.divider()

    st.subheader("Configuration technique")
    st.markdown(
        """
        - **Algorithme** : SVM (noyau linéaire)
        - **Vectorisation** : TF-IDF (5000 caractéristiques, bigrammes)
        - **Optimisation** : GridSearchCV
        - **Corpus d'entraînement** : 5 169 SMS (anglais)
        """
    )

    st.divider()
    st.caption("Langue supportée : anglais uniquement")

# ------------------------------------------------------------------
# 6. CONTENU PRINCIPAL
# ------------------------------------------------------------------
st.title("Détecteur de SMS Spam")
st.write("Analyse en temps réel de SMS par Traitement du Langage Naturel (NLP).")

st.divider()

if model is None or vectorizer is None:
    st.error("**Erreur :** Les fichiers `spam_model.pkl` et `tfidf_vectorizer.pkl` doivent se trouver dans le même dossier que ce fichier `app.py`.")
else:
    tab_detection, tab_dashboard, tab_methodo = st.tabs(["Détection", "Dashboard des résultats", "Méthodologie"])

    # ==============================================================
    # ONGLET DÉTECTION
    # ==============================================================
    with tab_detection:
        # Exemples pré-remplis EN ANGLAIS (le modèle ne comprend que l'anglais)
        exemples = {
            "Saisir mon propre texte...": "",
            "Exemple Spam (Gains)": "WINNER! As a valued network customer you have been selected to receive a $900 prize! Call 09061701461 to claim now.",
            "Exemple Spam (Urgence)": "URGENT! Your mobile number has been awarded a £2000 bonus. Call 09066362231 now to claim.",
            "Exemple Ham (Personnel)": "Hey! Are we still meeting for lunch today at 1pm? Let me know when you leave.",
            "Exemple Ham (Rendez-vous)": "Your appointment with Dr Smith is confirmed for tomorrow at 10am.",
            "Exemple limite (français, non supporté)": "GAGNANT ! Vous avez été sélectionné pour recevoir un prix de 900 dollars ! Appelez le 09061701461 pour le réclamer."
        }

        choix = st.selectbox("Choisir un exemple de test rapide :", list(exemples.keys()))
        texte_par_defaut = exemples[choix]

        message = st.text_area(
            "Entrez le texte d'un SMS (en anglais) :",
            value=texte_par_defaut,
            height=120,
            placeholder="Ex: Congratulations! You have won a prize..."
        )

        btn_analyser = st.button("Analyser le message", type="primary")

        if btn_analyser:
            if not message.strip():
                st.warning("Merci de saisir un texte avant de lancer l'analyse.")
            else:
                cleaned = clean_text(message)
                vect = vectorizer.transform([cleaned])
                prediction = model.predict(vect)[0]

                try:
                    proba = model.predict_proba(vect)[0]
                    proba_spam = proba[1] * 100
                    proba_ham = proba[0] * 100
                except Exception:
                    proba_spam, proba_ham = None, None

                resultat_txt = "SPAM" if prediction == 1 else "HAM"
                score = proba_spam if prediction == 1 else proba_ham

                st.divider()
                st.subheader("Résultat")

                col_res1, col_res2 = st.columns([2, 1])

                with col_res1:
                    if prediction == 1:
                        st.error("**SPAM DÉTECTÉ**")
                        st.write("Ce message contient des expressions caractéristiques de tentatives de hameçonnage ou de spams.")
                    else:
                        st.success("**MESSAGE LÉGITIME (HAM)**")
                        st.write("Ce message ne présente aucun motif suspect.")

                with col_res2:
                    if score is not None:
                        st.metric(label="Indice de confiance", value=f"{score:.1f} %")
                        st.progress(score / 100)

                with st.expander("Voir le traitement NLP appliqué au texte"):
                    st.markdown("**Texte nettoyé (sans majuscules, ponctuation, ni stop-words) :**")
                    st.code(cleaned if cleaned else "(Aucun mot conservé après filtrage)")
                    if vect.nnz == 0:
                        st.warning("Aucun mot de ce message n'est reconnu par le modèle (vocabulaire d'entraînement anglais). La prédiction n'est pas fiable pour ce message.")

                # Ajout à l'historique
                st.session_state.historique.insert(0, {
                    'Heure': datetime.now().strftime('%H:%M:%S'),
                    'Message': message[:60] + ('...' if len(message) > 60 else ''),
                    'Résultat': resultat_txt,
                    'Confiance': f"{score:.1f} %" if score is not None else "N/A"
                })

        if st.session_state.historique:
            st.divider()
            st.subheader("Historique de la session")
            st.dataframe(pd.DataFrame(st.session_state.historique), use_container_width=True, hide_index=True)
            if st.button("Effacer l'historique"):
                st.session_state.historique = []
                st.rerun()

    # ==============================================================
    # ONGLET DASHBOARD
    # ==============================================================
    with tab_dashboard:
        METRIQUES = pd.DataFrame({
            'Accuracy':  [0.8733, 0.9710, 0.9584, 0.9816],
            'Precision': [0.0000, 0.9903, 0.9889, 0.9828],
            'Recall':    [0.0000, 0.7786, 0.6794, 0.8702],
            'F1-score':  [0.0000, 0.8718, 0.8054, 0.9231],
        }, index=['Baseline', 'Naive Bayes', 'Régression Logistique', 'SVM (optimisé)'])

        VALIDATION_CROISEE = pd.DataFrame({
            'F1 moyen (5-fold)': [0.8820, 0.7818, 0.9065],
            'Écart-type': [0.0178, 0.0383, 0.0130],
        }, index=['Naive Bayes', 'Régression Logistique', 'SVM'])

        REEQUILIBRAGE = pd.DataFrame({
            'Precision': [0.9884, 0.8750, 0.9828, 0.9200],
            'Recall':    [0.6489, 0.9084, 0.8702, 0.8779],
            'F1-score':  [0.7834, 0.8914, 0.9231, 0.8984],
        }, index=['LogReg (normale)', 'LogReg (balanced)', 'SVM (normal)', 'SVM (balanced)'])

        st.subheader("Comparaison des modèles (jeu de test)")
        st.bar_chart(METRIQUES['F1-score'])
        st.dataframe(METRIQUES.style.format("{:.2%}"), use_container_width=True)
        st.caption(
            "La baseline (classe majoritaire) atteint 87,33% d'accuracy mais 0% de F1-score : "
            "elle ne détecte aucun spam, ce qui prouve que l'accuracy seule est trompeuse sur ce "
            "dataset déséquilibré (87% ham / 13% spam)."
        )

        st.divider()
        st.subheader("Validation croisée (5-fold) — Stabilité des modèles")
        st.dataframe(VALIDATION_CROISEE.style.format("{:.2%}"), use_container_width=True)
        st.caption("Le SVM combine le meilleur score moyen ET le plus faible écart-type : le modèle le plus performant et le plus stable.")

        st.divider()
        st.subheader("Test du rééquilibrage des classes (class_weight='balanced')")
        st.dataframe(REEQUILIBRAGE.style.format("{:.2%}"), use_container_width=True)
        st.caption(
            "Le rééquilibrage améliore la Régression Logistique (+10,8 pts de F1) mais dégrade "
            "légèrement le SVM (-2,5 pts) : c'est pourquoi le SVM sans rééquilibrage reste le modèle final retenu."
        )

    # ==============================================================
    # ONGLET MÉTHODOLOGIE
    # ==============================================================
    with tab_methodo:
        st.subheader("Pipeline du projet")
        st.markdown(
            """
            ```
            Dataset brut (5 572 SMS)
                  ↓
            Nettoyage (doublons, valeurs manquantes) → 5 169 messages uniques
                  ↓
            Prétraitement NLP (minuscules, ponctuation, stopwords)
                  ↓
            Vectorisation TF-IDF (5000 features, unigrammes + bigrammes)
                  ↓
            Entraînement : Naive Bayes / Régression Logistique / SVM
                  ↓
            Évaluation (baseline, validation croisée, GridSearchCV)
                  ↓
            Modèle final : SVM optimisé (F1 = 92,31%)
                  ↓
            Déploiement (application Streamlit)
            ```
            """
        )

        st.divider()
        st.subheader("Dataset")
        st.markdown(
            """
            - **Source** : UCI SMS Spam Collection
            - **Taille** : 5 572 messages (5 169 après suppression des doublons)
            - **Répartition** : 87,4 % ham / 12,6 % spam
            - **Langue** : anglais
            """
        )

        st.divider()
        st.subheader("Limites assumées")
        st.markdown(
            """
            - Dataset exclusivement en **anglais** — le modèle n'est pas fiable sur des messages en français
            - Données collectées vers **2011-2012** : le vocabulaire du spam évolue, un ré-entraînement
              périodique serait nécessaire en conditions réelles
            - Le modèle ne prend pas en compte le **numéro de l'expéditeur**, uniquement le texte
            - TF-IDF ne capture pas le **contexte** d'un mot (ex : "call" légitime vs "call" publicitaire)
            """
        )

        st.divider()
        st.subheader("Perspectives d'amélioration")
        st.markdown(
            """
            - Constitution d'un corpus de SMS en français et multilingue
            - Exploration de représentations contextuelles (embeddings) pour dépasser la limite du TF-IDF
            - Intégration en temps réel sur un flux SMS réel, avec ré-entraînement périodique
            """
        )