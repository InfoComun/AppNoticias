import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import joblib
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Buscador de Fake News", page_icon="logo.png", layout="wide")
st.image("Logo Contrasta con letras.jpg", width=400)

# Funciones de scraping
def get_article_info_elplural(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('div', class_='article-header').find('h1').get_text(strip=True) if soup.find('div', class_='article-header') else "Título no encontrado"
    author = soup.find('span', class_='author').find('a').get_text(strip=True) if soup.find('span', class_='author') else "Autor no encontrado"
    article_text = " ".join([p.get_text(strip=True) for p in soup.find('div', class_='article-body').find_all('p')]) if soup.find('div', class_='article-body') else "Contenido no encontrado"
    return {'url': url, 'title': title, 'author': author, 'text': article_text}

def get_article_info_eldiario(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('h1', class_='title').get_text(strip=True) if soup.find('h1', class_='title') else "Título no encontrado"
    author = soup.find('a', href=lambda x: x and 'autores' in x).get_text(strip=True) if soup.find('a', href=lambda x: x and 'autores' in x) else "Autor no encontrado"
    article_text = " ".join([p.get_text(strip=True) for p in soup.find_all('p', class_='article-text')]) if soup.find_all('p', class_='article-text') else "Texto no encontrado"
    return {'url': url, 'title': title, 'author': author, 'text': article_text}

def get_article_info(url):
    if "elplural.com" in url:
        return get_article_info_elplural(url)
    elif "eldiario.es" in url:
        return get_article_info_eldiario(url)
    else:
        return {'url': url, 'title': "Medio no soportado", 'author': "Medio no soportado", 'text': "Medio no soportado"}

# Cargar el modelo y el vectorizador
def load_model_and_vectorizer():
    model_url = "https://raw.githubusercontent.com/Evalen-software/modelo/main/model.pkl"
    vectorizer_url = "https://raw.githubusercontent.com/Evalen-software/modelo/main/vectorize.pkl"
    model = joblib.load(BytesIO(requests.get(model_url).content))
    vectorizer = joblib.load(BytesIO(requests.get(vectorizer_url).content))
    return model, vectorizer

# Función principal
def main():
    st.title("Análisis de Noticias")
    url = st.text_input("Introduce la URL del artículo:")

    if st.button("Analizar"):
        if url:
            article_info = get_article_info(url)
            model, vectorizer = load_model_and_vectorizer()
            X = vectorizer.transform([article_info['text']])
            prediction = model.predict(X)
            result = "Bulo" if prediction[0] == 1 else "Verdadera"

            # Mostrar información del artículo
            st.subheader("Información del Artículo")
            st.write(f"**Título:** {article_info['title']}")
            st.write(f"**Autor:** {article_info['author']}")
            st.write(f"**URL:** {article_info['url']}")

            # Resultados del análisis en tarjeta de color
            st.subheader("Resultado del Análisis")
            result_color = "red" if result == "Bulo" else "green"
            st.markdown(
                f"<div style='text-align:center; background-color:{result_color}; padding:10px; color:white; font-size:24px; border-radius:5px;'>{result}</div>",
                unsafe_allow_html=True,
            )

            # Configurar columnas para los gráficos de palabras
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Nube de Palabras")
                wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white').generate(article_info['text'])
                plt.figure(figsize=(5, 3))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            with col2:
                st.subheader("Palabras Frecuentes")
                words = article_info['text'].lower().split()
                word_counts = pd.Series(words).value_counts().head(20)
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(x=word_counts.values, y=word_counts.index, ax=ax)
                ax.set_title('Top 20 Palabras')
                st.pyplot(fig)
        else:
            st.error("Por favor, introduce una URL válida.")

if __name__ == "__main__":
    main()

