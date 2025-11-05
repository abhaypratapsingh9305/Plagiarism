import streamlit as st
import pandas as pd
import nltk
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px

# Download NLTK tokenizer
nltk.download('punkt')

# ------------------------- GOOGLE SEARCH (with API) -------------------------

API_KEY = "YOUR_GOOGLE_API_KEY"     # 🔑 Replace with your Google API key
SEARCH_ENGINE_ID = "b3e3a5be1d2824349"  # 🧭 Replace with your Programmable Search Engine ID (cx)

def get_url(sentence):
    """Use Google Custom Search API to fetch first relevant result."""
    try:
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            "key": API_KEY,
            "cx": SEARCH_ENGINE_ID,
            "q": sentence
        }
        response = requests.get(search_url, params=params, timeout=10)
        data = response.json()

        if "items" in data:
            for item in data["items"]:
                link = item.get("link", "")
                if link and "youtube" not in link.lower():
                    return link
        return None
    except Exception as e:
        st.warning(f"⚠️ Error using Google Search API: {e}")
        return None

# ------------------------- REST OF YOUR CODE (unchanged) -------------------------

def get_sentences(text):
    return tokenize.sent_tokenize(text)

def read_text_file(file):
    with io.open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx_file(file):
    return docx2txt.process(file)

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_from_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            return read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            return read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx_file(uploaded_file)
    return ""

def get_text(url):
    try:
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    except:
        return ""

def get_similarity(text1, text2):
    text_list = [text1, text2]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    return cosine_similarity(count_matrix)[0][1]

def get_similarity_list(texts, filenames=None):
    similarity_list = []
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            similarity = get_similarity(texts[i], texts[j])
            similarity_list.append((filenames[i], filenames[j], similarity))
    return similarity_list

def get_similarity_list2(text, url_list):
    similarity_list = []
    for url in url_list:
        if not url:
            similarity_list.append(0)
            continue
        text2 = get_text(url)
        similarity_list.append(get_similarity(text, text2))
    return similarity_list



# ------------------------- PLOTTING FUNCTIONS -------------------------

def plot_scatter(df):
    fig = px.scatter(df, x='File 1', y='File 2', color='Similarity', title='Similarity Scatter Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_line(df):
    fig = px.line(df, x='File 1', y='File 2', color='Similarity', title='Similarity Line Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_bar(df):
    fig = px.bar(df, x='File 1', y='Similarity', color='File 2', title='Similarity Bar Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_pie(df):
    fig = px.pie(df, values='Similarity', names='File 1', title='Similarity Pie Chart')
    st.plotly_chart(fig, use_container_width=True)

def plot_box(df):
    fig = px.box(df, x='File 1', y='Similarity', title='Similarity Box Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_histogram(df):
    fig = px.histogram(df, x='Similarity', title='Similarity Histogram')
    st.plotly_chart(fig, use_container_width=True)

def plot_3d_scatter(df):
    fig = px.scatter_3d(df, x='File 1', y='File 2', z='Similarity', color='Similarity', title='3D Similarity Plot')
    st.plotly_chart(fig, use_container_width=True)

def plot_violin(df):
    fig = px.violin(df, y='Similarity', x='File 1', title='Similarity Violin Plot')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------- STREAMLIT UI -------------------------

st.set_page_config(page_title='Plagiarism Detection')
st.title('🧠 Plagiarism Detector')

st.write("### Enter text or upload a file to check for plagiarism or find similarities between files")

option = st.radio(
    "Select input option:",
    ('Enter text', 'Upload file', 'Find similarities between files')
)

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    text = get_text_from_file(uploaded_file) if uploaded_file else ""
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload multiple files", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts, filenames = [], []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            text_content = get_text_from_file(uploaded_file)
            texts.append(text_content)
            filenames.append(uploaded_file.name)
    text = " ".join(texts)

if st.button('Check for plagiarism or find similarities'):
    if not text:
        st.warning("⚠️ No text found for plagiarism check or similarity detection.")
        st.stop()

    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity']).sort_values(by='Similarity', ascending=False)
        plot_scatter(df)
        plot_line(df)
        plot_bar(df)
        plot_pie(df)
        plot_box(df)
        plot_histogram(df)
        plot_3d_scatter(df)
        plot_violin(df)
    else:
        st.info("🔍 Checking Google for similar content...")
        sentences = get_sentences(text)
        urls = [get_url(sentence) for sentence in sentences]
        valid_urls = [u for u in urls if u]

        if not valid_urls:
            st.success("✅ No plagiarism detected!")
            st.stop()

        similarity_list = get_similarity_list2(text, valid_urls)
        df = pd.DataFrame({'Sentence': sentences[:len(valid_urls)], 'URL': valid_urls, 'Similarity': similarity_list})
        df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)

        # Make URLs clickable
        df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        df_html = df.to_html(escape=False)
        st.write(df_html, unsafe_allow_html=True)

