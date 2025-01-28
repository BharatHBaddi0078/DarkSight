import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import tempfile
import os

TOR_PROXY = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}

MODEL_PATH = './fine_tuned_model'

KEYWORDS = [
    "anonymous shipping", "next day delivery", "cash on delivery",
    "no questions asked", "bitcoin", "cryptocurrency", "btc", "monero",
    "anonymous payment", "gift cards", "western union", "moneygram",
    "pure quality", "high purity", "lab tested", "guaranteed delivery",
    "high potency", "top quality", "premium", "uncut", "bulk orders",
    "wholesale", "quick delivery", "discreet packaging", "trusted vendor",
    "verified seller", "repeat customer", "great communication", "good quality",
    "highly recommend", "end-to-end encryption", "pgp key", "secure transaction",
    "no logs", "tor only", "onion routing", "darknet", "black market",
    "cocaine", "heroin", "methamphetamine", "mdma", "ecstasy", "lsd",
    "fentanyl", "xanax", "oxycontin", "adderall", "valium", "opioids",
    "psychedelics", "steroids", "research chemicals", 
    "anonymous",
    "untraceable", "no kyc", "no verification", "no id required",
    "private transactions", "privacy coins", "weapons", "hacking services",
    "counterfeit goods", "fake ids", "stolen credit cards", "fraud",
    "ransomware", "money laundering", "tax evasion", "offshore accounts",
    "hidden transactions", "silk road", "alphabay", "mixing services",
    "bitcoin mixers", "tumbler services", "clean your coins",
    "high return on investment", "guaranteed profit", "ponzi scheme",
    "double your bitcoin", "get rich quick", "scam", "unregulated exchange",
    "decentralized exchange with no kyc", "peer-to-peer trading without verification",
    "instant exchange with no limits", "offshore exchange", "cash for bitcoin",
    "gift cards for bitcoin", "western union for bitcoin", "moneygram for bitcoin",
    "deep web", "hidden services", "escrow for illegal transactions",
    "fake reviews", "vendor"
]

st.set_page_config(
    page_title="Dark Web Analyzer",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def download_nltk_data():
    nltk.download('punkt_tab')
    nltk.download('stopwords')

@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word not in stop_words]
    return sorted(set(keywords))

def scrape_dark_web(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, proxies=TOR_PROXY, headers=headers, timeout=600)  
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.exceptions.ConnectionError:
        st.error("Connection refused. Please check if the Tor service is running and the URL is correct.")
        return None
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while scraping: {e}")
        return None

def classify_text(text, keywords, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)
        
        found_keywords = [keyword for keyword in keywords if keyword.lower() in text.lower()]
        
        return len(found_keywords) > 0, predictions, found_keywords
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None, []

def main():
    st.title("🔍 Dark Web Content Analyzer")
    
    download_nltk_data()
    
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Failed to load the model. Please check your model path and files.")
        return

    st.sidebar.title("About")
    st.sidebar.info(
        "This tool analyzes content from .onion URLs for potentially illicit content. "
        "It uses natural language processing and machine learning to classify text "
        "and identify keywords associated with illegal activities."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Enter a .onion URL in the input field
        2. Click 'Analyze' to start the process
        3. Review the results and statistics
        4. Download extracted keywords if needed
        
        ⚠️ **Note:** Ensure your Tor proxy is running on port 9050
        """
    )

    url = st.text_input("Enter .onion URL", "")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("Analyze")
    with col2:
        st.markdown("<div style='margin-top: 15px;'>ℹ️ Analysis may take a few moments.</div>", unsafe_allow_html=True)

    if analyze_button and url:
        with st.spinner("🔍 Scraping and analyzing content..."):
            content = scrape_dark_web(url)
            
            if content:
                extracted_keywords = extract_keywords(content)
                illicit, predictions, found_illegal_keywords = classify_text(content, KEYWORDS, tokenizer, model)
                
                st.markdown("<div class='stats-container'>", unsafe_allow_html=True)

                st.markdown(f"""
                    <div class='stat-box'>
                        <h3>Keywords Found</h3>
                        <h2>{len(extracted_keywords)}</h2>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class='stat-box'>
                        <h3>Illegal Keywords</h3>
                        <h2>{len(found_illegal_keywords)}</h2>
                    </div>
                """, unsafe_allow_html=True)

                if illicit:
                    status_color = "red"
                    status_text = "⚠️ Illicit"
                else:
                    status_color = "green"
                    status_text = "✅ Safe"
                
                st.markdown(f"""
                    <div class='stat-box'>
                        <h3>Classification</h3>
                        <h2 style='color: {status_color};'>{status_text}</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("Detailed Analysis")
                
                tab1, tab2, tab3 = st.tabs(["Classification", "Found Keywords", "Raw Text"])
                
                with tab1:
                    if predictions is not None:
                        st.write("Prediction probabilities:", predictions.tolist())
                    if found_illegal_keywords:
                        st.error("Found illegal keywords: " + ", ".join(found_illegal_keywords))
                
                with tab2:
                    if extracted_keywords:
                        st.write("Extracted keywords:", ", ".join(extracted_keywords))

                        temp_dir = tempfile.mkdtemp()
                        temp_file = os.path.join(temp_dir, "keywords.txt")
                        try:
                            with open(temp_file, "w", encoding='utf-8') as f:
                                for keyword in extracted_keywords:
                                    f.write(f"{keyword}\n")
                            
                            with open(temp_file, "r", encoding='utf-8') as f:
                                st.download_button(
                                    label="📥 Download Keywords",
                                    data=f,
                                    file_name="keywords.txt",
                                    mime="text/plain"
                                )
                        except Exception as e:
                            st.error(f"Error writing keywords to file: {e}")
                
                with tab3:
                    st.text_area("Scraped Content", content, height=200)
            else:
                st.error("Failed to scrape the URL. Please check if the URL is correct and accessible.")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center;'>
        ⚠️ For Project purpose only
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
