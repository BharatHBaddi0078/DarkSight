# Darksight

Darksight is a dark web monitoring system that scrapes `.onion` links, analyzes the content using advanced keyword classification, and categorizes websites as illicit or legal. The project leverages machine learning models for classification and provides an interactive interface for users to view results.

## Features
- **Dark Web Scraping**: Securely fetch `.onion` web pages using a proxy server configured with the Tor browser on Kali Linux.
- **Keyword Analysis**: Scrape all keywords from the fetched web pages and analyze their meaning.
- **Illicit vs. Legal Classification**: Use a fine-tuned BERT model to classify the content of the websites based on scraped keywords.
- **Interactive Frontend**: Visualize results and interact with the system through a Streamlit-powered interface.

## Technologies Used
### Backend
- **Python**: Core programming language for backend development.
- **BERT (Hugging Face)**: Pre-trained language model fine-tuned for keyword-based classification.
- **PyTorch**: Framework for fine-tuning the BERT model and implementing ML pipelines.

### Frontend
- **Streamlit**: Provides a simple and interactive web interface for users to view results.

### Tools
- **Tor Browser**: Configured with a proxy server for secure access to the dark web.
- **Kali Linux**: Operating system used to run the Tor browser and scrape `.onion` links.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BharatHBaddi0078/DarkSight.git
   cd darksight
   ```
2. **Set Up Environment**:
   - Install Python 3.9 or above.
   - Create a virtual environment and activate it:
     ```bash
     python -m venv env
     source env/bin/activate # For Linux/Mac
     env\Scripts\activate   # For Windows
     ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Tor Proxy**:
   - Install the Tor browser on Kali Linux and configure it as a proxy server.
   - Update the proxy settings in the project to point to your Tor configuration.

## Usage
1. **Keyword Classification**:
   - Run the trained BERT model on the scraped keywords:
     ```bash
     python classify_text.py
     ```
2. **Launch the Streamlit App**:
   - View the results in an interactive frontend:
     ```bash
     streamlit run app.py
     ```

## Model Training
- The BERT model is fine-tuned using a dataset of illicit and legal keywords.
- Training scripts are available in the main directory.
- To re-train the model, provide a labeled dataset and run:
  ```bash
  python train_model.py
  ```

## Project Architecture
1. **Scraper**: Uses the Tor browser to securely fetch `.onion` pages.
2. **Keyword Analysis**: Extracts keywords from the fetched pages and processes them for classification.
3. **Classification Model**: Fine-tuned BERT model predicts whether the website content is illicit or legal.
4. **Frontend**: Streamlit application displays the results in an easy-to-use interface.

## Future Enhancements
- Enhance the keyword dataset for improved classification accuracy.
- Add real-time monitoring and alert notifications.
- Incorporate additional ML models for better context understanding.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
