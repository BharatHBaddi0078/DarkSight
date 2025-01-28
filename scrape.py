import requests
from bs4 import BeautifulSoup
import socket
import re

session = requests.Session()
session.proxies = {
    'http': 'socks5h://localhost:9050',
    'https': 'socks5h://localhost:9050'
}

def scrape_and_format(onion_url, output_file):
    try:
        response = session.get(onion_url)
        response.raise_for_status()
        html_content = response.text
        
        soup = BeautifulSoup(html_content, 'html.parser')
        text_data = soup.get_text()

        formatted_text = clean_text(text_data)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(formatted_text)

        print(f"Scraped data saved to {output_file}")
        return output_file

    except (requests.RequestException, socket.error) as e:
        print(f"Error accessing {onion_url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def clean_text(text):
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
   
    return text

def main():
    onion_url = "http://wms5y25kttgihs4rt2sifsbwsjqjrx3vtc42tsu2obksqkj7y666fgid.onion/" 
    output_file = "keywords.txt" 

    formatted_data = scrape_and_format(onion_url, output_file)
    if formatted_data is None:
        print("Failed to scrape and format the site data.")
        return

if __name__ == "__main__":
    main()