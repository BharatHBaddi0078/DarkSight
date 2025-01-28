from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')
model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')

model.eval()

def classify_text(text, keywords):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return False, None

    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return False, None

    logits = outputs.logits
    predictions = torch.softmax(logits, dim=1)

    for keyword in keywords:
        if keyword.lower() in text.lower():
            return True, predictions

    return False, predictions

def main():
    input_file = "keywords.txt"  
    keywords = ["anonymous shipping", "next day delivery", "cash on delivery", "no questions asked", "bitcoin", "cryptocurrency", "btc", "monero", "anonymous payment", "gift cards", "western union", "moneygram", "pure quality", "high purity", "lab tested", "guaranteed delivery", "high potency", "top quality", "premium", "uncut", "bulk orders", "wholesale", "quick delivery", "discreet packaging", "trusted vendor", "verified seller", "repeat customer", "great communication", "good quality", "highly recommend", "end-to-end encryption", "pgp key", "secure transaction", "no logs", "tor only", "onion routing", "darknet", "black market","cocaine", "heroin", "methamphetamine", "mdma", "ecstasy", "lsd", "fentanyl", "xanax","oxycontin", "adderall", "valium", "opioids", "psychedelics", "steroids", "research chemicals","anonymous", "untraceable", "no kyc", "no verification", "no id required", "private transactions", "privacy coins", "darknet", "drugs", "weapons", "hacking services","counterfeit goods", "fake ids", "stolen credit cards", "fraud", "black market", "ransomware", "child pornography", "money laundering", "tax evasion", "offshore accounts", "hidden transactions", "silk road", "alphabay", "mixing services", "bitcoin mixers", "tumbler services", "clean your coins", "high return on investment", "guaranteed profit", "ponzi scheme", "double your bitcoin", "get rich quick", "scam", "unregulated exchange", "decentralized exchange with no kyc", "peer-to-peer trading without verification", "instant exchange with no limits", "offshore exchange", "cash for bitcoin", "gift cards for bitcoin","western union for bitcoin", "moneygram for bitcoin", "deep web", "hidden services","escrow for illegal transactions", "fake reviews", "vendor"]  # Replace with your keywords

    text_data = ""  

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    text_data += line + " "  

        illicit, predictions = classify_text(text_data, keywords)

        if predictions is None:
            print("Prediction process failed.")
            return

        if illicit:
            print("Illicit content detected!")
        else:
            print("No illicit content detected.")
        print(f"Predictions: {predictions}")

    except FileNotFoundError:
        print(f"File {input_file} not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
