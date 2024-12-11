from flask import Flask, request, jsonify
import spacy
from datetime import datetime, timedelta

app = Flask(__name__)

 
nlp = spacy.load("en_core_web_sm")

def preprocess_text(ocr_text):
 
    return ", ".join(ocr_text.split())

def parse_date(date_str):
 
    date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d%b%Y', '%B %d, %Y', '%m/%Y', '%b %Y', '%B %Y']

    for fmt in date_formats:
        try:
             
            if fmt in ['%m/%Y', '%b %Y', '%B %Y']:
                date_obj = datetime.strptime(date_str, fmt)
                last_day_of_month = (date_obj.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                return last_day_of_month
            else:
                return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def validate_expiry_date(expiry_date):
 
    if expiry_date > datetime.now():
        return f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')} - Valid"
    else:
        return f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')} - Expired"

@app.route('/extract_expiry_and_mrp', methods=['POST'])
def extract_expiry_and_mrp():
 
    try:
        data = request.json
        if not data or 'ocr_text' not in data:
            return jsonify({"error": "OCR text is missing"}), 400

        ocr_text = data['ocr_text']
        new_text = preprocess_text(ocr_text)
        doc = nlp(new_text)

        dates = []
        mrp = None

         
        for ent in doc.ents:
            if ent.label_ == "DATE":
                parsed_date = parse_date(ent.text)
                if parsed_date:
                    dates.append(parsed_date)
            elif ent.label_ == "MONEY" and mrp is None:
                mrp = ent.text   

        
        if dates:
            expiry_date = max(dates)
            expiry_message = validate_expiry_date(expiry_date)
        else:
            expiry_message = "No valid expiry date found."

        mrp_message = f"MRP: {mrp}" if mrp else "MRP not found."

        return jsonify({"expiry_message": expiry_message, "mrp_message": mrp_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
