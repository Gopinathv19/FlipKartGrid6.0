from flask import Flask, request, jsonify
import re
import spacy
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")

def preprocess_ocr_text(ocr_text):
    """
    Preprocess OCR text to handle MRP and date-related corrections.
    Args:
        ocr_text (str): Raw OCR text.
    Returns:
        str: Preprocessed and cleaned text.
    """
    # Normalize whitespace
    text = " ".join(ocr_text.split())

    corrections = {
        # MRP corrections
        r"\bmapr\b": "mrp",          # Common misread for MRP
        r"\bmpr\b": "mrp",           # Another misread for MRP
        r"rs\.?": "mrp",             # Replace 'Rs.' with 'MRP'
        r"inr\.?": "mrp",            # Replace 'INR' with 'MRP'
        r"₹": "mrp",                 # Handle the Indian currency symbol
        r"\brp\.?\b": "mrp",         # Misread of 'Rs.' as 'Rp.'
        r"\bmr\b": "mrp",            # Partial match for 'MRP'
        r"\bmrp[:\s]*₹?\s*\d+[.,]?\d*\b": lambda m: m.group().replace(",", "."),  # Normalize commas in MRP values (e.g., 1,250 -> 1250)

        # Date-related corrections
        r"~": "-",                   # Replace tilde with dash for date
        r":": "-",                   # Replace colon with dash for date
        r"[oO]": "0",                # Replace letter 'o' with zero in dates
        r"[lI]": "1",                # Replace lowercase L or uppercase I with one in dates
        r"\.": "-",                  # Replace dot with dash for date
        r"\bjan\b": "01",            # Normalize month names to numbers
        r"\bfeb\b": "02",
        r"\bmar\b": "03",
        r"\bapr\b": "04",
        r"\bmay\b": "05",
        r"\bjun\b": "06",
        r"\bjul\b": "07",
        r"\baug\b": "08",
        r"\bsep\b": "09",
        r"\boct\b": "10",
        r"\bnov\b": "11",
        r"\bdec\b": "12",
        r"\s(before|exp\b|mfg\b)": "-",  # Normalize " before", " exp", " mfg" with a dash

        # Noise removal for dates and MRP
        r"\s*[-]+(?=\d)": "-",       # Remove leading dashes before numeric values
        r"[^A-Za-z0-9\s.,:/-]": "",  # Remove irrelevant characters like special symbols
        r"\s+": " ",                 # Normalize whitespace
    }

    # Apply corrections
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Ensure standard lowercase for processing
    text = text.lower()

    return text


def parse_date(date_str):
    """Parse a date string into a datetime object."""
    date_formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d%b%Y', '%B %d, %Y', '%m/%Y', '%b %Y', '%B %Y']
    for fmt in date_formats:
        try:
            if fmt in ['%m/%Y', '%b %Y', '%B %Y']:
                date_obj = datetime.strptime(date_str, fmt)
                # Handle last day of the month
                last_day_of_month = (date_obj.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
                return last_day_of_month
            else:
                return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def extract_dates_and_mrp(text):
    """Extract dates and MRP from the text using regex."""
    # Define regex patterns
    date_pattern = r"(\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b)|(\b\d{1,2}[-/]\d{4}\b)|(\b[A-Za-z]{3,}\s\d{4}\b)"
    mrp_pattern = r"(\bMRP\s*[:\-]?\s*\d+[.,]?\d*\b)|(\b\d+[.,]?\d*\s*(INR|Rs\.?|₹)\b)"
    
    # Find all matches
    dates = re.findall(date_pattern, text)
    mrp_match = re.search(mrp_pattern, text)
    
    # Extract MRP
    mrp = None
    if mrp_match:
        mrp = mrp_match.group()
        mrp = re.sub(r"[^0-9.]", "", mrp)  # Clean up the numeric value of MRP
    
    # Extract and normalize dates
    parsed_dates = []
    for date_tuple in dates:
        date_str = next(filter(None, date_tuple))  # Get the matched string
        parsed_date = parse_date(date_str)
        if parsed_date:
            parsed_dates.append(parsed_date)
    
    return parsed_dates, mrp


def validate_expiry_date(expiry_date):
    """Validate the expiry date and check if it is expired."""
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
        cleaned_text = preprocess_ocr_text(ocr_text)
        
        # Extract data using regex
        dates, mrp = extract_dates_and_mrp(cleaned_text)
        
        # Process dates using spaCy NER
        doc = nlp(cleaned_text)
        spacy_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        
        # Normalize and combine spaCy and regex results
        parsed_dates = []
        for date_str in spacy_dates:
            parsed_date = parse_date(date_str)
            if parsed_date:
                parsed_dates.append(parsed_date)
        
        # Combine regex and spaCy results
        dates += parsed_dates
        dates = list(set(dates))  # Remove duplicates
        
        # Classify dates
        mfg_date, exp_date = None, None
        if dates:
            today = datetime.now()
            for date in dates:
                if date < today:
                    mfg_date = max(mfg_date, date) if mfg_date else date
                elif date > today:
                    exp_date = min(exp_date, date) if exp_date else date
        
        # Generate response
        mfg_message = f"MFG Date: {mfg_date.strftime('%Y-%m-%d')}" if mfg_date else "MFG Date not found."
        exp_message = f"Expiry Date: {exp_date.strftime('%Y-%m-%d')}" if exp_date else "Expiry Date not found."
        mrp_message = f"MRP: {mrp}" if mrp else "MRP not found."
        
        return jsonify({"mfg_message": mfg_message, "exp_message": exp_message, "mrp_message": mrp_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
