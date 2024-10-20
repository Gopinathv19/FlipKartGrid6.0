from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import os
import tempfile

 
ocr = PaddleOCR(lang='en', use_angle_cls=True)

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def process_ocr():
    
    if 'image' not in request.files:
        print("Images are not given from the main application")
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_image_path = temp_file.name
            file.save(temp_image_path)   

         
        print(f"Running OCR on the uploaded image {file.filename}...")
        results = ocr.ocr(temp_image_path, cls=True)

        extracted_text = []
        if not results or not isinstance(results, list):
            print(f"OCR processing returned unexpected results: {results}")
            return jsonify({'error': 'OCR failed to process the image'}), 500

         
        for line in results:
            if line is None or not isinstance(line, list):
                continue   

            for word_info in line:
                if word_info and isinstance(word_info, list) and len(word_info) > 1:
                    text = word_info[1][0]  
                    confidence = word_info[1][1]   
                    extracted_text.append({'text': text, 'confidence': confidence})

        # Return the extracted text as a JSON response
        return jsonify({'filename': file.filename, 'extracted_text': extracted_text}), 200

    except Exception as e:
        print(f"An error occurred while processing OCR: {str(e)}")
        return jsonify({'error': 'An internal server error occurred', 'details': str(e)}), 500

    finally:
      
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
