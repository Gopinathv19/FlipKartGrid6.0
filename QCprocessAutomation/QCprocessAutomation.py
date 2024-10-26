import os
import time
import cv2
import json
import requests
import spacy
import re
from datetime import datetime,timedelta
import tkinter as tk
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from transformers import pipeline
import numpy as np
from rapidfuzz import process 


qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0   
)


 
nlp = spacy.load("en_core_web_trf")


 
model = YOLO('best.pt')



 
frame = None
boxes = []
cap = None

 
BaP={
    "fortune": ["rice bran health"],
    "elite": ["family wonder bread", "magic sweet bread"],
    "britannia": ["50-50"],
    "nutri choice": ["thin arrowroot"],  
    "harpic": ["white & shine"],
    "kissan": ["jam"],
    "nescafe": ["gold blend"],
    "surf excel": ["matic top load"],
    "tide": ["tide double power"],
    "aachi": ["mutton masala", "chicken masala"],
    "lay's": ["india's magic masala", "american style cream & onion"],
    "dove": ["cream bar", "dove advanced sensitive care", "hair fall rescue"],
    "parle-g": ["original - gluco-biscuit", "parle-g gold"],
    "maggi": ["your favorite masala taste", "special masala"],
    "7 up": ["refreshing lemon taste"],
    "sprite": ["lemon-lime-flavour"],
    "nivea": ["nourishing lotion body milk", "express hydration"],
    "dettol": ["liquid handwash"],
     "ozamore": ["ozenoxacin cream"],
     "gillette": ["7 0 clock"],

}


product_counts = {}   

 
def save_rois_in_batches(current_frame, detected_boxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_images = []
    for i, box in enumerate(detected_boxes):
        x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        roi = current_frame[y_min:y_max, x_min:x_max]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(output_dir, f"roi_{timestamp}_{i}.png")
        count = 1
        while os.path.exists(image_path):
            image_path = os.path.join(output_dir, f"roi_{timestamp}_{i}_{count}.png")
            count += 1
        if cv2.imwrite(image_path, roi):
            saved_images.append(image_path)
    return saved_images

def extract_text_from_ocr_result(ocr_result):
    extracted_texts = {}
    for item in ocr_result:
        for image_path, details in item.items():
            text_blocks = []
            for entry in details.get('extracted_text', []):
                text_blocks.append(entry['text'])
            extracted_text = " ".join(text_blocks)
            if extracted_text:
                extracted_texts[image_path] = extracted_text
    return extracted_texts

def perform_ocr(image_path):
    api_url = "http://localhost:5000/ocr"
    try:
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            try:
                response = requests.post(api_url, files=files)
                response.raise_for_status()
                result = response.json()
                return {image_path: result}
            except requests.RequestException as e:
                print(f"Error in API request: {e}")
                return {image_path: None}
    except Exception as e:
        print(f"Error opening image file {image_path}: {e}")
        return {image_path: None}

def extract_text_concurrently(image_paths, output_json='output.json'):
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_image = {executor.submit(perform_ocr, img): img for img in image_paths}
        for future in future_to_image:
            img_path = future_to_image[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    if results:
        extracted_texts = extract_text_from_ocr_result(results)
        try:
            with open(output_json, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
        existing_data.update(extracted_texts)
        try:
            with open(output_json, 'w', encoding='utf-8') as json_file:
                json.dump(existing_data, json_file, indent=4, ensure_ascii=False)
            print(f"OCR results saved to {output_json}")
        except Exception as e:
            print(f"Error writing to {output_json}: {e}")

 
def find_best_product(brand, ocr_text):
    normalized_text = ocr_text.lower()
    products = BaP[brand]

 
    for product in products:
        if product.lower() in normalized_text:
            print(f"Direct or partial match found for product: {product}")
            return product

 
    best_match = process.extractOne(normalized_text, products)   
    if best_match and len(best_match)>=2:   
        product, score = best_match[:2]  
        if score > 60:   
            print(f"Fuzzy match found for product: {product} with score: {score}")
            return product
    return None


 
 
def find_best_brand(ocr_text):
    normalized_text = ocr_text.lower()
    brands = list(BaP.keys())

 
    for brand in brands:
        if brand.lower() in normalized_text:
            print(f"Direct match found for brand: {brand}")
            return brand

 
    best_match = process.extractOne(normalized_text, brands)   
    if best_match and len(best_match)>=2:  
        brand, score = best_match[:2]   
        if score > 60:   
            print(f"Fuzzy match found for brand: {brand} with score: {score}")
            return brand
    return None


def detect_products(current_frame, conf_threshold=0.5):
    results = model(current_frame)
    detected_boxes = []
    for result in results[0].boxes:
        if result.conf[0].item() >= conf_threshold:
            x_min = int(result.xyxy[0][0].item())
            y_min = int(result.xyxy[0][1].item())
            x_max = int(result.xyxy[0][2].item())
            y_max = int(result.xyxy[0][3].item())
            box = {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'confidence': float(result.conf[0].item()),
                'class_name': model.names[int(result.cls[0].item())]
            }
            detected_boxes.append(box)
    return detected_boxes

def start_video_stream():
    global frame, boxes, cap
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
        def update_frame():
            global frame, boxes
            ret, frame = cap.read()
            if ret:
                boxes = detect_products(frame)
                for box in boxes:
                    x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                flipped_frame = cv2.flip(frame, 1)
                resized_frame = cv2.resize(flipped_frame, (640, 480))
                cv2.imshow('YOLOv8 Detection', resized_frame)
            if cap is not None and cap.isOpened():
                root.after(10, update_frame)
        root.after(10, update_frame)
    except Exception as e:
        print(f"Error in starting video stream: {e}")

def capture_frame(output_json='output.json', output_dir='images'):
    global frame, boxes
    if frame is None or len(boxes) == 0:
        messagebox.showwarning("Warning", "No frame or objects detected to capture.")
        return
    image_paths = save_rois_in_batches(frame, boxes, output_dir)
    if image_paths:
        extract_text_concurrently(image_paths, output_json)
        messagebox.showinfo("Success", f"Captured and processed {len(image_paths)} images.")

def extract_ocr_details():
    try:
        with open('output.json', 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        ocr_text = " ".join(ocr_data.values())
        brand_name_response = qa_pipeline(context=ocr_text, question="What is the brand name?")
        pack_size_response = qa_pipeline(context=ocr_text, question="What is the pack size?")
        brand_details_response = qa_pipeline(context=ocr_text, question="What are the brand details?")
        brand_name = brand_name_response['answer']
        pack_size = pack_size_response['answer']
        brand_details = brand_details_response['answer']
        result_text.delete('1.0', tk.END)
        result_text.insert(tk.END, f"Brand Name: {brand_name}\n")
        result_text.insert(tk.END, f"Pack Size: {pack_size}\n")
        result_text.insert(tk.END, f"Brand Details: {brand_details}\n")
        open('output.json', 'w').close()
    except Exception as e:
        messagebox.showerror("Error", f"Could not extract details: {e}")

def validate_expiry_date_and_mrp():
    """
    Load OCR data, extract expiry date and MRP, display results, and clear the OCR data.
    """
    try:
         
        with open('output.json', 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        if not ocr_data:
            raise ValueError("OCR data file is empty or not found.")

        ocr_text = " ".join(ocr_data.values())   

        
        expiry_message, mrp_message = extract_expiry_date_and_mrp(ocr_text)

 

        messagebox.showinfo("Validation Results", f"{expiry_message}\n{mrp_message}")

         
        open('output.json', 'w').close()

    except ValueError as ve:
       
        messagebox.showerror("Validation Error", str(ve))

        
        messagebox.showinfo("Partial Validation Results", f"{expiry_message}\n{mrp_message}")

       
        open('output.json', 'w').close()

    except FileNotFoundError:
        messagebox.showerror("File Error", "OCR data file 'output.json' not found.")
        
    except json.JSONDecodeError:
        messagebox.showerror("Data Error", "Failed to decode JSON data from the OCR file.")

    except Exception as e:
        
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred: {e}")

def extract_expiry_date_and_mrp(ocr_text):
    """
    Process the OCR text to extract the expiry date and MRP using spaCy's NER.
    """
    new_text = preprocess_text(ocr_text)
    doc = nlp(new_text)

    dates = []
    mrp = None

    
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")   
        if ent.label_ == "DATE":
            parsed_date = parse_date(ent.text)
            if parsed_date:
                dates.append(parsed_date)
        elif ent.label_ == "CARDINAL" and mrp is None:
            mrp = ent.text  

    print(f"Extracted Dates: {dates}")   

   
    if dates:
        expiry_date = max(dates)
        expiry_message = validate_expiry_date(expiry_date)
    else:
        expiry_message = "No valid expiry date found."

    mrp_message = f"MRP: {mrp}" if mrp else "MRP not found."

    return expiry_message, mrp_message



def preprocess_text(ocr_text):
    """
    Add a comma after every space in the OCR text to improve separation of entities.
    This helps spaCy recognize dates and monetary values more effectively.
    """
    return ", ".join(ocr_text.split())




def parse_date(date_str):
    """
    Try to parse a date from the string using multiple date formats.
    Also handles month/year formats by assuming the last day of the month.
    """
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
    """
    Check if the expiry date is in the future and return a message indicating its status.
    """
    if expiry_date > datetime.now():
        return f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')} - Valid"
    else:
        return f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')} - Expired"



 
expected_counts = {
 "mutton masala" : 1,
 "chicken masala":1,
 "7 0 clock":1
}

def validate_counts():
    global product_counts

    
    validation_results = []

    for product, counted in product_counts.items():
        expected = expected_counts.get(product, 0)
        if counted == expected:
            validation_results.append(f"{product}: Count is correct (Expected: {expected}, Found: {counted})")
        else:
            validation_results.append(f"{product}: Count is incorrect (Expected: {expected}, Found: {counted})")

    if validation_results:
        result_message = "\n".join(validation_results)
        messagebox.showinfo("Count Validation", result_message)
    else:
        messagebox.showinfo("Count Validation", "No products counted.")




 
def count_products():
    global product_counts

    try:
         
        with open('output.json', 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)

        
        product_counts.clear()

        
        for image_path, text in ocr_data.items():
           
            normalized_text = text.lower()

             
            brand = find_best_brand(normalized_text)
            if brand:
              
                product = find_best_product(brand, normalized_text)
                if product:
                
                    product_counts[product] = product_counts.get(product, 0) + 1
                    print(f"Identified product: {product} for brand: {brand}")   
                else:
                    print(f"No product match found for brand: {brand}")
            else:
                print(f"No brand match found in text: {normalized_text}")

      
        count_message = "\n".join([f"Brand: {brand}, Product: {product}, Count: {count}" 
                                    for product, count in product_counts.items() 
                                    for brand in BaP if product in BaP[brand]])   
        if count_message:
            messagebox.showinfo("Product Count", f"Detected Products:\n{count_message}")
        else:
            messagebox.showinfo("Product Count", "No products detected.")

       
        with open('output.json', 'w') as f:
            json.dump({}, f)

    except Exception as e:
        messagebox.showerror("Error", f"Could not count products: {e}")

def quit_application():
    global cap
    try:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error during resource cleanup: {e}")
    finally:
        root.quit()
        root.destroy()

 
root = tk.Tk()
root.title("YOLOv8 Object Detection and OCR")
root.geometry("500x600")

 
start_video_stream()

 
capture_button = tk.Button(root, text="Capture Frame", command=lambda: capture_frame('output.json', 'images'))
capture_button.pack(pady=5)

validate_expiry_button = tk.Button(root, text="Validate Expiry Date & MRP", command=validate_expiry_date_and_mrp)
validate_expiry_button.pack(pady=5)

extract_button = tk.Button(root, text="Extract OCR Details", command=extract_ocr_details)
extract_button.pack(pady=5)

count_products_button = tk.Button(root, text="Count Products", command=count_products)
count_products_button.pack(pady=5)

 
validate_counts_button = tk.Button(root, text="Validate Counts", command=validate_counts)
validate_counts_button.pack(pady=5)


result_text = tk.Text(root, height=10, width=50)
result_text.pack(pady=5)

quit_button = tk.Button(root, text="Quit", command=quit_application)
quit_button.pack(pady=5)

root.protocol("WM_DELETE_WINDOW", quit_application)
root.mainloop()
