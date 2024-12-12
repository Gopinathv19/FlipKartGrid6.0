import os
import time
import cv2
import json
import requests
import tkinter as tk
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from transformers import pipeline
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
 
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=0   
)
Emodel = SentenceTransformer('all-MiniLM-L6-v2')

model = YOLO('best.pt')




# Global variables to store the current frame and detected boxes
frame = None
boxes = []
cap = None

# Predefined list of brands and products for counting
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
    "lays": ["india's magic masala", "american style cream & onion"],
    "dove": ["cream bar", "dove advanced sensitive care", "hair fall rescue"],
    "parle-g": ["original - gluco-biscuit", "parle-g gold"],
    "maggi": ["your favorite masala taste", "special masala"],
    "7 up": ["refreshing lemon taste"],
    "sprite": ["lemon-lime-flavour"],
    "nivea": ["nourishing lotion body milk", "express hydration"],
    "dettol": ["liquid handwash"],
     "ozamore": ["ozenoxacin cream"],
     "gillette": ["7 0 clock"],
     "grb": ["ghee"],
"aashirvaad": ["whole wheat atta", "superior mp atta"],
"gold winner": ["refined sunflower oil"],
"naga": ["whole wheat flour"],
"lg": ["asafoetida powder"],
"mtr": ["coriander powder"],
"everest": ["kashmirilal", "black pepper"],
"sakthi": ["chilli powder"],
"lion": ["qyno dates", "deseeded dates"],
"tata": ["salt", "himalayan rock salt"],
"ponds": ["dreamflower", "sandal"],
"engage": ["spirit", "intrigue"],
"yardley": ["english lavender", "english rose"],
"park avenue": ["good morning", "voyage"],
"nycil": ["germ expert"],
"sensodyne": ["Sensitive", "Rapid Relief"],
"colgate": ["SlimSoft Charcoal", "MaxFresh"],
"vicco": ["vajradanti"],
"listerine": ["Cool Mint", "Cavity Fighter"],
"brut": ["green label", "instant"],
"levista": ["strong", "premium"],
"chakra gold": ["premium tea","care"],
"3 roses": ["natural care", "top star"],
"dakshina kashyap": ["Yumfills Pie", "Choco Fills"],
"domex": ["lime fresh", "ocean fresh"],
"Lizol": ["citrus", "floral"],
"duracell": ["Ultra"],
"cycle": ["Agarbathi"],
"dheepam": ["Lamp Oil"],
"cherry": ["blossom"],
}

# Precompute embeddings for products
product_embeddings = {}
product_to_brand = {}

for brand, products in BaP.items():
    for product in products:
        embedding = Emodel.encode(product, convert_to_tensor=True)
        product_embeddings[product] = embedding
        product_to_brand[product] = brand

# the dictionary to store the count of the product

product_counts = defaultdict(int)


def find_closest_product(query, top_k=1):
    query_embedding = Emodel.encode(query, convert_to_tensor=True)
    similarities = {}

    for product, embedding in product_embeddings.items():
        similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
        similarities[product] = similarity

    # Sort products by similarity
    sorted_products = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_products[:top_k]


 

              
# Function to save ROIs detected by YOLO
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
        # Load OCR data from the JSON file
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
    Send OCR text to the Flask server for expiry date and MRP extraction.
    """
    api_url = "http://localhost:5001/extract_expiry_and_mrp"
    try:
        response = requests.post(api_url, json={"ocr_text": ocr_text})
        response.raise_for_status()
        result = response.json()

        expiry_message = result.get("expiry_message", "No expiry information.")
        mrp_message = result.get("mrp_message", "No MRP information.")

        return expiry_message, mrp_message

    except requests.RequestException as e:
        return f"Error in connecting to Flask server: {e}", "Error in connecting to Flask server"
 

# Predefined expected product counts
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

        # Clear existing product counts
        product_counts.clear()

        # Iterate through each OCR text extracted from images
        for image_path, text in ocr_data.items():
            closest_products = find_closest_product(text)

            for product, similarity in closest_products:
                if similarity > 0.7:  # Use a threshold for valid matches
                    product_counts[product] += 1
                    print(f"Matched Product: {product} (Similarity: {similarity:.2f})")
                else:
                    print(f"No close match found for OCR text: {text}")

        # Display results with brand and product information
        count_message = "\n".join(
            [
                f"Brand: {product_to_brand[product]}, Product: {product}, Count: {count}"
                for product, count in product_counts.items()
            ]
        )
        if count_message:
            messagebox.showinfo("Product Count", f"Detected Products:\n{count_message}")
        else:
            messagebox.showinfo("Product Count", "No products detected.")

        # Clear the OCR data in the JSON file after counting is complete
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

# Set up the GUI using Tkinter
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
