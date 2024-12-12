**Smart Quality Testing System for Flipkart GRID 6.0**

This repository contains the code and resources for a smart quality testing system developed as part of the Flipkart GRID 6.0 competition. The project leverages computer vision and natural language processing to automate quality checks using camera vision technology without the need for extra hardware. It includes three main components: Freshness Detection, QC Process Automation, and a PaddleOCR Server for text extraction.

**Project Overview**

The goal of this project is to create a cost-effective, intelligent system capable of analyzing product quality, extracting text information from packaging, and matching brand details, all with the use of a single camera.


Here's a polished and visually appealing version of your "How to Run the Project" section for your GitHub README file:  

---

## **How to Run the Project**

### **Application 1: Freshness Detection**

Follow these steps to set up and run the Freshness Detection module.

---

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Gopinathv19/FlipKartGrid6.0.git
cd FreshnessDetection
```

---

### **Step 2: Set Up the Virtual Environment and Install Requirements**

Create a virtual environment (optional but recommended) and install all required dependencies:  
```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate    # On Linux/MacOS
venv\Scripts\activate       # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

### **Step 3: Set Up the MySQL Database**

Set up a **MySQL database** and create the table with the schema below:  

```sql
CREATE TABLE FreshnessData (
    Sl_no INT AUTO_INCREMENT PRIMARY KEY,
    Timestamp DATETIME NOT NULL,
    Produce VARCHAR(50) NOT NULL,
    Freshness VARCHAR(20) NOT NULL,
    Expected_Life_Span_Days INT NOT NULL
);
```

Ensure your database connection details (username, password, host, and database name) are configured correctly in the code.

---

### **Step 4: Run the Application**

Run the application using **Streamlit** from the command line:  

```bash
streamlit run bananaFreshnessDetection.py
```

---

### **Expected Output**

- The application will launch in your default web browser.  
- Turn on the camera permission in the browser  the system will classify their freshness, display results, and store data in the MySQL database.  

--- 

 





**Freshness Detection:** Detecting the freshness of products using image classification.
Quality Control (QC) Process Automation: Automating QC checks such as reading expiry dates, identifying brands, and counting products.
Text Extraction using OCR: Setting up a server using PaddleOCR for efficient text extraction from product images.
Repository Structure
The repository is organized into three main folders, each containing code and resources specific to different components of the project:

**1. Freshness Detection**

This folder contains the code for detecting the freshness of products using a YOLOv8 model for object detection and an EfficientNetB0 model for image classification.
It processes images of products and classifies their freshness status based on predefined criteria.
**Technologies Used:** Python, YOLOv8, EfficientNetB0, OpenCV.

**2. QC Process Automation**
This folder includes scripts and models for automating quality control processes, such as reading and analyzing product packaging details.
It uses YOLOv8 for object detection and various NLP models (e.g., RoBERTa and SpaCy) for text recognition and processing.
The module also implements fuzzy logic for brand name matching, ensuring accurate identification of products.
**Technologies Used:** Python, YOLOv8, RoBERTa, SpaCy, Fuzzy Logic.

**3. PaddleOCR Server**
This folder contains the implementation of a server using PaddleOCR, which handles the OCR (Optical Character Recognition) tasks for extracting text from product images.
The server is designed to be integrated with other modules, providing text data in real-time to assist with various checks, such as reading expiry dates or MRP.
**Technologies Used: ** Python, PaddleOCR, Flask (for server setup).

**Key Technologies**

**YOLOv8:** Utilized for detecting objects (products) in images, crucial for locating products and regions of interest on packaging.

**EfficientNetB0:** Used for classifying the freshness of detected products.

**PaddleOCR:** Facilitates the extraction of text from images, acting as a standalone OCR server.

**SpaCy:** Used for text recognition, analysis, and processing, particularly in understanding brand names, product details, and context from extracted text.

**Fuzzy Logic:** Applied for matching and verifying brand names from text data, ensuring accurate identification during the QC process.

**Flask:** Used to set up a lightweight server for running the OCR model and serving text extraction requests.
