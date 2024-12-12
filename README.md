## Smart Quality Testing System for Flipkart GRID 6.0 ## 

This repository contains the code and resources for a smart quality testing system developed as part of the Flipkart GRID 6.0 competition. The project leverages computer vision and natural language processing to automate quality checks using camera vision technology without the need for extra hardware. It includes three main components: Freshness Detection, QC Process Automation, and a PaddleOCR Server for text extraction.

**Project Overview**

The goal of this project is to create a cost-effective, intelligent system capable of analyzing product quality, extracting text information from packaging, and matching brand details, all with the use of a single camera.


  

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

 
### **Application 2: QC Process Automation**

---

## **Note on Running SpaCy and PaddleOCR Applications**

This repository includes three independent applications—**SpaCy**, **PaddleOCR**, and **QCprocessAutomation**—designed to run in separate environments simultaneously. Follow the steps below to set up and run them correctly.

---

### **Directory Structure**

1. **SpaCy Application**  
   - Located in the `SpacyServer` folder.  
   - Handles NLP tasks such as MRP and DATE recognition and processing.  

2. **PaddleOCR Application**  
   - Located in the `PaddleServer` folder.  
   - Facilitates OCR tasks for extracting text from images.

3. **QC Process Automation Application**  
   - Located in the `QCprocessAutomation` folder.  
   - Automates quality control tasks like product brand matching and data analysis.

---

### **Environment Setup**

Each application has its own dependencies, and it's essential to create separate virtual environments for them.

#### **Step 1: Set Up the SpaCy Application**
1. Navigate to the `SpacyServer` folder:
   ```bash
   cd spacy
   ```
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv spacy_env
   source spacy_env/bin/activate    # On Linux/MacOS
   spacy_env\Scripts\activate       # On Windows

   pip install -r requirements.txt
   ```
3. Start the SpaCy server by running:
   ```bash
   python spacyServerTestApp.py
   ```

#### **Step 2: Set Up the PaddleOCR Application**
1. Navigate to the `PaddleServer` folder:
   ```bash
   cd paddleocr
   ```
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv paddle_env
   source paddle_env/bin/activate    # On Linux/MacOS
   paddle_env\Scripts\activate       # On Windows

   pip install -r requirements.txt
   ```
3. Start the PaddleOCR server by running:
   ```bash
   python paddleTextExtraction.py
   ```

---

#### **Step 3: Set Up the QC Process Automation Application**
1. Navigate to the `qc_process_automation` folder:  
   ```bash
   cd qc_process_automation
   ```
2. Create a virtual environment and install requirements:  
   ```bash
   python -m venv qc_env
   source qc_env/bin/activate    # On Linux/MacOS
   qc_env\Scripts\activate       # On Windows

   pip install -r requirements.txt
   ```
3. Run the QC Process Automation application:  
   ```bash
   python QCprocessAutomation.py
   ```

---


### **Usage Instructions**

1. Ensure that **all three virtual environments** are set up and activated in their respective directories.  
2. Start the servers/applications individually:
   - **SpaCy Application:** Run `spacyServerTestApp.py`.  
   - **PaddleOCR Application:** Run `paddleTextExtraction.py`.  
   - **QC Process Automation Application:** Run `QCprocessAutomation.py`.  
3. Ensure proper configuration of server ports to avoid conflicts.

---

### **Key Notes**

- Each application must run in its **own environment** to avoid dependency conflicts.  
- Verify that the dependencies specified in the `requirements.txt` file for each folder are installed correctly.  

This modular setup ensures smooth execution of all applications, working seamlessly together to achieve the system's objectives. If you encounter any issues, refer to the respective folders for troubleshooting guides.  

--- 


**Key Technologies**

**YOLOv8:** Utilized for detecting objects (products) in images, crucial for locating products and regions of interest on packaging.

**EfficientNetB0:** Used for classifying the freshness of detected products.

**PaddleOCR:** Facilitates the extraction of text from images, acting as a standalone OCR server.

**SpaCy:** Used for text recognition, analysis, and processing, particularly in understanding brand names, product details, and context from extracted text.

**Sentence Transformer:** Applied for matching and verifying brand names from text data,matching by the cossine similarity ,ensuring accurate identification during the QC process.

**Flask:** Used to set up a lightweight server for running the OCR model and serving text extraction requests.
