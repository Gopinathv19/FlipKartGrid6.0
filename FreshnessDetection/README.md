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