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