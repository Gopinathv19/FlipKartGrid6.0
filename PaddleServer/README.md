#### **Step 2: Set Up the PaddleOCR Application**
1. Navigate to the `PaddleServer` folder:
   ```bash
   cd PaddleServer
   ```
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv paddle_env
   source paddle_env/bin/activate    # On Linux/MacOS
   paddle_env\Scripts\activate       # On Windows

   git clone https://github.com/PaddlePaddle/PaddleOCR.git
   cd PaddleOCR
   pip install requirements.txt

   cd ..

   pip install paddlepaddle

   ```
3. Start the PaddleOCR server by running:
   ```bash
   python paddleTextExtraction.py
   ```

---