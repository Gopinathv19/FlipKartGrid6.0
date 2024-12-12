#### **Step 1: Set Up the SpaCy Application**
1. Navigate to the `SpacyServer` folder:
   ```bash
   cd SpacyServer
   ```
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv spacy_env
   source spacy_env/bin/activate    # On Linux/MacOS
   spacy_env\Scripts\activate       # On Windows

   pip install spacy

   python -m spacy download en_core_web_sm
   ```
3. Start the SpaCy server by running:
   ```bash
   python spacyServerTestApp.py
   ```