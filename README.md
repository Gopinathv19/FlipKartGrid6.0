#Smart Quality Testing System for Flipkart GRID 6.0
This repository contains the code and resources for a smart quality testing system developed as part of the Flipkart GRID 6.0 competition. The project leverages computer vision and natural language processing to automate quality checks using camera vision technology without the need for extra hardware. It includes three main components: Freshness Detection, QC Process Automation, and a PaddleOCR Server for text extraction.

Project Overview
The goal of this project is to create a cost-effective, intelligent system capable of analyzing product quality, extracting text information from packaging, and matching brand details, all with the use of a single camera. The main applications include:

Freshness Detection: Detecting the freshness of products using image classification.
Quality Control (QC) Process Automation: Automating QC checks such as reading expiry dates, identifying brands, and counting products.
Text Extraction using OCR: Setting up a server using PaddleOCR for efficient text extraction from product images.
Repository Structure
The repository is organized into three main folders, each containing code and resources specific to different components of the project:

1. Freshness Detection
This folder contains the code for detecting the freshness of products using a YOLOv8 model for object detection and an EfficientNetB0 model for image classification.
It processes images of products and classifies their freshness status based on predefined criteria.
Technologies Used: Python, YOLOv8, EfficientNetB0, OpenCV.
2. QC Process Automation
This folder includes scripts and models for automating quality control processes, such as reading and analyzing product packaging details.
It uses YOLOv8 for object detection and various NLP models (e.g., RoBERTa and SpaCy) for text recognition and processing.
The module also implements fuzzy logic for brand name matching, ensuring accurate identification of products.
Technologies Used: Python, YOLOv8, RoBERTa, SpaCy, Fuzzy Logic.
3. PaddleOCR Server
This folder contains the implementation of a server using PaddleOCR, which handles the OCR (Optical Character Recognition) tasks for extracting text from product images.
The server is designed to be integrated with other modules, providing text data in real-time to assist with various checks, such as reading expiry dates or MRP.
Technologies Used: Python, PaddleOCR, Flask (for server setup).
Key Technologies
YOLOv8: Utilized for detecting objects (products) in images, crucial for locating products and regions of interest on packaging.
EfficientNetB0: Used for classifying the freshness of detected products.
PaddleOCR: Facilitates the extraction of text from images, acting as a standalone OCR server.
RoBERTa & SpaCy: Used for text recognition, analysis, and processing, particularly in understanding brand names, product details, and context from extracted text.
Fuzzy Logic: Applied for matching and verifying brand names from text data, ensuring accurate identification during the QC process.
Flask: Used to set up a lightweight server for running the OCR model and serving text extraction requests.
