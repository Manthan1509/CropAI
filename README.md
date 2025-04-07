# KrishiMitra
# Project Overview :

KrishiMitra is an advanced agricultural technology project designed to assist farmers by predicting crop diseases through image analysis. Users can upload images of their crops, and the system utilizes AI models to identify potential diseases with a confidence score.

Additionally, KrishiMitra features an interactive chatbot that provides instant assistance to users. The chatbot can answer common agricultural queries, offer disease prevention tips, and guide users on the best steps to take after a diagnosis. This combination of AI-powered disease detection and real-time user support makes CropAI a valuable tool for modern farming.

# Problems Solved :
1. Early Detection of Crop Diseases
Farmers often struggle to detect plant diseases at an early stage, leading to significant crop damage. KrishiMitra helps by analyzing plant images and identifying diseases before they spread, allowing for timely intervention.

2. Reducing Crop Loss & Increasing Yield
Unidentified and untreated crop diseases can lead to reduced yield and financial losses. KrishiMitra helps farmers take preventive actions, ensuring healthier crops and better harvests.

3. Lack of Easy Access to Expert Advice
Many farmers do not have immediate access to agricultural experts. KrishiMitra’s chatbot provides real-time support, offering solutions, treatment suggestions, and preventive measures.

4. Minimizing Chemical Overuse
Farmers often use excessive pesticides and fungicides due to a lack of accurate disease diagnosis. KrishiMitra provides precise identification, reducing unnecessary chemical usage and promoting sustainable farming.

5. Bridging the Knowledge Gap
Many small-scale farmers lack scientific knowledge about plant diseases. CropAI simplifies complex agricultural information, providing easy-to-understand insights through AI-driven predictions and chatbot assistance.

By addressing these problems, KrishiMitra empowers farmers with technology-driven solutions, improving agricultural productivity and sustainability.


# Dependencies :
1. Backend (FastAPI)
These are required for API development and serving the ML model.

2. Machine Learning & Image Processing
For handling image input and running deep learning models.

pip install tensorflow torch torchvision  # Deep learning frameworks
pip install opencv-python  # Image processing
pip install numpy pandas  # Data manipulation
pip install pillow  # Image handling

3. Model Deployment & Serving
To handle pre-trained models.

pip install scikit-learn  # For preprocessing & ML utilities
pip install onnxruntime  # If using ONNX models
pip install keras  # If using Keras-based models

4. Chatbot & NLP

pip install transformers  # For NLP-based chatbot
pip install nltk  # Natural Language Processing
pip install sentence-transformers  # For embedding-based chatbot responses
pip install googlegenerativeai

5. Web Application (Frontend)
For building the user interface.

pip install streamlit

# Setup Instructions

1. Set Up Environment Variables

GOOGLE_API_KEY=AIzaSyAIwPMVkmR-oQn89jPrNIAB793IOLG_b-U
WEATHER_API_KEY=0532ccbb0f0b466588d171018253003


2. Start the Backend Server
main backend file is main.py, which starts the API.
Run it using:
python main.py

it should show output like :
Running on http://127.0.0.1:8000/

3. Start the Frontend (If applicable)
To open the UI:
simply open index.html in a browser.

4. Use KrishiMitra :D

#OR JUST DOWNLOAD THE CODE FROM THIS DRIVE
https://drive.google.com/drive/folders/1nwUMlaNCypbl8-iVO3lweHN_cM15cP4r

# FUTURE PLANS
1. Add authentication .
2. store data in database for memory .
3. Mobile app integration .
4. Multi Language Support .
5. Community Support Forum .

# Team Member Details :
1. Manthan Singhal : Enhanced Frontend
2. Anany Dev : Enhanced Backend
3. Shivam Singh : Developed Chatbot
4. Shubhneet Kumar : Developed Disease_prediction ml model
