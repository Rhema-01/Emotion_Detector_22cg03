# Emotion Detection Web App

## Overview
This project is an Emotion Detection Web Application that utilizes AI to detect human emotions from images or live captures. The application is built using Flask for the backend and incorporates a trained emotion detection model.

## Project Structure
```
STUDENTS-SURNAME_MAT.NO_EMOTION_DETECTION_WEB_APP
├── app.py                  # Main backend of the web application
├── model.py                # Code for training the emotion detection model
├── requirements.txt        # Required libraries and packages
├── link_to_my_web_app.txt  # Hosting link to the web app
├── trained_models           # Directory containing trained models
│   └── emotion_detector_v1.h5
├── database                # SQLite database for user records
│   └── users.db
├── templates               # HTML templates for the web app
│   └── index.html
├── static                  # Static files (CSS, JS)
│   ├── css
│   │   └── styles.css
│   └── js
│       └── capture.js
├── utils                   # Utility functions
│   └── helpers.py
├── notebooks               # Jupyter notebooks for experiments
│   └── model_experiments.ipynb
├── tests                   # Unit tests for the model
│   └── test_model.py
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
└── LICENSE                 # Licensing information
```

## Setup Instructions
1. **Clone the Repository**: 
   ```
   git clone <repository_link>
   cd STUDENTS-SURNAME_MAT.NO_EMOTION_DETECTION_WEB_APP
   ```

2. **Install Requirements**: 
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**: 
   ```
   python app.py
   ```

4. **Access the Web App**: Open your web browser and go to `http://127.0.0.1:5000`.

## Usage
- Upload an image or use the live capture feature to detect emotions.
- The application will display the detected emotion based on the input.

## Contributing
Contributions are welcome! Please submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.