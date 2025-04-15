    # DermaWiseAI - Skin Disease Analysis and Prediction using Machine Learning

    ## Abstract

    A number of patients around the world suffer from skin diseases, and early detection is crucial for effective treatment. Diagnosing skin conditions can be challenging due to the variety of skin diseases, their similar symptoms, and the need for expert dermatological evaluation. In such a scenario, adopting machine learning (ML) offers a promising solution, improving the accuracy of diagnosis and its availability, even in resource-constrained areas.

    The focus of this project is to develop a machine learning-based skin disease detection system. The aim is to create a tool capable of classifying various skin diseases using dermatoscopic images, which can be accessible and beneficial even in low-income regions.

    ## Keywords:
    - Machine Learning
    - Skin Disease Detection
    - Dermatoscopic Images
    - Random Forest
    - Gradient Boosting
    - AdaBoost

    ## Introduction

    Detecting skin diseases in the early stages is critical for providing effective treatment. With the wide variety of skin conditions and their overlapping symptoms, accurate diagnosis often relies on expert dermatological evaluation, which may not be accessible in many regions. This project explores the potential of machine learning models to assist in skin disease detection using dermatoscopic images.

    We aim to develop a machine learning model that can classify different forms of skin diseases. The goal is to offer a tool that is both accurate and accessible, particularly for use in areas with limited access to medical professionals.

    ## Methodology

    ### Machine Learning Models Used:
    - **Random Forest**: A powerful ensemble learning technique used for classification.
    - **Gradient Boosting**: A boosting algorithm that combines weak learners to form a strong predictive model.
    - **AdaBoost**: Another boosting algorithm that focuses on improving the prediction accuracy by adjusting the weights of misclassified samples.

    ### Data:
    The model utilizes dermatoscopic images to classify skin conditions. These images serve as the input to the machine learning model, which learns to distinguish between different types of skin diseases.

    ## Objective

    The main objective of this project is to:
    - Develop a machine learning model that can detect and classify various skin diseases.
    - Ensure that the system is robust, accurate, and can be used in low-resource settings.

    ## Conclusion

    The application of machine learning in skin disease detection holds great promise, especially in areas where expert dermatological services are limited. By automating the classification of dermatoscopic images, we aim to improve the accessibility and accuracy of skin disease diagnosis, offering a potential tool for early detection and treatment.


    make .venv for python version 10 - source .venv/bin/activate

    Alright bet, here’s a clean and crisp `README.md` snippet tailored for that exact scenario — **new laptop**, **fresh clone**, **no `requirements.txt`**, and you need **Python 3.10.x** via `pyenv`. Plug and play 🚀:

    ---

    ## ⚙️ Setup Instructions (for a fresh laptop)

    ### 1. Clone the repo
    ```bash
    git clone https://github.com/your-username/DermaWiseAI.git
    cd DermaWiseAI
    ```

    ---

    ### 2. Install Python 3.10.x using pyenv (if not already installed)
    > You need pyenv to manage Python versions. Install it via Homebrew:

    ```bash
    brew install pyenv
    pyenv install 3.10.14
    pyenv local 3.10.14  # sets it for this project
    ```

    ---

    ### 3. Create a virtual environment
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Mac/Linux
    ```

    ---

    ### 4. Install dependencies manually
    > Since there's no `requirements.txt`, install deps like this:

    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow python-dotenv djangorestframework django-cors-headers boto3
    ```

    for Powershell
    Perfect. Here's the TL;DR of what you did in PowerShell:

---

### 🧠 **What You Did (Windows / PowerShell version)**

1. **Created a virtual env**  
   ```powershell
   python -m venv .venv
   ```

2. **Activated it (PowerShell style)**  
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. **(Optional fix)** Allowed script execution if blocked  
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

4. **Installed your dependencies**  
   ```powershell
   pip install django numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```

5. **Ran the Django backend**  
   ```powershell
   cd backend
   python manage.py runserver
   ```

---

    folder structure
    DermaWIseAI/
    ├── .venv/
    ├── README.md                   # Project documentation
    ├── backend/
    │   ├── backend/                # Django project settings
    │   ├── manage.py               # Django management script
    │   ├── db.sqlite3              # SQLite database
    │   └── predictor/              # Django app for AI prediction
    │       ├── __init__.py
    │       ├── admin.py            # Django admin configuration
    │       ├── apps.py             # App configuration
    │       ├── models.py           # Database models
    │       ├── views.py            # API views/endpoints
    │       ├── tests.py            # Unit tests
    │       ├── model.py            # AI model implementation (active file)
    │       ├── dataset/            # Training/testing datasets
    │       ├── logs/               # Model training logs
    │       ├── model/              # Saved model files
    │       └── results/            # Prediction results
    │
    └── frontend/                   # Next.js frontend
        ├── src/                    # Source code
        ├── package.json            # Node.js dependencies
        ├── next.config.mjs         # Next.js configuration
        ├── tailwind.config.mjs     # Tailwind CSS configuration
        └── README.md               # Frontend documentation


