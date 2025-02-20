# Project Structure

```
backend/                     # Django Project Root
│── api/                      # Django App (Handles API)
│   ├── migrations/           # DB Migrations (Auto-Generated)
│   ├── ml_models/            # ML Models Directory
│   │   ├── cnn_model.py      # CNN Model for Image Classification
│   │   ├── rag_model.py      # RAG Model for Text Generation
│   ├── serializers.py        # Serializes Data for APIs
│   ├── views.py              # API Logic (Handles Image Upload & Prediction)
│   ├── urls.py               # API Endpoints
│   ├── models.py             # Django Database Models (if needed)
│   ├── tests.py              # Tests for API Endpoints
│   ├── apps.py               # App Configuration
│── backend/                  # Main Django Project
│   ├── settings.py           # Main Configuration File
│   ├── urls.py               # Main URL Config (Includes `api.urls`)
│   ├── wsgi.py               # WSGI for Deployment
│   ├── asgi.py               # ASGI for Async Support
│── manage.py                 # Django Command-Line Utility
│── media/                    # Uploaded Images (Auto-Saved)
│── requirements.txt          # Project Dependencies
```

