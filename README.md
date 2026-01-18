# AI Music Critique System

## Problem Statement
Developing high-quality music is a complex process. Musicians often lack objective, data-driven feedback on their work before release. Traditional feedback loops are either subjective (peer reviews) or expensive (professional consultations), leading to uncertainty about a song's potential market performance.

## Solution
This project provides an automated, AI-driven music critique system. By leveraging machine learning (Random Forest) and signal processing (Librosa), the system extracts acoustic features from audio files to predict popularity scores. Furthermore, it integrates Large Language Models (GPT-3.5) to provide qualitative, human-like critiques that help artists understand the "why" behind their scores and how to improve.

---

## Project Details

### Features
- **Automated Feature Extraction**: Real-time extraction of spectral and rhythmic features using Librosa.
- **Popularity Prediction**: Scalable ML model to estimate market appeal.
- **AI Critique Agent**: Natural language feedback provided by GPT-3.5.
- **Feedback Loop**: Continuous learning through user-provided ratings and model retraining.
- **S3 Integration**: Seamless audio file handling via Amazon S3.

### Project Structure
```text
/src
  /core
    /audio      # Librosa feature extraction logic
    /model      # ML prediction and training services
    /agents     # AI-driven critique generation
  /web          # Flask API and frontend interfaces
  /utils        # Database and S3 helper classes
/models         # Persisted model files
/docker         # Containerization configs
```

### Setup & Usage

#### Prerequisites
- Docker & Docker Compose
- AWS S3 Account
- OpenAI API Key
- PostgreSQL Database

#### Running with Docker
1. Clone the repository.
2. Create a `.env` file with your credentials (see `.env.example`).
3. Build and run the container:
   ```bash
   docker-compose up --build
   ```

#### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python -m src.web.app
   ```
