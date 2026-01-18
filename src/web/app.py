from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import os

from src.utils.db_helper import DatabaseClient
from src.utils.aws_helper import AWSClient
from src.core.audio.processor import AudioProcessor
from src.core.model.predictor import Predictor
from src.core.model.trainer import ModelTrainer
from src.core.agents.critique import CritiqueAgent

app = Flask(__name__)
CORS(app)

# Initialize components
db = DatabaseClient()
aws = AWSClient()
audio_processor = AudioProcessor()
predictor = Predictor()
trainer = ModelTrainer()
critique_agent = CritiqueAgent()

@app.route('/api/predict', methods=['POST'])
def predict():
    song_name = request.form.get('filepath')
    bucket_name = request.form.get('bucket_name')
    bucket_region = request.form.get('bucket_region')
    
    if not all([song_name, bucket_name, bucket_region]):
        return jsonify({"error": "Missing required fields"}), 400
    
    if not song_name.lower().endswith('.mp3'):
        return jsonify({"error": "Only .mp3 files are supported"}), 400

    try:
        local_path = aws.download_file(song_name, bucket_name, bucket_region)
        y, sr = librosa.load(local_path)
        features = audio_processor.extract_features(y, sr)
        aws.remove_temp_file(local_path)
        
        # Check for existing rating first
        popularity = db.check_existing_rating(features, 'song_feedback')
        if popularity is None:
            popularity = predictor.predict(features)
        
        analysis = critique_agent.get_critique(features.to_dict('records')[0], popularity)
        return jsonify({"popularity": popularity, "analysis": analysis})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    song_path = request.form.get('filepath')
    bucket_name = request.form.get('bucket_name')
    bucket_region = request.form.get('bucket_region')
    user_feedback = request.form.get('feedback')

    if not all([song_path, bucket_name, bucket_region, user_feedback]):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        local_path = aws.download_file(song_path, bucket_name, bucket_region)
        y, sr = librosa.load(local_path)
        features = audio_processor.extract_features(y, sr)
        aws.remove_temp_file(local_path)
        
        db.store_feedback(features, float(user_feedback))
        return jsonify({"message": "Feedback stored successfully"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/retrain', methods=['GET'])
def retrain():
    if trainer.retrain():
        return jsonify({"message": "Model retrained successfully"})
    return jsonify({"message": "Not enough data or retrain failed"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
