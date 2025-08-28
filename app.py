import os
from flask import Flask, request, render_template, jsonify
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Initialize Flask app
app = Flask(__name__)

# Load Azure Vision credentials (set these in your environment variables)
VISION_KEY = os.getenv("VISION_KEY", "your_vision_key_here")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT", "your_vision_endpoint_here")

# Create Image Analysis client
client = ImageAnalysisClient(
    endpoint=VISION_ENDPOINT,
    credential=AzureKeyCredential(VISION_KEY)
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']

    try:
        # Call Azure Vision API
        result = client.analyze(
            image_data=image_file,
            visual_features=[VisualFeatures.CAPTION]
        )

        caption_text = "No caption found"
        if result.caption is not None:
            caption_text = result.caption.text

        return jsonify({"caption": caption_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
