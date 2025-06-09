#!/usr/bin/env python3

from flask import Flask, request, jsonify
import os
from json_gen import load_json_data, format_questions_flowable

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "Flask app is running, My Lord!"


@app.route('/generate', methods=['POST'])
def generate_from_json():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Check MIME type to be sure it's JSON
        if file.mimetype != 'application/json':
            return jsonify({"error": "Invalid file type. Only JSON files are allowed."}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Replace .json with .pdf
        filename_wo_ext = os.path.splitext(file.filename)[0]
        output_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename_wo_ext}.pdf")

        # Load and process the JSON file
        data = load_json_data(filepath)
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Use output_pdf_path to save the PDF
        formatted_data = format_questions_flowable(data, output_pdf_path)
        if not formatted_data:
            return jsonify({"error": "Failed to format JSON data"}), 500

        return jsonify({
            "message": "JSON data processed successfully",
            "pdf_path": output_pdf_path 
        }), 200


if __name__ == '__main__':
    app.run(debug=True)
