import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from utilities import allowed_file, enhance_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size: 100MB


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_video_path)

        output_filename = 'enhanced_' + filename
        output_video_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        enhance_video(input_video_path, output_video_path)

        return redirect(url_for('download_file', filename=output_filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
