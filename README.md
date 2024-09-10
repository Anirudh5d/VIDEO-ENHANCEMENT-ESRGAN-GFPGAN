# AI-Driven Video Enhancement and Upscaling

## Project Description
This project aims to enhance and upscale video quality using ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) and GFPGAN (Generative Facial Prior GAN) models. The Flask API allows users to upload videos, and the system processes them to increase resolution and improve clarity, particularly in facial features and motion-heavy scenes.

## Tech Stack
- **Backend:** Flask, Python
- **Machine Learning Models:** ESRGAN, GFPGAN
- **Libraries:** PyTorch, OpenCV, Real-ESRGAN, facexlib
- **Frontend:** HTML, CSS

## Features
- **Video Upload and Processing:** Upload videos through a web interface and process them using AI models.
- **AI-based Video Enhancement:** Utilizes ESRGAN and GFPGAN to improve video resolution and clarity.
- **Smoother Motion and Clearer Visuals:** Enhances video content by fine-tuning ESRGAN for better performance on real-world video datasets.
- **Custom Weight Support:** Allows the usage of custom-trained weights for different enhancement preferences.

## Installation and Setup

### Prerequisites
- Python 3.8+
- Pip
- Virtual Environment (optional but recommended)

### Clone the repository
\`\`\`bash
git clone https://github.com/yourusername/ai-video-enhancement.git
cd ai-video-enhancement
\`\`\`

### Create a virtual environment and install dependencies
\`\`\`bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

### Download Pretrained Models
\`\`\`bash
# Download ESRGAN and GFPGAN models
mkdir models
cd models
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2/RealESRGAN_x4plus.pth
\`\`\`

### Run the Application
\`\`\`bash
flask run
\`\`\`

Visit \`http://127.0.0.1:5000\` in your browser to access the video upload and processing interface.

## Usage

1. Open the application in your browser.
2. Upload a video file in the supported formats.
3. Click the "Upload" button to enhance and upscale the video.
4. Download the processed video once it's ready.

## Directory Structure
\`\`\`
ai-video-enhancement/
│
├── app.py               # Flask API with video upload and response handling
├── templates/
│   └── index.html        # Frontend for video upload
├── static/
│   └── styles.css        # Styles for the frontend
├── utilities.py          # Contains GAN-related processing functions
├── experiments/pretrained_models               # Pretrained models (GFPGAN, RealESRGAN)
└── README.md             # Project documentation
\`\`\`

## Key Files
- **\`app.py\`:** The main file that handles the Flask API.
- **\`utilities.py\`:** Contains all functions related to video enhancement using the GAN models.
- **\`index.html\`:** Frontend for video upload.

## Known Issues
- **Model Weights:** Ensure the pretrained model files are in the correct directory (\`models/\`).
- **Reflection Padding Issue:** Current PyTorch warnings regarding reflection padding for 'Half' tensors might cause runtime errors.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOL