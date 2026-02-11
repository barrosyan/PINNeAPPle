🍍 PINNeAPPle — Physics-Informed AI Case Study: STL Field Mapping
This repository contains a full-stack implementation demonstrating how to integrate Physics-Informed Neural Networks (PINNs) with standard CAD geometry (STL) to predict and visualize physical scalar fields (e.g., temperature) directly in the browser.

🚀 Overview
The project uses the PINNeAPPle Toolkit to bridge the gap between classic Finite Difference Method (FDM) baselines and modern Deep Learning solvers. It features a Flask-based backend for heavy computation and a Three.js frontend for interactive 3D visualization.

Key Features
Geometry Ingestion: Automated STL processing and normalization using trimesh.

Smart Cache: MD5 binary hashing to reuse trained models for identical geometries.

Hybrid Solver: Combines classic Laplace/Dirichlet solvers with PINN optimization.

Interactive 3D Mapping: Real-time scalar field projection with vertex-colored PLY exports.

🛠️ Installation & Setup
1. Prerequisites
Ensure you have Python 3.9+ and a modern web browser installed.

2. Clone the Repository
Bash
git clone https://github.com/your-username/pinneapple-stl-demo.git
cd pinneapple-stl-demo
3. Setup Environment & Dependencies
It is highly recommended to use a virtual environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
4. Directory Structure Initialization
The backend requires specific folders to manage data. The script will attempt to create them, but you can ensure they exist manually:

Bash
mkdir -p runs/uploads runs/outputs runs/checkpoints
uploads/: Stores ingested STL files.

outputs/: Stores generated PLY results for inference.

checkpoints/: Stores .pt model weights and .json metadata.

🏃 How to Run
Step 1: Start the Backend
Navigate to the backend folder and run the Flask application:

Bash
# Navigate to the correct directory if needed
python app.py
The server will start at http://localhost:8000. CORS is enabled to allow requests from your frontend.

Step 2: Launch the Frontend
Since the frontend is a standalone HTML file using ES Modules (Import Maps), you should serve it through a simple local server to avoid browser security restrictions:

Bash
# Using Python's built-in server
python -m http.server 5175
Open your browser at http://localhost:5175.

🔄 The Pipeline Workflow
Ingest Geometry: Upload an .stl file. The backend generates a unique MD5 hash for the content.

Optimize Model: Click Train Model.

If a model for that hash exists, it loads instantly (Cache Hit).

Otherwise, it solves the baseline and trains the PINN in background threads.

Execute Inference: Once "Optimized", click to run high-speed inference. The scalar field is mapped to the mesh and rendered in 3D.

🧪 Technical Stack
Backend: Flask, PyTorch, Trimesh, SciPy.

Frontend: HTML5, CSS3 (Modern Dashboard), Three.js (WebGL).

Core AI: PINNeAPPle Physics AI Toolkit.

📄 License
Distributed under the MIT License. See LICENSE for more information.
