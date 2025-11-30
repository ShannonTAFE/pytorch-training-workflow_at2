PyTorch Training Workflow (Colab GPU)

A clean and reproducible workflow for training a CIFAR-10 classifier using PyTorch.
The project is designed to run in Google Colab with GPU enabled, with optional support for VS Code Remote.

📌 Features

CIFAR-10 dataset loading and preprocessing

Custom PyTorch model & training loop

Validation and test evaluation

Confusion matrix generation

Checkpoint saving (best_model.pth)

Optional Weights & Biases logging

Supports CPU, GPU, MPS, and Colab environments

🖥️ 1. Environment Requirements
Hardware

Google Colab GPU (T4 / P100 / A100)

Software

Python 3.10+

PyTorch + Torchvision

Git

(Optional) Weights & Biases

Dependencies

Install all dependencies with:

pip install -r requirements.txt

🚀 2. Getting Started (Google Colab)
Step 1 — Enable GPU

Open Google Colab → New Notebook

Go to Runtime → Change runtime type

Select Hardware accelerator → GPU

Verify GPU:

import torch
torch.cuda.is_available()


Expected output:

True

Step 2 — Clone This Repository
git clone https://github.com/yourusername/pytorch-training-workflow_at2.git
cd pytorch-training-workflow_at2

Step 3 — Install Dependencies
pip install -r requirements.txt


Expected output (truncated):

Successfully installed torch ... torchvision ... numpy ... tqdm ...

Step 4 — Run Training
python train.py --epochs 3 --batch-size 128


Example output:

Epoch 1/3 - Loss: 1.87 - Acc: 33%
Epoch 2/3 - Loss: 1.41 - Acc: 48%
Epoch 3/3 - Loss: 1.17 - Acc: 58%
Saving best_model.pth
Saving confusion_matrix.npy

Optional: Use the Launcher Script
chmod +x run_distributed.sh
./run_distributed.sh --epochs 3 --batch-size 128


Log files will appear in:

runs/

👨‍💻 Optional: VS Code Remote (Colab)

If you want to work entirely inside VS Code:

Enable SSH or VS Code Tunnels in the Colab VM

Connect using VS Code → Remote Explorer

Open the project folder

Run the training script from VS Code’s terminal

(See screenshots in the evidence/ folder.)

📂 Project Structure
pytorch-training-workflow/
│── train.py
│── run_distributed.sh
│── requirements.txt
│── outputs/
│   ├── best_model.pth
│   └── confusion_matrix.npy
│── runs/
│── evidence/
│── README.md

🧪 Sample Outputs
Confusion Matrix

Saved automatically as:

outputs/confusion_matrix.npy

Checkpoint

Saved as:

outputs/best_model.pth

🛠️ Troubleshooting
GPU not detected

Ensure GPU is enabled in Runtime settings

Restart the runtime after switching hardware

“torch not found”

Install manually:

pip install torch torchvision

Out of Memory (OOM)

Reduce batch size:

python train.py --batch-size 64

Git access issues

Ensure the repository is public or use HTTPS cloning.

📘 Mapping to Unit of Competency (UoC)
UoC Requirement	How This Project Meets It
Configure environment	Colab GPU setup, verification, dependency installation
Install dependencies	requirements.txt + successful installation in Colab
Execute ML workflow	Training script, validation, testing, saved outputs
Use tools for execution	Colab GPU, optional VS Code Remote
Document workflow	Clear README, evidence folder, mapping file

A detailed mapping statement is included in:

evidence/mapping_statement_uoc.md

✅ Summary

This repository provides a simple, reproducible, and well-documented ML workflow using PyTorch, designed specifically for Google Colab GPU environments and suitable for training, experimentation, and assessment requirements.
