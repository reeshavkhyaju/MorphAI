# MorphAI

AI-powered facial appearance prediction — React frontend, Flask backend, LGNet generator.

Requires Python 3.10 and Node.js 18+.

## Install

Run from the project root.

**Backend**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
pip install -r backend/requirements-arcface.txt --no-deps
```

**Frontend**

```powershell
cd frontend
npm install
cd ..
```

## Run

Two terminals.

**Terminal 1 — backend**

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

**Terminal 2 — frontend**

```powershell
cd frontend
npm run dev
```

Open **http://localhost:5173**
