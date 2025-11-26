from flask import Flask
import subprocess
import sys

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Trader Bot — Running"

@app.route('/run')
def run_bot():
    try:
        out = subprocess.check_output([sys.executable, "main.py"], stderr=subprocess.STDOUT)
        return out.decode()
    except Exception as e:
        return str(e)

@app.route('/health')
def health():
    return {"status": "ok"}
