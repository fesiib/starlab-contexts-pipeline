from flask import Flask
from flask_cors import CORS
from flask import request, send_file, redirect, url_for

from pathlib import Path

app = Flask(__name__)

CORS(app, origins=["http://localhost:7777", "http://internal.kixlab.org:7777"])
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["UPLOAD_EXTENSIONS"] = [".mp4", ".jpg", ".png", "webm"]


ROOT = Path('.')

DATABASE = ROOT / 'static' / 'database'

def launch_server():
    app.run(host="0.0.0.0", port=7778)

if __name__ == "__main__":
    #test_video("https://www.youtube.com/live/4LdIvyfzoGY?feature=share")
    #test_video("https://youtu.be/XqdDMNExvA0")
    #test_video("https://youtu.be/pZ3HQaGs3uc")
    launch_server()