from flask import Flask
from bicep_curls import video_feed
from app import app

app.register_blueprint(video_feed, url_prefix='/pose')

if __name__ == '__main__':
    app.run(debug=True)
