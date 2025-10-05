from flask import Flask
from routes import route_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(route_bp)

if __name__ == '__main__':
    app.run(debug=True, port= 4000)