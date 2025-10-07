from flask import Flask, current_app
from routes import route_bp
from flask_cors import CORS
from flask_apscheduler import APScheduler
import os
import shutil

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(route_bp)

scheduler = APScheduler()

def limpiar_carpeta_tmpModels():
    
    folder_path = os.path.join(current_app.root_path, 'tmp_models')
    if os.path.exists(folder_path):
        try:
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Borra subcarpetas si hay
            print(f"Success: File of '{folder_path}' deleted.")
        except Exception as e:
            print(f"Error: {e}")


@scheduler.task('interval', id='limpiar_tmp',hours=1)
def trabajo_limpieza_programado():
    limpiar_carpeta_tmpModels()


if __name__ == '__main__':
    # Configurar e iniciar el planificador con la app
    scheduler.init_app(app)
    scheduler.start()
    app.run(debug=True, port= 4000, use_reloader = False)