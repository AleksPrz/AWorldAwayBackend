from flask import Flask
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
    ruta_carpeta = 'tmpModels'
    if os.path.exists(ruta_carpeta):
        try:
            for nombre_archivo in os.listdir(ruta_carpeta):
                ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
                if os.path.isfile(ruta_archivo):
                    os.remove(ruta_archivo)
                elif os.path.isdir(ruta_archivo):
                    shutil.rmtree(ruta_archivo)  # Borra subcarpetas si hay
            print(f"Éxito: File of '{ruta_carpeta}' deleted.")
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