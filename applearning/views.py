from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
import tensorflow as tf
import joblib

# Create your views here.
class PredictCareerView(APIView):
    def get(self, request):
        # healtcheck
        return Response({"message": "Toy vivito"}, status=status.HTTP_200_OK)
    
    def post(self, request, *args, **kwargs):
        # Datos del nuevo estudiante
        nuevo_estudiante = request.data.get('respuestas', None)
        if nuevo_estudiante is None or not isinstance(nuevo_estudiante, list):
            return Response({"error": "Invalid input. Please provide a list of responses."}, status=status.HTTP_400_BAD_REQUEST)

        if len(nuevo_estudiante) != 69:
            return Response({"error": f"Se envío {len(nuevo_estudiante)} respuestas, pero se requieren 69."}, status=status.HTTP_400_BAD_REQUEST)

        # Cargar el scaler y el modelo de la red neuronal
        scaler = joblib.load('applearning/scaler.pkl')
        model = tf.keras.models.load_model('applearning/modelo_vocacional_tf.h5')
        
        # Convertir las respuestas a un DataFrame y escalar los datos
        nuevo_estudiante = pd.DataFrame([nuevo_estudiante])
        nuevo_estudiante_escalado = scaler.transform(nuevo_estudiante)
        
        # Hacer predicciones
        prueba = model.predict(nuevo_estudiante_escalado)[0]
        
        # Diccionario de carreras
        carreras = {
            0: "Diseño y desarrollo de software",
            1: "Diseño y Desarrollo de Simuladores y Videojuegos",
            2: "Administración de Redes y Comunicaciones",
            3: "Big Data y Ciencia de Datos",
            4: "Aviónica y Mecánica Aeronáutica",
            5: "Gestión y Mantenimiento de Maquinaria Industrial",
            6: "Gestión y Mantenimiento de Maquinaria Pesada",
            7: "Operaciones Mineras",
            8: "Procesos Químicos y Metalúrgicos",
            9: "Diseño Industrial",
            10: "Producción y Gestión Industrial",
            11: "Electrónica y Automatización Industrial",
            12: "Mecatrónica Industrial",
            13: "Electricidad Industrial con mención en Sistemas Eléctricos de Potencia"
        }
        
        # Obtener los índices de las carreras con mayor probabilidad
        indices = prueba.argsort()[-3:][::-1]
        
        # Mapear los índices a las carreras
        carreras_predichas = [carreras[indice] for indice in indices]
        
        return Response(
            {
                "status": "success",
                "message": "Predicción exitosa",
                "carreras": carreras_predichas
            }, 
            status=status.HTTP_200_OK)
