# Reconocimiento Facial y Análisis de Expresiones 🤖

Este proyecto utiliza técnicas de reconocimiento facial y análisis de expresiones faciales para detectar y reconocer rostros en imágenes, así como para identificar ciertas expresiones faciales. Se basa en las bibliotecas `cv2`, `face_recognition`, y `mediapipe`.

## Funcionalidades ⚙️

- **Reconocimiento Facial**: Compara caras en una imagen dada con una base de datos de imágenes de referencia y determina si hay una coincidencia.
- **Análisis de Expresiones**: Detecta si los ojos están cerrados o abiertos, si la boca está abierta o cerrada, y si hay una sonrisa.

## Requisitos 📚

- Python 3.x
- OpenCV
- face_recognition
- mediapipe
- scipy

Puedes instalar las dependencias necesarias ejecutando (este comando varia dependiento el sistema y versiones de python):
```bash
py -m install opencv-python face_recognition mediapipe scipy
```

## Uso 💻
Para ejecutar el reconocimiento facial y análisis de expresiones, utiliza el siguiente comando:
```bash
py facial_recognition.py <ruta_de_la_carpeta_de_referencia> <ruta_de_la_imagen>
```
- <ruta_de_la_carpeta_de_referencia>: La ruta a la carpeta que contiene las imágenes de referencia para el reconocimiento facial.
- <ruta_de_la_imagen>: La ruta a la imagen en la que deseas realizar el reconocimiento y análisis.

## Resultados 📈
Los resultados del análisis de expresiones se guardan en un diccionario data que contiene las siguientes claves:

- match: Indica si se encontró una coincidencia en el reconocimiento facial.
- ambosOjosCerrados
- ambosOjosAbiertos
- ojoIzquierdoCerrado
- ojoDerechoCerrado
- bocaAbierta
- bocaCerrada
- sonrisa

# 😉😁😶😮😑😐

¡Gracias por explorar mi proyecto de reconocimiento facial y analisis de expresiones! 🤖😁
