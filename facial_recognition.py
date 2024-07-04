import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
import face_recognition
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import mediapipe as mp
from scipy.spatial import distance

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

data = {
    'match': False,
    'ambosOjosCerrados':False,
    'ambosOjosAbiertos':False,
    'ojoIzquierdoCerrado': False,
    'ojoDerechoCerrado': False,
    'bocaAbierta': False,
    'bocaCerrada':False,
    'sonrisa':False
}

# Load the image file
IMAGE_FILE = "annotated_image.jpg"

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def eye_vertical_distance(eye_landmarks):
    # Extract x and y coordinates from the eye landmarks
    eye_coordinates = [(landmark.x, landmark.y) for landmark in eye_landmarks]

    # Compute the vertical distance between the upper and lower eyelids
    vertical_distance = distance.euclidean(eye_coordinates[1], eye_coordinates[5])

    return vertical_distance

def mouth_open_distance(mouth_landmarks):
    # Extract x and y coordinates from the mouth landmarks
    mouth_coordinates = [(landmark.x, landmark.y) for landmark in mouth_landmarks]

    # Compute the vertical distance between the upper and lower lips
    vertical_distance = distance.euclidean(mouth_coordinates[0], mouth_coordinates[1])

    return vertical_distance

def mouth_smile_distance(mouth_landmarks):
    # Extract x and y coordinates from the mouth landmarks
    mouth_coordinates = [(landmark.x, landmark.y) for landmark in mouth_landmarks]

    # Compute the horizontal distance between the corners of the mouth
    horizontal_distance = distance.euclidean(mouth_coordinates[0], mouth_coordinates[1])

    return horizontal_distance


def detect_expressions():
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        image = cv2.imread(IMAGE_FILE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                )

                # Detectar landmarks del ojo
                left_eye_landmarks = [face_landmarks.landmark[idx] for idx in [33, 160, 158, 133, 153, 144]]
                right_eye_landmarks = [face_landmarks.landmark[idx] for idx in [362, 385, 387, 263, 373, 380]]

                # Calcular la distancia vertical del ojo para cada ojo
                left_eye_distance = eye_vertical_distance(left_eye_landmarks)
                right_eye_distance = eye_vertical_distance(right_eye_landmarks)

                # Definir un umbral para la detección de ojos cerrados basado en la distancia
                EYE_CLOSED_THRESHOLD = 0.04

                # Comprobar si algún ojo está cerrado
                if left_eye_distance < EYE_CLOSED_THRESHOLD and right_eye_distance < EYE_CLOSED_THRESHOLD:
                    data['ambosOjosCerrados'] = True
                if left_eye_distance < EYE_CLOSED_THRESHOLD:
                    data['ojoIzquierdoCerrado'] = True
                elif right_eye_distance < EYE_CLOSED_THRESHOLD:
                    data['ojoDerechoCerrado'] = True
                else:
                    data['ambosOjosAbiertos'] = True

                # Detectar landmarks de la boca
                mouth_landmarks_open = [face_landmarks.landmark[idx] for idx in [13, 14]]  # 13 y 14 son puntos en el labio superior e inferior
                mouth_landmarks_smile = [face_landmarks.landmark[idx] for idx in [61, 291]]  # 61 y 291 son los extremos de la boca

                # Calcular la distancia vertical de la boca
                mouth_distance_open = mouth_open_distance(mouth_landmarks_open)

                # Definir un umbral para la detección de la boca abierta basado en la distancia
                MOUTH_OPEN_THRESHOLD = 0.05

                # Comprobar si la boca está abierta
                if mouth_distance_open > MOUTH_OPEN_THRESHOLD:
                    data['bocaAbierta'] = True
                else:
                    data['bocaCerrada'] = True

                # Calcular la distancia horizontal de la sonrisa
                mouth_distance_smile = mouth_smile_distance(mouth_landmarks_smile)

                # Definir un umbral para la detección de la sonrisa basado en la distancia
                SMILE_THRESHOLD = 0.3

                # Comprobar si hay una sonrisa
                if mouth_distance_smile > SMILE_THRESHOLD:
                    data['sonrisa'] = True


    # Guardar la imagen anotada
    cv2.imwrite('annotated_image.jpg', image)
    return data

def load_reference_images(folder_path):
    face_encodings = []
    face_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)
            if image is not None:
                face_locations = face_recognition.face_locations(image, model="hog")
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
                    face_encodings.append(face_encoding)
                    face_names.append(os.path.splitext(file_name)[0])

    return face_encodings, face_names

def recognition(reference_folder, img):
    reference_encodings, reference_names = load_reference_images(reference_folder)

    frame = cv2.imread(img)
    face_locations = face_recognition.face_locations(frame, model="hog")

    face_detected = False

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_frame_encoding = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]

        matches = face_recognition.compare_faces(reference_encodings, face_frame_encoding)

        name = "Desconocido"
        if True in matches:
            index = matches.index(True)
            name = reference_names[index]
            face_detected = True

            color = (125, 220, 0)
            text = name

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            face_crop = frame[top:bottom, left:right]
            cv2.imwrite('annotated_image.jpg', face_crop)
            break  # Guardar solo la primera coincidencia y salir del bucle
        else:
            color = (0, 0, 255)
            text = name

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            face_crop = frame[top:bottom, left:right]
            cv2.imwrite('annotated_image.jpg', face_crop)  

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data['match'] = face_detected

    return face_detected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconocimiento facial y análisis de dirección de la cabeza y estado de la boca.')
    parser.add_argument('reference_folder', type=str, help='Ruta de la carpeta con imágenes de referencia.')
    parser.add_argument('image_path', type=str, help='Ruta de la imagen en la que realizar el reconocimiento.')

    args = parser.parse_args()
    recognition(args.reference_folder, args.image_path)
    data = detect_expressions()
    print(data)