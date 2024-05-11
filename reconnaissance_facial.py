# Importation des bibliothèques nécessaires
import cv2
import numpy as np

# Chargement du modèle pré-entraîné pour la reconnaissance faciale
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialisation de la webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture d'une frame
    ret, frame = cap.read()
    
    # Conversion en niveaux de gris pour une meilleure détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détection des visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Dessin des rectangles autour des visages détectés
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
    # Affichage de l'image avec les visages détectés
    cv2.imshow('Reconnaissance faciale', frame)
    
    # Quitter la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération de la webcam et fermeture de la fenêtre d'affichage
cap.release()
cv2.destroyAllWindows()
