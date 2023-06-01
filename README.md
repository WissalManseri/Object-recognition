# Object-recognition-
Object recognition: by using deep learning models to detect and classify objects in real-time video.


Le code utilise un modèle de Deep Learning appelé YOLOv3 pour détecter des objets en temps réel dans une vidéo en utilisant la webcam de l'ordinateur.

Il charge le modèle pré-entraîné à partir des fichiers 'yolov3.cfg' et 'yolov3.weights'. Ensuite, il lit les noms de classe à partir du fichier 'coco.names'.

Le code utilise ensuite la webcam pour capturer une image à la fois. Il transforme l'image en une forme appropriée pour l'entrée du modèle YOLOv3.

Ensuite, il applique le modèle à l'image pour détecter des objets.

Une fois que le modèle a détecté des objets, le code utilise la technique de Non-Maximum Suppression pour éliminer les détections redondantes.

Enfin, le code dessine des boîtes englobantes autour des objets détectés et affiche le résultat de la détection d'objets dans l'image.

Le code est écrit en Python et utilise la bibliothèque OpenCV pour la capture vidéo et la détection d'objets.


# Prérequis
  Python 3.x
  OpenCV
  NumPy
# Installation
1. Clonez ce dépôt:

      git clone https://github.com/WISSAL-MN/Object-recognition-.git
2. Installez les dépendances requises:

       pip install -r requirements.txt
3.Téléchargez le modèle YOLOv3 pré-entraîné depuis le site web de l'auteur et placez les fichiers yolov3.cfg et yolov3.weights dans le répertoire racine du projet.

4.Téléchargez le fichier coco.names depuis le site web de l'auteur et placez-le également dans le répertoire racine du projet.

# Utilisation
Pour exécuter le programme, ouvrez une fenêtre de terminal dans le répertoire racine du projet et exécutez la commande suivante:


      python object_detection.py
      
Lorsque le programme démarre, il ouvre une fenêtre de la webcam ou du fichier vidéo spécifié et commence à détecter et classifier des objets en temps réel.
Vous pouvez appuyer sur la touche q pour quitter le programme à tout moment.

# Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE pour plus d'informations.

# Auteur
Ce projet a été développé par wissalmanseri@gmail.com.
