import cv2
import numpy as np

# Charger le modèle YOLOv3 pré-entraîné
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Charger les noms de classes
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialiser les couleurs pour chaque classe d'objet détectée
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Définir les paramètres du réseau
conf_threshold = 0.5
nms_threshold = 0.4

# Initialiser la capture vidéo
cap = cv2.VideoCapture(0)

# Boucle principale de détection d'objets
while True:
    # Lire une image depuis la vidéo
    ret, img = cap.read()
    
    # Transformer l'image pour l'analyse du modèle YOLOv3
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Initialiser les listes de détection d'objets
    boxes = []
    confidences = []
    classIDs = []
    
    # Analyser les sorties de chaque couche de détection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    # Appliquer la non-maximum suppression pour éliminer les détections redondantes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # Afficher les résultats de la détection d'objets dans l'image
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.2f}%'.format(classes[classIDs[i]], confidences[i] * 100)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Afficher l'image avec les résultats de la détection d'objets
    cv2.imshow('Object Detection', img)
    
    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) == ord('q'):
    break
    
     # Libérer les ressources
     # cap.release()
      # cv2.destroyAllWindows()

# Load the pre-trained model and classes
model = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
classes = open(args["classes"]).read().strip().split("\n")

# Initialize the video stream and start processing frames
vs = cv2.VideoCapture(args["video"] if args["video"] else 0)
writer = None
while True:
    # Grab the next frame
    (grabbed, frame) = vs.read()

    # If the frame could not be grabbed, we have reached the end of the video
    if not grabbed:
        break

    # Preprocess the frame by resizing it and normalizing its pixel values
    frame = imutils.resize(frame, width=400)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and get the detections
    model.setInput(blob)
    detections = model.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args["confidence"]:
            # Extract the index of the class label from the detection and compute the (x, y)-coordinates of the bounding box
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If an output video file path has been specified, write the output frame to the file
    if args["output"]:
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

    # Show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.release()
if writer is not None:
    writer.release()
