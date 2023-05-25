import cv2

img = cv2.imread('OM.jpg')
#Importam imaginea dorita pentru detectarea oamenilor

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
  # Se deschide fisierul 'coco.names'

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# Calea catre modelul pre-antrenat folosit pentru detectarea persoanelor

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)
print(classIds, bbox)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    className = classNames[classId - 1]
 # classId reprezintă identificatorul clasei pentru obiectul detectat.
 # confidence reprezintă coeficientul corectitudine asociat obiectului detectat.
 # box reprezintă dreptunghiul pentru obiectul detectat.
    if className == 'person':
        cv2.rectangle(img, box, color=(0, 0, 255), thickness=1)
        # Am adaugat un dreptunghi in jurul fiecarui om detectat
        cv2.putText(img, className, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)
        # Am scirs persoana deasupra fiecarui om detectat
        cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 2)
        # Procentul de acuratețe

with open('detectie.log', 'a') as file:
    # Scrie numele imaginii
    file.write("Nume imagine: {}\n".format('om.jpg'))
    # Scrie coordonatele si precizia pentru fiecare obiect detectat

    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        className = classNames[classId - 1]
        if className == 'person':
            file.write("Obiect detectat: {}\n".format(className))
            # scrie numele clasei obiectului detectat în fișierul de log
            file.write("Coordonate: {}, {}, {}, {}\n".format(box[0], box[1], box[2], box[3]))
            # scrie coordonatele casetei încadratoare în fișierul de log
            file.write("Precizie: {}%\n".format(round(confidence * 100, 2)))
            # scrie coeficientul de detectare în fișierul de log
            file.write("\n")

cv2.imshow("Output", img)
cv2.waitKey(0)
# Se afiseaza imaginea dupa identificarea persoanelor

# Bibliografie: https://www.youtube.com/watch?v=HXDD7-EnGBY&ab_channel=Murtaza%27sWorkshop-RoboticsandAI