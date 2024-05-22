# Plane Tracking

## Description du système

La caméra SONY EVI-D90P est utilisée pour faire du tracking d'avion. Une arduino méga 2560 est utilisée pour faire le pont entre le port USB de l'ordinateur et le port RS232 de la caméra.

Le manuel de la caméra est dans `python/docs/evid90_manual.pdf`

## Connections de l'arduino à la caméra

- Port 22 <-> RS232 IN, ligne du milieu, à gauche
- GND <-> RS232 IN, ligne du milieu, au milieu
- Port 13 <-> RS232 IN, ligne du milieu, à droite

RS232 est un protocole série symétrique (-5V / +5V) inversé (un 1 logique est codé par -5V) alors que les ports séries standards des arduino ne sont pas symétriques (0V / +5V) et pas inversés (un 1 logique est codé par +5V). Coup de bol, pour discuter avec la caméra, on peut utilise le port série virtuel (hardware serial) de l'arduino en utilisant l'option invert_logic. Le soucis du port série virtuel, c'est qu'il utilise des interrupt pour lire la donnée en provenance de la caméra, mais désactive les interrupt quand il écrit à la caméra (pour pas merder le timing si la caméra lui parle en même temps). Du coup, si la caméra et l'arduino se parlent en même temps, les bites envoyés par la caméra seront pas lus (ou mal lu) par l'arduino. Du coup ça force à être assez conservatif dans le code vis-à-vis de la communication avec la caméra. A noter de plus que ça force l'arduino à utiliser un port capable de générer des interrupts pour la lecture des signaux de la caméra.

## Comment tracker un nouvel avion

- Modifier le fichier blender pour modéliser l'avion qui vous intéresse
- Lancer le script python à l'intérieur de ce fichier blender pour créer un dataset synthétique de 1000 images d'entrainement et 1000 images de test
- Lancer l'entraînement en lançant le fichier `yolo/train_yolo_model.py`
- Choisir le bon modèle en modifiant le chemin et tenter de suivre l'avion en utilisant le fichier `yolo/yolo_kamera_kalman.py`
- Appuyer sur `t` pour lancer le tracking. Appuyer sur `Space` pour activer / désactiver l'enregistrement de la vidéo sous forme de frames.
- Si les performances du tracking ne suffisent pas, utiliser le script d'annotation `yolo/frames_to_dataset.py` pour générer un nouveau dataset.
- Renseigner le nouveau dataset dans `yolo/train_yolo_model.py`, réentraîner un model, etc...

## Limitations du système

- La commande de zoom fait merder le protocole de communication avec la caméra. Pour l'instant j'ai décidé de ne pas utiliser le zoom pendant le tracking. Ça réduit évidemment la zone effective qui peut être utilisée pour le tracking.

- Le système n'est prévu que pour tracker un avion bien spécifique, connu à l'avance (celui dans le dataset)

- Rien n'a encore été testé si il y a deux détections à la fois sur une même frame

- Le PC a besoin d'une alimentation pour faire tourner le GPU, tourner au max de sa puissance et fonctionner plus de 1h. Rien n'a été testé sans alimentation. A voir comment faire pour aller sur le terrain avec ces contraintes. 

- OpenCV est codé avec les pieds et les capacités graphiques de OpenCV ne peuvent pas être utilisés avec matplotlib + PyQt dans la même instalation de python. Du coup, ici, il faut installer opencv-python en mode headless (`pip install opencv-python-headless`).