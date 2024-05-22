# Plane Tracking

## Description du système

La caméra SONY EVI-D90P est utilisée pour faire du tracking d'avion. Une arduino méga 2560 est utilisée pour faire le pont entre le port USB de l'ordinateur et le port RS232 de la caméra.

Le manuel de la caméra est dans `python/docs/evid90_manual.pdf`

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