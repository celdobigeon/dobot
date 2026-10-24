# Perception : reconnaître et localiser des pièces avec une RealSense

Une caméra **Intel RealSense D435i** regarde la table. À partir de la
**profondeur**, les scripts trouvent le plan de la table, isolent ce qui
dépasse, et reconnaissent des pièces de dimensions connues : sur quelle face
elles sont posées, où se trouve le centre de leur face supérieure, et leur
orientation. Les positions sont rendues sous forme de **matrices homogènes
4×4**, dans le repère de la caméra et dans celui de la table.

Pas d'apprentissage, pas de YOLO : pour des pièces posées sur une table, la
géométrie suffit.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `serveur_camera.py` | ouvre la caméra **une fois** et la garde ouverte ; le **seul** script qui demande `sudo` sous macOS |
| `camera.py` | réglages communs de la caméra, et le client : `camera.capturer()` demande une capture au serveur |
| `vision.py` | la reconnaissance : plan de la table, objets, face d'appui, position, orientation |
| `detecter_pieces.py` | détection sur un fichier, ou en direct (`--direct`) ; écrit une image annotée |
| `capture_realsense.py` | enregistre une capture dans `captures/` (par le serveur s'il tourne) |
| `diagnostic_realsense.py` | teste la caméra étape par étape |
| `installer_realsense_macos.sh` | compile et installe le module Python RealSense pour macOS |

## Séance type

Dans un **premier terminal**, une fois :

```sh
sudo .venv/bin/python examples/perception/serveur_camera.py
```

La **couleur** est demandée par défaut ; le message de démarrage dit si
elle est active (« avec la couleur » ou « SANS la couleur »). Si elle
n'arrive pas, le serveur se relance tout seul, sans elle :
« couleur indisponible : profondeur et infrarouge seulement ». Pour s'en
passer d'emblée : `serveur_camera.py --sans-couleur`.

Dans un **second terminal**, autant de fois qu'on veut, **sans sudo** :

```sh
.venv/bin/python examples/perception/detecter_pieces.py --direct
.venv/bin/python examples/perception/detecter_pieces.py --direct --dimensions 16 20 45
.venv/bin/python examples/perception/detecter_pieces.py --direct --enregistrer scene5
```

Chaque détection affiche les pièces reconnues et écrit
`captures/derniere_detection.png` (ou `<nom>_detection.png`).

## Installation

**macOS (Apple Silicon)** — le module de PyPI plante sous macOS 15 :

```sh
.venv/bin/pip install opencv-python
./examples/perception/installer_realsense_macos.sh      # 10 à 15 min, sans sudo
```

**Linux** — le module officiel s'installe directement :

```sh
.venv/bin/pip install opencv-python pyrealsense2
```

et l'administrateur installe une fois la règle d'accès à la caméra
(`sudo ./admin/configurer_poste_linux.sh`) : ensuite, **plus aucun `sudo`**,
pas même pour le serveur.

## À savoir

- **Sous macOS, `sudo` est inévitable pour ouvrir la caméra** : le système
  réserve l'accès USB direct à l'administrateur. Photo Booth, lui, n'utilise
  que l'image couleur, comme une simple webcam. D'où le serveur : `sudo` une
  fois par séance, et plus du tout ensuite.
- **Câble USB 3, branché directement sur l'ordinateur.** Avec un câble USB 2
  ou trop faible, la caméra se coupe dès que la profondeur démarre :
  « failed to set power state », images qui n'arrivent pas, plantages.
- **Surface mate et claire, lumière douce.** Une table vernie et sombre
  laisse la profondeur pleine de trous ; un reflet de lampe aussi. Le bois
  clair peut saturer l'infrarouge : la détection reconstitue les pixels
  saturés d'une pièce, mais une pièce entièrement saturée est perdue.
- **La pince s'ouvre d'environ 26 mm** : pour qu'elle saisisse une pièce
  quelle que soit sa face d'appui, ses deux plus petites dimensions doivent
  rester sous 22 mm environ.
- **Sous macOS, l'image couleur peut manquer** (le système garde parfois
  le capteur couleur pour lui) : la détection n'en a pas besoin, et le
  serveur se replie alors sur la profondeur et l'infrarouge. Si la caméra
  reste muette après ce repli, débranchez-la, rebranchez-la et relancez le
  serveur avec `--sans-couleur`.

### Ne plus taper le mot de passe (macOS, optionnel)

Une règle `sudoers` peut autoriser **cette seule commande** sans mot de
passe. Ajoutez avec `sudo visudo` une ligne de la forme (chemins complets) :

```
votre_compte ALL=(root) NOPASSWD: /chemin/dobot/.venv/bin/python /chemin/dobot/examples/perception/serveur_camera.py
```

**Attention :** le script et le Python du `.venv` sont modifiables par votre
compte. Quiconque peut écrire dans ces fichiers peut donc, par cette règle,
exécuter n'importe quoi en administrateur. Acceptable sur une machine
personnelle, à proscrire sur un poste partagé.
