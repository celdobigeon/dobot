# Perception : reconnaître et localiser des pièces avec une RealSense

Une caméra **Intel RealSense D435i** regarde la table. À partir de la
**profondeur**, les scripts trouvent le plan de la table, isolent ce qui
dépasse, et reconnaissent des pièces de dimensions connues : sur quelle face
elles sont posées, où se trouve le centre de leur face supérieure, et leur
orientation. Les positions sont rendues sous forme de **matrices homogènes
4×4**, dans le repère de la caméra et dans celui de la table.

Trois étages, du plus simple au plus robuste :

1. **la géométrie** (profondeur seule) : sans apprentissage, elle trouve
   l'essentiel des pièces posées à plat ;
2. **un réseau YOLO11-seg** (`modeles/pieces-seg.pt`, entraîné dans
   `apprentissage/`) segmente les pièces dans l'image couleur ;
3. **l'affinage de la scène** (`raffinement.py`) confronte le tout à la
   profondeur : pièces inclinées, empilées, cachées, et pièces que le réseau
   a manquées (fond inconnu).

`Perception()` enchaîne les trois ; sans modèle entraîné, la géométrie et
l'affinage suffisent à faire marcher la détection.

## Les fichiers

| Fichier | Rôle |
|---|---|
| `serveur_camera.py` | ouvre la caméra **une fois** et la garde ouverte ; le **seul** script qui demande `sudo` sous macOS |
| `camera.py` | réglages communs de la caméra, et le client : `camera.capturer()` demande une capture au serveur |
| `vision.py` | la reconnaissance : plan de la table, objets, face d'appui, position, orientation |
| `perception.py` | la bibliothèque à utiliser : réseau (s'il est entraîné) + géométrie + affinage |
| `raffinement.py` | affinage de la scène entière « rendre et comparer » : pièces inclinées, occultations, physique |
| `evaluer_detection.py` | mesures chiffrées (scènes synthétiques, jeu réel annoté, captures) |
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

## Pièces empilées

Une pièce posée **sur une autre** a sa face supérieure à la hauteur de sa
porteuse **plus** sa propre épaisseur : une épaisse à plat sur une épaisse
à plat est vue à 34 mm, une fine à plat sur une épaisse sur la tranche à
27 mm. La détection (`vision_modele.py`, et l'ajustement des masques du
réseau dans `perception.py`) essaie donc chaque pièce posée sur la table
**ou** sur le sommet d'une pile (8, 17, 19, 16, 25, 27, 34… mm) : c'est
l'**épaisseur** (sommet − base) qui dit le type et la face. Une pile doit
se prouver : la porteuse est détectée sous la pièce, ou elle dépasse
autour d'elle ; sinon la pièce est reposée sur la table (pas de pile
fantôme). Une porteuse en grande partie cachée est quand même cherchée et
rendue, avec une confiance plus faible (`cachee` : part de sa face cachée).

Chaque pièce porte en plus :

| Champ | Sens |
|---|---|
| `support` | numéro de la pièce porteuse ; `None` : sur la table (ou, si `hauteur_base > 0`, sur un objet non identifié) |
| `hauteur_base` | hauteur de sa face d'appui au-dessus de la table, en mm |
| `dessus` | numéros des pièces posées sur elle (`piece.libre` : rien dessus) |

```python
from perception import Perception, ordre_de_prise
p = Perception()
for piece in ordre_de_prise(p.detecter()):     # les libres d'abord, du haut vers le bas
    print(piece.numero, piece.type, piece.face, piece.support, piece.dessus)
```

**Limites.** Sans l'affinage (`Perception(raffinement=False)`), seules les
pièces **horizontales** sont décrites : une pièce qui s'appuie d'un bout
sur la table et de l'autre sur une pièce (inclinée, cas fréquent dans un
tas) est mal lue ; voir la section suivante. Deux fines sur la tranche accolées
et une fine à plat sur une fine à plat se ressemblent beaucoup vu de
dessus. En vue rasante, la pièce du dessus cache sa porteuse et projette
une grande ombre : les piles y sont souvent manquées. Chiffres et
méthode : `evaluer_detection.py --empilement` et l'en-tête de
`vision_modele.py`.

## Pièces inclinées, tas : l'affinage de la scène

Dans un vrai tas, beaucoup de pièces sont **inclinées** (un bout sur la
table, l'autre sur l'arête ou contre le flanc d'une pièce), et se cachent
les unes les autres. `perception.py` termine donc par un **affinage de la
scène entière** (`raffinement.py`, actif par défaut) : les détections (le
réseau s'il est entraîné, la géométrie de `vision_modele.py`) ne sont plus
que des *propositions* ; une scène 3D de boîtes de dimensions connues est
**rendue** dans les vues de la caméra (lancer de rayons, occultations
comprises) et **corrigée** jusqu'à expliquer la profondeur mesurée, sous
des contraintes physiques (pas d'interpénétration, appui sur la table ou
sur une pièce, centre de gravité entre les appuis). Une pièce qui
n'explique rien est retirée ; une zone que rien n'explique devient une
nouvelle pièce, éventuellement inclinée.

Le réseau n'est qu'un indice : il ne peut pas imposer une pièce que la
profondeur contredit, et une pièce qu'il manque (fond inconnu, autre
éclairage) est retrouvée par la géométrie seule.

Chaque pièce porte en plus :

| Champ | Sens |
|---|---|
| `inclinaison` | angle du grand côté avec la table, en degrés (0 : à plat) |
| `appuis` | numéros de **toutes** les pièces sur lesquelles elle s'appuie (`support` : la principale) |
| `confiance` | part de ses pixels expliqués, marge avec la meilleure autre lecture (type, face), physique |

`T_table` et `T_camera` décrivent alors la face supérieure **inclinée**.
Une pièce inclinée sur une autre est prise **avant** elle
(`ordre_de_prise`).

```python
p = Perception()                         # affinage compris
p = Perception(raffinement=False)        # les détections telles quelles
for piece in ordre_de_prise(p.detecter()):
    print(piece.numero, piece.type, piece.face, round(piece.inclinaison), piece.support)
```

**Mesures** (`evaluer_detection.py --raffinement`). Jeu réel annoté, les
40 images de validation (que le réseau n'a pas apprises), scène estimée
rendue dans l'image couleur et comparée aux masques annotés :

| Méthode | Rappel | Précision | IoU | Classe | s/image |
|---|---|---|---|---|---|
| géométrie seule | 0,79 | 0,85 | 0,82 | 0,86 | 2,3 |
| géométrie + affinage | **0,90** | **0,94** | 0,86 | 0,93 | 3,2 |
| YOLO | 0,98 | 0,98 | 0,90 | 0,94 | 0,7 |
| YOLO + affinage | 0,99 | 0,99 | 0,91 | 0,95 | 3,7 |
| YOLO abîmé (masques perdus, fusionnés, faux sur le fond) | 0,66 | 0,93 | 0,87 | 0,95 | 0,5 |
| YOLO abîmé + affinage | **0,94** | 0,95 | 0,89 | 0,93 | 3,7 |
| **YOLO v2 + affinage** (modèle installé) | **0,99** | **0,98** | 0,91 | **0,98** | 3,2 |

Les cinq premières lignes sont mesurées avec le réseau v1 (images réelles
seules). Le modèle installé est **v2** : images réelles + 2000 images
rendues à fonds variés (`apprentissage/synthese_images.py`). Sur la
nappe d'entraînement, il fait jeu égal avec v1 et se trompe moins de
classe ; sur des fonds jamais vus (synthétiques), mAP50 des masques 0,89
contre 0,11 pour v1, et 0,96 contre 0,72 avec de vraies pièces collées sur
ces fonds. v1 reste disponible : `Perception(modele="modeles/pieces-seg-v1.pt")`.

Scènes synthétiques avec 3 pièces inclinées sur 8 (vue de dessus) :
rappel 0,84 → 0,94, fausses 8 → 1, erreur d'inclinaison des inclinées
11° → 0,3°, leur support juste 0,17 → 0,68, position 1,5 → 0,5 mm. Les
scènes à plat ne changent presque pas (rappel 1,00, 0,4 mm). En vue
rasante (caméra à 1 m), le gain demeure (rappel 0,62 → 0,72 avec les
inclinées) mais les erreurs restent de 2 à 4 mm.

**Limites.** L'affinage ajoute 1 à 3 s par image (8 pièces) ; avec le
réseau, la détection géométrique, qui le rend robuste au fond, coûte
encore 2 à 3 s. Une épaisse à plat et une fine sur la tranche accolées
(même hauteur, 19 + 8 mm de large) restent ambiguës pour la profondeur :
seul le réseau (la couleur) tranche. Les pièces très lointaines au bord
de la table, en vue rasante, sont souvent manquées. Une surface plane
étrangère (livre, planche) n'est pas pavée de pièces, mais une pièce
posée dessus peut être manquée.

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
