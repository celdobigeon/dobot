# Apprentissage : un réseau qui voit les pièces, la géométrie qui les mesure

La détection purement géométrique (`vision.py`, `vision_modele.py`) se
fonde sur la profondeur. Or le bois clair sature souvent l'infrarouge, la
profondeur y est trouée, et deux pièces qui se touchent forment une seule
tache. Ici, un réseau de **segmentation d'instance** (YOLO11-seg, affiné
sur nos propres images) trouve les pièces dans l'image **couleur** ; puis la
**géométrie** ajuste à chaque masque une pièce de dimensions connues posée
sur la table, et rejette ce qu'aucune pièce n'explique (une règle, un motif
de la nappe). Le résultat est le même que celui de la géométrie : des
matrices 4×4.

```
collecter.py ──► pre_annoter.py ──► annoter.py ──► entrainer.py ──► perception.py
 (caméra)         (géométrie+SAM)    (vous)          (exporter.py      (bibliothèque)
                                                      + copier-coller)
```

| Fichier | Rôle |
|---|---|
| `collecter.py` | session de collecte : fenêtre caméra, capture à la touche ou automatique |
| `pre_annoter.py` | masques proposés : géométrie projetée dans l'image → SAM 2.1 ; + le réseau déjà entraîné s'il existe |
| `annoter.py` | revue et correction rapides (clics, SAM), statut « à revoir » / « validée » |
| `exporter.py` | export YOLO-seg (polygones, `data.yaml`, découpage stable) |
| `copier_coller.py` | augmentation par copier-coller de pièces découpées |
| `synthese_images.py` | images **rendues** par randomisation de domaine (pièces 3D, fonds variés), étiquettes exactes |
| `entrainer.py` | fine-tuning, validation, installation de `modeles/pieces-seg.pt` |
| `commun.py` | dossiers, format des annotations, SAM |
| `../perception.py` | **la bibliothèque** pour les programmes (et les étudiants) |

Tout ce qui est produit (images, annotations, entraînements) va dans
`dataset/` et `runs/`, ignorés par git ; le modèle final dans
`../modeles/pieces-seg.pt`.

## 0. Relancer le serveur de caméra (une fois)

Le serveur fournit maintenant, avec l'image couleur **brute** (entière,
sans trous), sa géométrie : intrinsèques `K_couleur` et passage
profondeur → couleur `T_couleur_profondeur`. Sans elle, on ne peut se servir
que de la couleur *recalée*, trouée partout où la profondeur manque — c'est
justement sur les pièces. Relancez-le :

```sh
sudo .venv/bin/python examples/perception/serveur_camera.py
```

puis vérifiez, dans un autre terminal :

```sh
cd examples/perception && ../../.venv/bin/python -c "import camera; print(sorted(camera.capturer(1)))"
```

La liste doit contenir `K_couleur` et `T_couleur_profondeur`.

## 1. Collecte (≈ 30 à 45 min)

```sh
.venv/bin/python examples/perception/apprentissage/collecter.py
```

Dans la fenêtre : **Entrée** ou **Espace** = capture, **a** = mode
automatique (une capture toutes les 3 s, *seulement* si la scène a changé
et ne bouge plus : on déplace les pièces, on retire la main, ça capture),
**f** = fond sans pièce, **q** = fin. Les conseils défilent en bas.

**Combien ?** Visez **200 images** pour un premier modèle fiable (150 au
minimum), puis complétez là où il se trompe. Mieux vaut 200 scènes
variées que 1000 presque identiques. Plus **10 à 20 fonds** (touche f) :
table nue, chaque nappe, sous chaque éclairage.

**Quoi varier** (c'est ce qui fait la robustesse) :

- **positions** : toute la table, près et loin, aux bords, en partie hors champ ;
- **orientations** : tous les angles, pas seulement face à la caméra ;
- **faces d'appui** : à plat, sur la tranche, debout ;
- **contacts** : côte à côte, bout à bout, en T, en tas, une pièce qui en
  cache une autre — le cas où la géométrie seule échoue ;
- **nombre** : de 0 à 10 pièces, les deux types mêlés ;
- **éclairage** : lampe, jour, ombres portées, reflets ;
- **fonds** : plusieurs nappes, table nue, papier, tissu sombre ;
- **intrus** : règle, crayon, gomme, main : le réseau doit apprendre
  qu'ils ne sont pas des pièces (les images sans pièce comptent aussi).

Si la caméra doit bouger un jour, collectez aussi depuis la nouvelle place.

Pour ajouter des captures déjà enregistrées :
`collecter.py --importer examples/perception/captures/*.npz`.

## 2. Pré-annotation (automatique, ≈ 5 s par image)

```sh
.venv/bin/python examples/perception/apprentissage/pre_annoter.py
```

Pour chaque nouvelle image : les pièces trouvées par la géométrie sont
projetées dans l'image couleur (les 8 coins de chaque parallélépipède), ce
qui donne une boîte à **SAM 2.1**, qui découpe le masque précis ; la
classe (le type) vient de l'ajustement géométrique. Dès qu'un modèle est
entraîné, ses détections s'ajoutent : chaque tour rend le suivant plus
rapide. Un coup d'œil : `dataset/apercus/`.

Une annotation **validée** n'est jamais écrasée. `--refaire` refait les
autres ; `--sans-reseau` n'utilise que la géométrie.

## 3. Revue (≈ 10 à 20 s par image)

```sh
.venv/bin/python examples/perception/apprentissage/annoter.py
```

Les images « à revoir » défilent. Pour chacune : si tout est juste,
**Espace** (validée, enregistrée, image suivante). Sinon :

| Action | Effet |
|---|---|
| clic sur une pièce | la sélectionner |
| clic ailleurs | **ajouter** une pièce (SAM sur ce point) |
| glisser | ajouter une pièce par une **boîte** (plus sûr pour une pièce collée à une autre) |
| Maj + clic | point **positif** : étendre le masque sélectionné jusque-là |
| Ctrl/Alt + clic | point **négatif** : retirer cette zone du masque |
| **m** | si Maj/Ctrl ne passent pas : le clic gauche devient « point + » puis « point − » |
| clic droit | supprimer la pièce sous la souris |
| **1**, **2**… | classe de la pièce sélectionnée (`1=epaisse 2=fine`) |
| Tab / x / z | pièce suivante / supprimer / annuler |
| n / p | image suivante / précédente |
| v / s / h / q | statut / enregistrer / masquer les masques / quitter |
| **r** | **rejeter** l'image (annotation impossible ou fausse) : exclue de l'entraînement et des revues ; `r` à nouveau pour la reprendre |

**Règles d'annotation** (à tenir pour toutes les images) : le masque couvre
**toute la partie visible** de la pièce (dessus *et* côtés visibles), pas
son ombre ; une pièce coupée par le bord de l'image ou en partie cachée est
annotée pour sa partie visible ; une pièce cachée aux trois quarts peut
être laissée de côté ; les intrus ne sont **pas** annotés.

`--tout` pour tout revoir, `--depuis 0120` pour reprendre.

## 4. Entraînement

```sh
.venv/bin/python examples/perception/apprentissage/entrainer.py
```

Export (images **validées** seulement ; 20 % en validation selon le nom de
l'image, donc toujours les mêmes), copier-coller (2 images synthétiques par
image d'entraînement, faites de pièces découpées collées ailleurs, à
l'échelle de la perspective, souvent en contact), puis fine-tuning de
`yolo11s-seg` sur le GPU (`mps`), 100 époques au plus (arrêt anticipé si la
validation stagne). Le meilleur poids est copié dans
`examples/perception/modeles/pieces-seg.pt` (l'ancien est gardé :
`pieces-seg.precedent.pt`), et le rapport dans `runs/<nom>/rapport.json`.

Durée (Apple M4 Max) : environ 0,12 s par image et par époque, soit, pour
200 images + 400 synthétiques, ≈ 70 s par époque et **1 h 30 à 2 h** au
total ; `--modele yolo11n-seg.pt` va environ trois fois plus vite, un peu
moins précis.

**Lire le rapport** : `masque_mAP50` au-dessus de 0,9 et `masque_mAP50_95`
au-dessus de 0,7 sont de bons scores pour des objets aussi simples. Les
images `runs/<nom>/val_batch*_pred.jpg` montrent les erreurs : collectez
alors des scènes qui leur ressemblent, pré-annotez (le réseau aide
désormais), revoyez, réentraînez.

`entrainer.py --demo` fait un entraînement court (15 époques, images non
validées comprises) pour vérifier que la chaîne tourne ; il installe
`pieces-seg.demo.pt`, que la bibliothèque **n'utilise pas** par défaut.

## 4 bis. Images synthétiques : randomisation de domaine

Le jeu réel n'a qu'**une nappe**, un éclairage, une pose de caméra : un
réseau entraîné dessus risque d'apprendre le **fond** (« une pièce, c'est ce
qui n'est pas une fleur rose sur du blanc ») et d'échouer sur une autre
table. `synthese_images.py` rend des milliers de scènes où **tout varie
sauf ce qui définit une pièce** — un parallélépipède de dimensions connues,
en bois clair — et dont les étiquettes sont exactes par construction.

```sh
# une fois (≈ 1 min) : textures de bois tirées des pièces annotées
.venv/bin/python examples/perception/apprentissage/synthese_images.py --textures
# 3000 images d'entraînement (≈ 0,07 s par image sur 6 processus)
.venv/bin/python examples/perception/apprentissage/synthese_images.py --n 3000
# un coup d'œil : synthetique/entrainement/controle.jpg (images + masques)
```

**Ce qui est tiré au hasard, image par image :**

- **la caméra** : la vraie (un fond réel de `dataset/fonds/`, avec le plan
  de la table mesuré par la profondeur), ou une pose voisine, ou une pose
  très différente (10 à 45 cm de haut, de 8 à 70° de plongée, roulis) ;
- **la table** (le point central) : le fond réel tel quel ; la **même pièce
  avec une autre nappe** (la zone de table est retexturée, l'éclairage réel
  gardé) ; ou une table entièrement synthétique devant un décor. Les
  textures viennent de **familles** : `uni` (dont des teintes de bois clair,
  le cas difficile), `bruit` (marbre, béton, granit), `tissu` (rayures,
  vichy, tissage), `motifs` (fleurs, pois, feuillages), `papier`, `bois`
  (planches, y compris claires), `carrelage`, `nappe` (la vraie, redressée
  vue de dessus, parfois recolorée), `photo` (morceaux de la pièce) ;
- **les pièces** : 0 à 10, posées à plat, sur la tranche, debout, **en
  contact**, **empilées**, **inclinées** (un bout sur la table, l'autre
  appuyé sur une pièce : l'angle est celui où la face inférieure touche
  l'obstacle le plus haut), jamais interpénétrées (test des axes
  séparateurs) ; plus des **intrus** non étiquetés (règle, crayon, gomme,
  feuille, cube, trousse) ;
- **le rendu** : lancer de rayons par face avec z-buffer, sur-échantillonné
  2 × 2 ; textures de bois **réelles** (faces des pièces annotées, ajustées
  par la géométrie puis redressées par homographie) ou procédurales ; bois
  de bout plus sombre ; ombrage de Lambert ; **ombres portées** exactes
  (rayon vers la lumière) adoucies, assombrissement de contact ;
- **la caméra encore** : flou, bougé, exposition, gamma, balance des blancs,
  vignettage, bruit de capteur, compression JPEG.

Chaque image a son masque **visible** par pièce (occultations comprises),
au format YOLO-seg, et dans `meta/` la pose de chaque pièce et les **coins
de sa face supérieure** dans l'image (pour un futur modèle de points clés ;
`--profondeur` écrit aussi la profondeur rendue).

**Entraîner avec** (les images rendues vont à l'entraînement seulement ; la
validation reste faite d'images réelles) :

```sh
.venv/bin/python examples/perception/apprentissage/entrainer.py --nom v2 \
    --synthetiques examples/perception/apprentissage/synthetique/entrainement --n-synth 2000 \
    --test examples/perception/apprentissage/synthetique/test_fonds/data.yaml \
    --test examples/perception/apprentissage/synthetique/test_collage/data.yaml
```

`--export DOSSIER` écrit l'export ailleurs que dans `dataset/yolo` (pour
ne pas écraser celui d'un entraînement en cours). `--test` évalue en plus le
meilleur poids sur un autre jeu ; les mAP vont dans `rapport.json`
(clé `tests`).

**Mesurer la robustesse au fond sans autres vraies images.** Les familles
`bois` et `carrelage` servent de « fonds jamais vus » :

```sh
synthese_images.py --test-fonds --n 300      # pièces rendues sur bois/carrelage
synthese_images.py --test-collage --n 300    # pièces RÉELLES (validation) collées sur bois/carrelage
synthese_images.py --nom cmp_entrainement --exclure bois,carrelage --n 1500
```

puis on compare un entraînement réel seul et un entraînement réel +
`cmp_entrainement` sur ces deux jeux. Le jeu « collage » ne pénalise pas le
réseau entraîné sur le réel seul par l'aspect des pièces rendues ; aucun des
deux ne remplace de **vraies** photos d'autres tables : une cinquantaine de
scènes par nouveau fond (table nue, autre nappe, bois clair) restent le
meilleur test, et le meilleur complément à l'entraînement.

**Résultat de la comparaison** (octobre 2026, `yolo11n-seg`, 30 époques,
même validation réelle de 40 images ; `cmp_synth` = réel + copier-coller +
1000 images rendues **sans** les familles `bois` et `carrelage`) :

| mAP masque (50 / 50-95) | validation réelle | fonds jamais vus, pièces rendues | fonds jamais vus, pièces réelles collées |
|---|---|---|---|
| réel seul (`runs/cmp_reel`) | 0,978 / 0,872 | 0,185 / 0,109 | 0,661 / 0,541 |
| réel + rendues (`runs/cmp_synth`) | 0,984 / 0,860 | **0,790 / 0,604** | **0,923 / 0,776** |
| pour mémoire, `v1` (`yolo11s`, 100 époques, réel seul) | 0,979 / 0,860 | 0,110 / 0,063 | 0,720 / 0,591 |

Sur la nappe connue, rien ne change (l'écart est dans le bruit de 40
images). Sur une table jamais vue, le réseau entraîné sur le réel seul perd
un tiers de ses détections même avec de vraies pièces : il a bien appris le
fond. Les images rendues comblent l'essentiel de l'écart. Entraînement
conseillé (« v2 ») : la commande plus haut, avec `synthetique/entrainement`
(toutes les familles).

## 5. Utilisation : la bibliothèque `perception.py`

```python
import sys; sys.path.insert(0, "examples/perception")
from perception import Perception

p = Perception(types={"epaisse": (17, 19, 68), "fine": (8, 17, 68)})
pieces = p.detecter()                        # capture fraîche (serveur de caméra)
# pieces = p.detecter("examples/perception/captures/env2_reglee.npz")
for piece in pieces:
    print(piece.type, piece.face, piece.confiance)
    print(piece.position_table, piece.lacet)  # mm, degrés
    T = piece.T_table                        # 4x4 : face supérieure, repère table
    T = piece.T_camera                       # 4x4 : la même, repère caméra (profondeur)
    m = piece.masque                         # masque dans l'image couleur
import cv2; cv2.imwrite("vue.png", p.dessiner(pieces))
```

ou en ligne de commande : `.venv/bin/python examples/perception/perception.py [capture.npz]`.

- Sans `modeles/pieces-seg.pt` (ou sans couleur), **repli automatique**
  sur la détection géométrique ; `p.mode` dit lequel a servi.
- Les noms des types doivent être ceux des classes de l'entraînement
  (`dataset/config.json`). Ajouter un type de pièce = l'ajouter à
  `config.json` **à la fin** (les numéros de classe des anciennes
  annotations ne doivent pas changer), collecter, annoter, réentraîner.
- Quand l'étalonnage caméra → robot existera :
  `Perception(T_robot_camera=T)` (repère de la caméra de **profondeur**,
  mètres), et chaque pièce porte `piece.T_robot`.
- Temps mesuré (M4 Max, 848×480) : capture 1,1 s (médiane de 15 images),
  puis ≈ 0,15 s pour la table, 0,02 à 0,09 s pour le réseau, ≈ 0,05 à
  0,08 s par pièce pour l'ajustement. La détection géométrique seule
  (repli) prend 4 à 5 s.

### Comment la géométrie vérifie le réseau

Pour chaque masque : les points de profondeur qui s'y projettent donnent
la hauteur de la pièce (donc les faces d'appui possibles) ; s'ils sont
nombreux mais **au niveau de la table**, le masque est rejeté (motif,
papier, ombre). Puis, pour chaque type et face plausibles, une recherche
locale de (x, y, angle) sur la table maximise l'IoU entre le masque et la
silhouette **projetée** du parallélépipède ; les pixels des autres pièces
sont ignorés, si bien qu'une pièce en partie cachée n'est pas tirée vers sa
partie visible. Une IoU sous 0,45 : rejet. Sur des scènes synthétiques, la
position est retrouvée à moins d’un millimètre et l’angle à 1 ou 2° près, même
sans aucune mesure de profondeur dans le masque (pièce saturée).
