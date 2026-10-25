# Points clés : les coins de la face supérieure, puis la pose par PnP

Un réseau YOLO11-pose prédit, pour chaque pièce, sa classe (`epaisse`,
`fine`) et les **4 coins de sa face supérieure** dans l'image couleur. Avec
les dimensions connues des pièces, `K_couleur` et le plan de la table, la
pose 3D s'en déduit directement (PnP), sans recherche itérative : c'est une
**source de propositions** supplémentaire pour `raffinement.py`, indépendante
du masque (et donc d'une partie des erreurs de segmentation).

| Fichier | Rôle |
|---|---|
| `coins.py` | géométrie : ordre canonique des coins, PnP (IPPE), pose contrainte au plan de la table, erreurs |
| `etiquettes.py` | étiquettes (réel : dérivées des masques validés et de l'ajustement 3D ; synthétique : coins exacts) et export YOLO-pose |
| `entrainer_cles.py` | entraînement YOLO11-pose avec perte invariante à la symétrie ; poids → `modeles/pieces-cles.pt` |
| `proposer.py` | **l'interface** : `ProposeurCles.proposer(image, vue, table, proj)` → hypothèses de pose |
| `evaluer_cles.py` | erreur des coins, erreur de pose comparée à l'ajustement de référence, temps |

## Utilisation

```bash
# 1. étiquettes + export (réel validé + 500 images rendues), puis entraînement
.venv/bin/python examples/perception/apprentissage/points_cles/entrainer_cles.py \
    --synthetiques synthetique/cmp_entrainement --n-synth 500 --epochs 60 --patience 20 --nom cles_v1
# essai rapide sur CPU (une époque, 5 % des images) : vérifie que tout tourne
.venv/bin/python examples/perception/apprentissage/points_cles/entrainer_cles.py --essai
# 2. évaluation sur la validation réelle (et un jeu rendu à fonds jamais vus)
.venv/bin/python examples/perception/apprentissage/points_cles/evaluer_cles.py \
    --modele runs/cles_v1/weights/best.pt --synthetique synthetique/test_fonds
# sans réseau : contrôle de la chaîne PnP sur les coins de référence
.venv/bin/python examples/perception/apprentissage/points_cles/evaluer_cles.py --sans-reseau
```

```python
import sys; sys.path.insert(0, "examples/perception/apprentissage/points_cles")
from proposer import ProposeurCles
prop = ProposeurCles()                                   # modeles/pieces-cles.pt
hyps = prop.proposer_capture("…/dataset/brut/0042.npz")  # ou proposer(image, vue, table, proj)
for h in hyps:
    print(h.type, h.face, h.T_table[:3, 3] * 1000, h.confiance, h.erreur_px, h.methode)
pieces = [h.piece(k + 1) for k, h in enumerate(hyps)]   # perception.PieceVue
```

## Interface pour `raffinement.py` (non branchée)

`raffinement.raffiner(capture, propositions, types, masques, vue, table=…)`
consomme des `vision.Piece` dont `T_camera` est dans le repère de la caméra
de **profondeur**, avec `type` et `face`. `Hypothese.piece()` rend
exactement cela (`perception.PieceVue`, `origine="points_cles"`,
`hauteur_base`, `inclinaison`). Il suffit d'ajouter ces pièces à la liste :

```python
props = pieces_reseau + [h.piece(k) for k, h in enumerate(
    proposeur.proposer(vue.image, vue=vue, table=table, proj=proj))]
raffinement.raffiner(capture, props, types, masques, vue, table=table)
```

Les doublons (même pièce proposée par YOLO-seg et par les coins) sont déjà
éliminés par l'affinage (pénétration mutuelle). Chaque hypothèse porte aussi
ses **variantes** `(type, face, base, score, erreur_px)`, triées : utiles pour
essayer plusieurs interprétations discrètes. Les hypothèses dont l'erreur de
reprojection dépasse `erreur_max` (12 px) sont écartées.

## Choix techniques

**Étiquettes réelles.** Il n'existe pas de coins annotés : pour chaque masque
validé, `perception.ajuster_masque` (lu, non modifié) cherche la pièce de
taille connue posée sur la table dont la silhouette explique le masque ; les
coins de sa face supérieure, projetés, sont l'étiquette. Écartées (725
gardées sur 813) : IoU masque/silhouette < 0,85 (76, surtout des pièces
inclinées), type ≠ classe annotée (7), instances douteuses (3), sans
ajustement (2). Une instance écartée reste dans l'image avec sa boîte et sa
classe mais **sans coins** (visibilité 0, aucune perte de coins) : sinon le
réseau apprendrait qu'une pièce non étiquetée est du fond. Visibilité 1 pour
un coin caché par une autre pièce, 0 hors de l'image.

**Ordre canonique et symétrie.** Une face rectangulaire est identique après
une rotation de 180° (et la pièce est symétrique). L'ordre est défini dans
l'image : grand côté orienté vers la droite ; 0 = bout gauche « haut »,
1 = bout droit « haut », 2 = bout droit « bas », 3 = bout gauche « bas ».
Il est compatible avec le retournement horizontal (`flip_idx [1, 0, 3, 2]`),
mais **saute de 180°** quand le grand côté passe par la verticale de l'image
(aucune règle continue n'existe : l'orientation modulo 180° est un cercle).
D'où, pendant l'entraînement, une **perte invariante** : la perte des coins
d'ultralytics est remplacée (dans le processus seulement, les poids restent
des `PoseModel` ordinaires) par son minimum sur {identité, rotation de 180°},
choisi instance par instance ; l'OKS de validation prend de même le maximum.
Pour la pose, la permutation de 180° est sans effet.

**Pose.** Pour chaque (type, face, base) plausible (comme
`ajuster_masque` : hauteur mesurée par la profondeur dans le quadrilatère,
à 8 mm près sur la table, 5 mm empilée), la face supérieure est posée
**horizontale** à sa hauteur connue et ajustée aux 4 coins (rayons coupés par
ce plan, Procrustes 2D : 3 degrés de liberté). C'est beaucoup plus robuste
qu'un PnP libre : à 0,5 m, 1 px d'erreur sur un coin vaut ~5 mm de profondeur
par PnP libre, alors que la hauteur vient ici de la table. Score : erreur de
reprojection (σ = 8 px, adapté aux ~4 px d'erreur des coins prédits) ×
accord avec la hauteur mesurée (5 mm) × 0,8 si le type diffère de la classe
du réseau. Une pièce **inclinée** n'est retenue (pose libre IPPE) que si la
**profondeur confirme la pente** : la différence de hauteur mesurée entre
les deux bouts de la face doit valoir celle de la pose libre à 5 mm près (et
dépasser 8 mm), en plus d'une reprojection nettement meilleure. Les coins
seuls ne suffisent pas : un PnP libre (6 degrés de liberté pour 8 mesures,
vue presque affine) « explique » le bruit des coins prédits et inventait des
inclinaisons (sur la validation, p90 de l'erreur de position 30 mm avant ce
contrôle, 10,5 mm après).

**Poids de départ.** Pas de `yolo11s-pose.pt` local : le modèle est construit
depuis `yolo11s-pose.yaml` et reçoit tronc et cou de `yolo11s-seg.pt`.

## Résultats

Validation réelle : les 40 images de `runs/v1` (découpage crc32 d'`exporter.py`),
164 pièces avec coins de référence.

**Chaîne géométrique seule** (coins de référence, avec bruit gaussien ;
`evaluer_cles.py --sans-reseau`) :

| bruit des coins | position (médiane / p90) | lacet (médiane / p90) | bon (type, face) |
|---|---|---|---|
| 0 px | 0,002 / 6,0 mm | 0,002 / 0,005° | 89 % |
| 1 px | 0,6 / 5,4 mm | 0,5 / 1,4° | 90 % |
| 2 px | 1,2 / 6,1 mm | 1,1 / 3,3° | 84 % |
| 3 px | 1,8 / 6,5 mm | 1,5 / 3,8° | 85 % |

Sans bruit, la pose est exacte quand le bon candidat est choisi ; les ~11 %
de (type, face) « faux » (p90 de 6 mm) sont surtout l'ambiguïté `epaisse` à
plat (68 × 19, haute de 17) / sur la tranche (68 × 17, haute de 19) : 2 mm
d'écart, sous le bruit de la profondeur. (Avec σ = 3 px, 95 % sans bruit
mais 80 % et un p90 de 28 mm à 3 px de bruit : σ = 8 px est le bon compromis
pour des coins prédits.)

**Réseau** `runs/cles_v1` (YOLO11s-pose, 153 images réelles + 500 rendues de
`synthetique/cmp_entrainement`, 60 époques, 40 min sur MPS) :

| mesure | valeur |
|---|---|
| détection (IoU boîtes ≥ 0,5) | rappel 98,3 % (172/175), précision 86,9 % |
| mAP50-95 boîtes / coins (OKS symétrique, ultralytics) | 0,855 / 0,876 |
| erreur des coins (161 pièces) | médiane 3,9 px, moyenne 4,7, p90 7,2 |
| position après PnP vs ajustement de référence | médiane 2,2 mm, moyenne 4,3, p90 10,5 |
| lacet | médiane 2,0°, moyenne 2,6, p90 5,4 |
| bon (type, face) | 77,6 % (classe du réseau juste : 94 %) |
| temps (CPU) | réseau 88 ms/image, table + projection 137 ms/image, PnP 2 à 4 ms/pièce |

Médiane de position par (type, face) de référence : épaisse à plat 3,3 mm,
épaisse sur la tranche 2,0, fine à plat 1,9, fine sur la tranche 2,4.

**Fonds jamais vus** (`synthetique/test_fonds` : bois, carrelage, absents de
`cmp_entrainement` ; caméra et table exactes du méta, sans profondeur) :
rappel 74 % (IoU ≥ 0,3), coins médiane 7,1 px (p90 32), position 11,4 mm
(p90 62), lacet 3,2°. Le réseau de coins généralise donc mal à un fond
inconnu ; ce test cumule fond inconnu, poses de caméra variées et pièces
rendues.

## Limites

- Les étiquettes réelles héritent des erreurs de l'ajustement par masque
  (pièces supposées horizontales ; les pièces inclinées sont écartées du
  réel, présentes seulement dans le synthétique).
- Les 4 coins de la face supérieure d'une pièce vue très en biais forment un
  quadrilatère très aplati : l'erreur de coins pèse surtout sur la largeur,
  donc sur la distinction des faces de 17 et 19 mm.
- Pièce debout : face supérieure presque carrée (19 × 17), l'ordre canonique
  y est moins stable (le grand côté est mal défini dans l'image).
