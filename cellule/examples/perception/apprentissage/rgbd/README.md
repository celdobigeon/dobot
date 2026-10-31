# RGB-D : la hauteur au-dessus de la table comme entrée de YOLO-seg

Le jeu réel n'a qu'une nappe (blanche à fleurs) : un réseau RGB risque
d'apprendre le **fond** autant que les pièces. La hauteur au-dessus de la
table, elle, ne dépend ni de la nappe ni de l'éclairage : 0 sur la table, 8,
17 ou 19 mm sur une pièce couchée. On la donne au réseau comme entrée
supplémentaire, alignée pixel à pixel avec l'image couleur brute.

| Fichier | Rôle |
|---|---|
| `rgbh.py` | carte de hauteur dans l'image couleur, trous, codage, image RGB-H ; simulation d'un **fond changé** |
| `exporter_rgbh.py` | jeux YOLO-seg `rgb`, `rgbh`, `rgh` (+ `*_fonds`), validation réelle et validation à fond changé |
| `entrainer_rgbh.py` | YOLO11-seg sur 4 canaux (ou RG-H), témoins RGB dans les mêmes conditions |
| `evaluer_rgbh.py` | mAP des masques de tous les modèles, validation réelle et fond changé |

## Utilisation

```bash
.venv/bin/python examples/perception/apprentissage/rgbd/exporter_rgbh.py --fonds 2
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh   # 4 canaux
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgb    # témoin
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgh    # H à la place du bleu
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh_fonds
.venv/bin/python examples/perception/apprentissage/rgbd/evaluer_rgbh.py               # tableau comparatif
# essai sur CPU (une époque, 10 % des images)
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgbh --essai
# installer le modèle retenu (ici RG-H) dans modeles/pieces-seg-rgbd.pt
.venv/bin/python examples/perception/apprentissage/rgbd/entrainer_rgbh.py --jeu rgh --installer
```

En inférence : `rgbh.image_rgbh(capture, variante)` donne l'image d'entrée
du modèle (`"rgh"` pour le modèle installé, `"rgbh"` pour un modèle à 4
canaux).

## Construction de l'image RGB-H

1. `perception.analyser_table` : plan RANSAC + correction de courbure, hauteur
   de chaque pixel de **profondeur**.
2. Chaque point 3D passe dans le repère couleur (`T_couleur_profondeur`) et se
   projette par `K_couleur` (`perception.projeter_profondeur`).
3. *Splatting* avec tampon de profondeur : chaque point marque 2 × 2 pixels
   couleur (un pixel de profondeur en couvre ~1,3), le plus proche gagne
   (parallaxe aux bords des pièces).
4. Trous (bois clair saturé, ombres de parallaxe, bord du champ) : inpainting
   Navier-Stokes à demi-résolution. Mesuré : 82 à 87 % des pixels couleur
   ont une mesure, mais **11 à 37 % des pixels des pièces n'en ont pas** (bois
   clair) ; ils reçoivent une hauteur interpolée depuis leurs bords.
5. Codage `uint8 = clip(3 × h_mm, 0, 255)` (1/3 mm, 0 à 85 mm).

Le test `tests/test_rgbd.py` vérifie l'alignement : sur une scène rendue
(caméra couleur décalée de 15 mm, tournée de 1°, autres intrinsèques), le
sommet d'une pièce, point 3D connu, tombe au bon pixel couleur et y porte la
bonne hauteur (à 1,5 mm près).

## Quatre canaux dans ultralytics 8.4.187 (vérifié dans le code installé)

- `channels: 4` dans `data.yaml` : images lues avec `IMREAD_UNCHANGED` (PNG
  BGRA), modèle construit avec `ch=4`. Mosaïque, perspective, retournement,
  letterbox et copy-paste acceptent N canaux.
- Poids de départ : tout est repris de `yolo11s-seg.pt` **sauf la première
  convolution** (forme différente) ; un rappel `on_pretrain_routine_end` la
  reconstruit (poids RGB pré-entraînés, 4e canal = leur moyenne), pour le
  modèle et l'EMA.
- `RandomHSV` ne traite que 3 canaux (sinon ne fait rien) : remplacé, dans le
  processus et ses chargeurs (installé à l'import, variable `RGBD_VARIANTE`),
  par une version qui n'agit que sur la couleur. Variante `rgh` : gain de
  luminosité et de couleur sur G et R seulement. La hauteur n'est jamais
  modifiée (testé).
- Bourrage (letterbox, mosaïque) : 114 sur tous les canaux, soit 38 mm de
  « hauteur » dans les bandes de bourrage ; identique à la validation.

## Fond changé (simulation)

Faute d'images réelles sur d'autres fonds : les pixels de **table** (hauteur
mesurée < 4 mm, ouverture morphologique contre les bandes du décor qui
traversent la hauteur 0, plus les pixels sans mesure dans l'enveloppe de la
table, hors des masques annotés dilatés de 3 px) sont remplacés par une table
texturée rendue **en perspective avec la vraie caméra** (plan de la table de
l'annotation), modulée par l'éclairage lent de la scène. La hauteur reste
intacte. Textures : familles procédurales de `synthese_images.py` (lu, non
modifié) ; pour le test, **bois** et **carrelage** (celles que l'étude des
images rendues garde pour son test « fonds jamais vus ») ; pour
l'augmentation d'entraînement (`--fonds`), uni, bruit, tissu, motifs, papier
seulement. 2 versions par image de validation (80 images, 350 pièces).

Limites de la simulation : pas d'ombre portée des pièces sur le nouveau fond,
liseré de 3 px de la nappe d'origine autour des pièces.

## Résultats

mAP des **masques** (ultralytics, IoU 0,5:0,95 et 0,5), YOLO11s-seg,
imgsz 864. Validation réelle : les 40 images de `runs/v1` (175 pièces) ;
fond changé : les mêmes, 2 versions (bois, carrelage), 80 images.
Entraînement sur les 153 images réelles d'entraînement seulement (sans
copier-coller) ; `*_fonds` : + 2 copies par image à table repeinte (autres
familles), 60 époques.

| modèle | entrée | val mAP50-95 | val mAP50 | fond changé mAP50-95 | fond changé mAP50 |
|---|---|---|---|---|---|
| v1 (référence : RGB + copier-coller + fonds réels, 100 ép.) | RGB | 0,860 | 0,979 | 0,743 | 0,868 |
| `rgbd_rgb` (témoin) | RGB | **0,863** | 0,979 | 0,663 | 0,844 |
| `rgbd_rgbh` | B G R H | 0,861 | 0,977 | 0,754 | 0,916 |
| `rgbd_rgh` | H G R | **0,864** | **0,981** | **0,816** | **0,951** |
| `rgbd_rgb_fonds` | RGB + fonds repeints | 0,855 | 0,976 | 0,871* | 0,975* |
| `rgbd_rgbh_fonds` | B G R H + fonds repeints | 0,842 | 0,975 | 0,867* | 0,972* |

\* même méthode de fabrication que l'augmentation d'entraînement : optimiste.

Par classe (mAP50-95 sur fond changé) : la pièce **fine** souffre le plus du
changement de fond (témoin RGB 0,614 ; RGB-H 0,692 ; RG-H 0,777 ;
`*_fonds` 0,87). Temps d'inférence identiques (~68 ms/image sur CPU,
imgsz 864) ; préparation de l'image RGB-H : ~0,2 s par capture (table +
projection, déjà calculées par `perception.py`) + 10 ms (bouchage).
Entraînements : 27 à 30 min (100 époques, 153 images) et 43 min
(`*_fonds`, 60 époques, 459 images) sur MPS.

À retenir :

- Sur le fond **réel**, la hauteur n'apporte rien de mesurable (0,861 contre
  0,863) : la nappe est apprise, la couleur suffit.
- Sur un fond **changé**, à conditions égales, la hauteur fait gagner 9 points
  de mAP50-95 (0,754 contre 0,663) et 7 points de mAP50 (0,916 contre 0,844).
  v1 (RGB, copier-coller sur les fonds réels) récupère une partie de l'écart
  (0,743) mais garde un mAP50 plus bas (0,868) : il rate plus de pièces.
- **RG-H** (hauteur à la place du bleu, 3 canaux) fait nettement mieux que
  4 canaux (0,816 contre 0,754) : la première convolution pré-entraînée est
  utilisée telle quelle (pas de 4e canal initialisé à la moyenne), et le
  réseau perd le canal bleu, ce qui l'oblige à s'appuyer sur la hauteur.
- L'**augmentation de fond** (table repeinte grâce à la hauteur) est le levier
  le plus fort, et elle profite autant au RGB seul (0,871) qu'au RGB-H
  (0,867) : une fois les fonds variés à l'entraînement, la hauteur en entrée
  n'apporte plus rien sur ce test. Mais la validation à fond changé est
  fabriquée par la MÊME méthode (familles de textures différentes, même
  rendu, même absence d'ombres) : ce chiffre est optimiste pour `*_fonds`.
  Seules de vraies images sur un autre fond trancheront.

**Modèle installé** : `modeles/pieces-seg-rgbd.pt` = `runs/rgbd_rgh`
(meilleur sur la validation réelle, et meilleur sur fond changé parmi les
modèles dont le test n'est pas fabriqué comme l'entraînement). Son entrée
est **H, G, R** : `img, _ = rgbh.image_rgbh(capture, "rgh")`, puis
`YOLO("modeles/pieces-seg-rgbd.pt").predict(img, imgsz=864, retina_masks=True)`.
`modeles/pieces-seg.pt` n'est pas touché.

**Recommandation.** (1) Ajouter l'augmentation de fond par la hauteur
(`--fonds`) à l'entraînement de production, en RGB ou en RG-H ; c'est le
gain le plus net. (2) Préférer RG-H à 4 canaux. (3) Valider sur quelques
vraies captures sur une autre nappe (ou la table nue) avant de trancher
entre RGB+fonds et RG-H+fonds ; RG-H a l'avantage de ne pas dépendre du
fond par construction, mais la hauteur manque sur 11 à 37 % des pixels des
pièces (bois clair saturé).
