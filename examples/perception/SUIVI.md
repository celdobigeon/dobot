# Suivi des pièces d'une image à l'autre (`suivi.py`)

`Perception.detecter()` analyse **une** image : les poses sont bruitées de
quelques millimètres et de quelques degrés, le type ou la face se trompe
parfois, une pièce cachée par le bras disparaît, un reflet devient une pièce.
Quand le robot travaille, la scène change peu d'une image à l'autre :
`Suivi` rapproche chaque détection de ce qu'on savait déjà.

```python
from perception import Perception, ordre_de_prise
from suivi import Suivi

perception, suivi = Perception(), Suivi()
while True:
    pieces = suivi.mettre_a_jour(perception.detecter())
    for p in pieces:
        print(p.id_piste, p.type, p.face, p.position_table, p.lacet, p.confiance, p.vue)
    for e in suivi.evenements:
        print(e)                       # apparue, disparue, deplacee, changee, saisie_probable
    cible = ordre_de_prise([p for p in pieces if p.vue])[0]
```

## Ce que rend `mettre_a_jour`

Les mêmes objets que `detecter()` (des copies), avec :

| attribut | sens |
|---|---|
| `id_piste` | identifiant **stable** de la pièce (jamais réutilisé) ; `numero` lui est égal, `support` et `dessus` sont traduits : `ordre_de_prise(pieces)` fonctionne directement |
| `T_table`, `T_camera`, `T_robot` | pose **lissée** (cohérente dans les trois repères) |
| `type`, `face` | **votés** sur les dernières images (avec hystérésis) |
| `confiance` | confiance de la **piste** ; `confiance_detection` : celle de la dernière détection |
| `vue` | `False` : pièce non détectée cette fois, rendue de mémoire (`absences` : depuis combien d'images) |
| `T_table_brute` | la pose de la dernière détection, non lissée |
| `probabilites_type`, `probabilites_face` | les votes normalisés |

Une piste nouvelle n'est rendue qu'après **2 détections** (un faux positif
isolé n'apparaît jamais) ; exception : la toute première image, qui n'a rien à
quoi se comparer, est rendue telle quelle.

## Événements (`suivi.evenements`, ceux de la dernière mise à jour)

- `apparue` : nouvelle pièce confirmée (ou revenue à sa place après une longue absence : même identifiant) ;
- `disparue` : non vue depuis plus de `absences_max` images (4 par défaut) ;
- `deplacee` : saut de position ou d'angle au-delà du bruit, vu sur deux images ; la pose repart de la nouvelle mesure, sans lissage. Une pièce emportée loin (jusqu'à 30 cm) puis reposée garde son identifiant ;
- `changee` : le type ou la face voté a changé (pièce retournée ?) ;
- `saisie_probable` : une pièce **libre** (rien dessus) a disparu.

Si le programme sait qu'il prend une pièce, il le dit :
`suivi.signaler_saisie(piece.id_piste)` ; elle est alors supprimée dès qu'elle
n'est plus vue (sans attendre `absences_max`).

## Réglages

`Suivi(portee=40, absences_max=6, ...)` ; tous les réglages sont dans la
classe `Reglages` (commentés). Les plus utiles :

- `portee` (30 mm) : distance maximale entre une détection et une piste ;
- `absences_max` (4) : nombre d'images pendant lesquelles une pièce cachée est gardée ; `duree_absence_max` (s) pour une limite en temps ;
- `confirmations` (2) : détections pour confirmer une pièce ;
- `confirmer_sauts` (vrai) : un déplacement doit être vu deux fois ; `False` pour le prendre tout de suite (plus réactif, moins robuste) ;
- `sigma_position`, `sigma_lacet` : bruit attendu des détections ; avec `bruit_adaptatif` (vrai), le suivi corrige lui-même ces valeurs d'après les écarts observés (`suivi.echelles`).

Aucune cadence n'est supposée : `detecter()` peut prendre plusieurs secondes.

## Démonstration

```bash
.venv/bin/python examples/perception/suivi.py --direct              # captures par le serveur, Ctrl-C pour arrêter
.venv/bin/python examples/perception/suivi.py --direct --nombre 5 --sortie suivi.png
.venv/bin/python examples/perception/suivi.py a.npz b.npz c.npz     # captures enregistrées, dans l'ordre
```

## Principe et limites

Filtre de Kalman (modèle constant) sur x, y, z et le lacet (périodique de
180°), association par l'algorithme hongrois (coût : distance, écart d'angle,
incompatibilité de type), cycle de vie provisoire / confirmée / absente /
supprimée. Le détail est dans l'en-tête de `suivi.py`.

Limites :

- une pièce poussée **continûment** (plusieurs images de suite) est suivie avec un retard de quelques millimètres : le modèle suppose des pièces immobiles qui bougent par sauts ; `T_table_brute` donne la mesure brute ;
- la face supérieure d'une pièce **debout** est presque carrée : son lacet est mal défini et n'est presque pas lissé ;
- deux pièces du même type, l'une emportée et l'autre apparue ailleurs au même moment, peuvent échanger leur identité (rien ne les distingue) ;
- une pièce rendue de mémoire (`vue` faux) est à sa **dernière** place connue : ne pas la saisir sans la revoir.
