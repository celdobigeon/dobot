# Dobot Magician — deux dossiers, deux usages

Ce dépôt est rangé en deux, depuis le 31/10/2026.

## `cellule/` — ce qu'on donne aux étudiants

Tout ce qu'il faut pour **piloter le robot et le convoyeur**, et rien
d'autre. Le dossier se donne tel quel : copié, zippé, cloné.

```
cellule/
    dobot_sim/            le paquet : modèle, matrices, liaison série
    main.py               le simulateur
    dobot.py              les commandes d'atelier (mise en route, état…)
    examples/             les programmes du cours
    Modele3D/  assets/    les maillages
    fiches/  index.html   les fiches de référence
    README.md             la documentation
    PROCEDURE_*.md        les procédures de mise en route
    ACTIVITE_*.md         les activités de TP (prendre et poser,
                          convoyeur, structure de kapla)
```

Les commandes des procédures se lisent **depuis ce dossier** :

```bash
cd cellule
python main.py
python examples/convoyeur.py --reel
```

## `atelier/` — la mise au point

Les diagnostics, les essais, les outils qui réparent. Chacun est né
d'une panne réelle et ne répond qu'à une question. Rien là-dedans n'est
nécessaire pour faire tourner la cellule ; tout y est nécessaire le jour
où elle ne tourne plus.

```bash
cd atelier
python diag_capteurs_angle.py COM9 --origine
```

Ces scripts trouvent le paquet grâce à `cellule_sur_le_chemin.py`, qui
ajoute `cellule/` au chemin de recherche de Python. Un nouveau script
d'atelier doit donc commencer par :

```python
import cellule_sur_le_chemin        # noqa: F401
from dobot_sim import Pilote
```

## Pourquoi cette séparation

Parce que les deux publics ne cherchent pas la même chose. Un étudiant
qui ouvre le dossier doit voir un robot, un convoyeur et des exemples —
pas dix-huit diagnostics dont chacun raconte une panne qu'il n'a pas.
Et celui qui dépanne veut l'inverse.

> La frontière n'est pas « ce qui marche » contre « ce qui est brouillon ».
> Les outils de l'atelier sont écrits avec le même soin ; ils répondent
> seulement à des questions qu'on ne se pose qu'en cas de panne.
