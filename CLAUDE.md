# Conventions du projet APPS_DOBOT

Ce que tout ce qui touche au Dobot doit respecter — énoncés, corrigés,
exemples, fiches.

## Les matrices s'écrivent en 4×4, toujours

**Une cible, un repère, une transformation se déclarent sous la forme
d'une matrice homogène 4×4 écrite en clair**, dernière ligne comprise :

```python
DEPOSE = np.array([[1,  0,  0,       140.0],
                   [0, -1,  0,       200.0],
                   [0,  0, -1,   EPAISSEUR],
                   [0,  0,  0,         1.0]], dtype=float)
```

Et **non** `hg.translation(...) @ hg.rot_x(...)` comme déclaration.

**Pourquoi.** C'est la forme que l'étudiant doit savoir lire et écrire :
une colonne par axe, la position en quatrième colonne, et le
`0 0 0 1` qu'on oublie. Un produit de fonctions cache les seize
nombres au moment précis où il faut les voir.

**Le produit reste**, mais comme justification — il dit *d'où viennent*
les nombres, et il se vérifie :

```python
# un demi-tour autour de x, a cet endroit
assert np.allclose(DEPOSE,
                   hg.translation(140.0, 200.0, EPAISSEUR) @ hg.rot_x(np.pi))
```

L'ordre compte : une matrice écrite **à droite** agit dans le repère
que les précédentes viennent de construire.

**Ce qui échappe à la règle**, et pour une raison :

- la **composition de repères** — `PRISE = MAGASIN @ DANS_LE_MAGASIN`
  — qui est l'objet même de l'enseignement : chaque facteur y est déjà
  une 4×4 écrite en clair ;
- `cellule/examples/exemple_matrices.py`, qui démontre précisément les
  fonctions de `homogeneous` ;
- les brouillons de l'atelier, qui sont ceux de Céline.

## Le vocabulaire

| On écrit | Pas |
|---|---|
| pièce | planche |
| structure (de kapla) | mur |
| transférer sur la machine | descendre sur la machine |

## Les figures des énoncés

Elles sont **calculées**, jamais dessinées à la main : un générateur
`atelier/dessiner_*.py` les écrit dans `cellule/assets/`. Une figure
faite à part dérive du code dès la première modification, et une
figure fausse dans un énoncé coûte plus cher qu'une figure absente.

Lignes et colonnes des matrices y portent le nom des axes — `Xo Yo Zo`
en colonnes, `Xb Yb Zb` en lignes — et le troisième axe se note point
dans le cercle quand il sort de la feuille, croix quand il rentre.

## Les corrigés ne descendent pas dans `cellule/`

`atelier/corrige_*.py` reste dans l'atelier : l'activité consiste à
écrire ce fichier, le donner c'est la supprimer.
