# Empiler des pièces — activité de programmation

IUT, université de Bordeaux · Dobot Magician et sa ventouse
À lire avec le [manuel](README.md) et la [procédure de mise en route](PROCEDURE_CONVOYEUR.md)

---

## Ce que vous allez faire

Construire une tour de pièces, à la manière d'un jeu de kapla : prendre
les pièces toujours au même endroit, et les poser les unes sur les
autres.

Le geste est simple. Ce que l'activité vous demande d'apprendre, c'est
de l'écrire **sans jamais déplacer le bras à la main pour lui montrer
où aller**. Toutes les cibles seront des **matrices homogènes 4×4**,
construites à partir de cotes mesurées au réglet.

> **Pourquoi ne pas apprendre les points à la main ?** Parce qu'un point
> appris ne dit pas *pourquoi* il est là. Déplacez le magasin de deux
> centimètres et il faut tout recommencer. Une cible écrite comme
> « le magasin, puis la pièce dans le magasin » demande, elle, de
> changer **un seul nombre**.

À la fin, vous devez pouvoir déplacer le magasin sur la table, changer
une ligne, et voir le cycle continuer de fonctionner.

---

## 1. Mettre le robot en route

```bash
cd cellule
python dobot.py mise-en-route
```

Attendez la ligne **`mise en route terminee : defauts effaces`**. C'est
elle, et non « prise d'origine terminée », qui dit que la machine est
prête.

**Avant de lancer**, dégagez la zone et amenez le bras à une posture
franche et haute, la base au milieu de sa course : le bras parcourt
toute sa course pendant l'origine.

> **Question 1.** Pourquoi faut-il refaire cette opération à chaque mise
> sous tension ? Que perd le contrôleur quand on l'éteint ?

---

## 2. Mesurer l'outil, et le déclarer en matrice

### 2.1 Mesurer

```bash
python mesurer_outil.py
```

Le script vous demande d'amener la **pointe de la ventouse au contact du
plan de travail**, à la main, en plusieurs endroits. Il lit les angles,
calcule la hauteur de la bride, et en déduit la longueur :

```
L = z_bride(q) − z_plan
```

**Relevez la valeur sur au moins trois points de la table.**

> **Question 2.** Les trois mesures doivent donner la même longueur. Si
> elles s'écartent de plusieurs millimètres, ce n'est pas l'outil —
> une longueur d'outil est la même partout. Qu'est-ce que cela
> révélerait alors ?

Le bras a été déplacé à la main : **refaites une prise d'origine** avant
la suite.

### 2.2 Déclarer

L'outil est une **transformation** : de l'axe de rotation du poignet
jusqu'à la pointe. On l'écrit en matrice.

```python
L_OUTIL = ...        # votre mesure, en millimetres

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot.definir_repere_outil(OUTIL)
```

Lisez cette matrice colonne par colonne — c'est ainsi qu'une matrice
homogène se lit, et pas autrement :

| Colonne | Ce qu'elle dit |
|---|---|
| 0 : `(1, 0, 0)` | l'axe **x** de l'outil est celui de la bride |
| 1 : `(0, −1, 0)` | l'axe **y** est retourné |
| 2 : `(0, 0, −1)` | l'axe **z** pointe vers le **bas** : la direction d'approche |
| 3 : `(0, 0, −L)` | la **pointe**, L millimètres sous la bride |

> **Question 3.** Pourquoi l'axe z de l'outil pointe-t-il vers le bas,
> alors que celui de la table pointe vers le haut ? Que se passerait-il
> si on l'oubliait au moment d'écrire un dégagement ?

### 2.3 Vérifier

Une mesure non vérifiée n'est pas une mesure. Faites descendre la
ventouse sur un plan de cote connue et comparez.

Un écart **constant**, le même partout, est une erreur de longueur
d'outil : corrigez `L_OUTIL`. Un écart **qui change selon l'endroit**
n'est pas l'outil — c'est l'origine des axes, et il faut reprendre
l'étape 1.

---

## 3. Déclarer un repère de prise

Le magasin est posé sur la table. Mesurez au réglet, **par rapport au
centre de la base du robot**, où se trouve son coin de référence.

```python
MAGASIN = np.array([[1, 0, 0,  ...],
                    [0, 1, 0,  ...],
                    [0, 0, 1,    0],
                    [0, 0, 0,    1]], dtype=float)
```

Ici la partie rotation est l'identité : le magasin est **parallèle** à
la base. La dernière colonne porte sa position.

> Remarquez le contraste avec `OUTIL` : le z d'un repère de table pointe
> vers le **haut**, celui d'un outil vers le **bas**. Ce ne sont pas les
> mêmes objets.

> **Question 4.** Que faudrait-il changer dans cette matrice si le
> magasin était posé en biais, tourné d'un quart de tour autour de la
> verticale ? Combien de nombres faut-il reprendre ?

**Vérification.** `repere_travail(x=..., y=...)` construit la même chose
à partir de coordonnées. Comparez votre matrice avec elle :

```python
assert np.allclose(MAGASIN, repere_travail(x=..., y=...))
```

Si l'assertion passe, votre matrice est juste. **Gardez la vôtre** :
c'est elle que l'activité demande.

---

## 4. Les cibles, et le `@`

### 4.1 La prise, exprimée dans le repère

La pièce est dans le magasin, et la ventouse se pose sur son **dessus**.
Écrivez d'abord sa situation **dans le repère du magasin** :

```python
PIECE = 25.0          # cote du cube, mesuree

DANS_LE_MAGASIN = np.array([[1,  0,  0,     0],
                            [0, -1,  0,     0],
                            [0,  0, -1, PIECE],
                            [0,  0,  0,     1]], dtype=float)
```

Puis composez :

```python
PRISE = MAGASIN @ DANS_LE_MAGASIN
```

**C'est le cœur de l'activité.** La cible n'est pas un triplet de
coordonnées : c'est *le magasin*, suivi de *la pièce dans le magasin*.
Déplacez le magasin, et seule la matrice `MAGASIN` change.

> **Question 5.** Dans `MAGASIN @ DANS_LE_MAGASIN`, l'ordre compte-t-il ?
> Essayez l'autre sens et regardez où part la cible. Expliquez.

### 4.2 Les approches et les retraits

Un dégagement s'écrit lui aussi par composition, mais **dans le repère
de l'outil** :

```python
HAUT = hg.translation(0, 0, -70)     # 70 mm AU-DESSUS

robot.movj(PRISE @ HAUT)             # approche
robot.movl(PRISE)                    # descente
...
robot.movl(PRISE @ HAUT)             # retrait
```

Le `-70` surprend : on monte, et le nombre est négatif. C'est que l'axe
z de l'outil pointe vers le **bas** (question 3). Vérifiez-le plutôt
que de le croire :

```python
assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"
```

> **Question 6.** `PRISE @ HAUT` et `HAUT @ PRISE` ne donnent pas la même
> chose. Laquelle des deux déplace l'outil **dans son propre repère** ?

### 4.3 Le poignet a des butées

Une ventouse se moque de son orientation autour de z — mais le poignet,
lui, ne tourne que dans `[−90° ; +90°]`. Si vous ajoutez un quart de
tour à vos matrices, `robot.controler()` vous dira si J4 sort.

> **Question 7.** Avec une **pince** à la place de la ventouse,
> l'orientation ne serait plus libre : elle serait imposée par la pièce.
> Que faudrait-il déplacer, alors, plutôt que de tordre le poignet ?

---

## 5. La boucle, et le décalage d'empilement

Le chantier est un second repère, déclaré comme le premier. La n-ième
pièce s'y pose **n épaisseurs plus haut**.

```python
def depose(n):
    """Cible de depose de la n-ieme piece, n comptee a partir de zero."""
    dans_le_chantier = np.array([[1,  0,  0,  ...],
                                 [0, -1,  0,  ...],
                                 [0,  0, -1,  ...],      # <- le decalage
                                 [0,  0,  0,    1]], dtype=float)
    return CHANTIER @ dans_le_chantier
```

Puis la boucle :

```python
robot.repli()
for n in range(PILE):
    cible = depose(n)
    robot.movj(PRISE @ HAUT, lissage=70)
    robot.movl(PRISE, vitesse=40)
    robot.ventouse(True)
    robot.attendre(0.3)
    robot.movl(PRISE @ HAUT)
    # ... la depose, sur le meme modele
robot.repli()
```

> **Question 8.** Pourquoi écrire le décalage comme une **translation
> dans le repère du chantier**, plutôt qu'en retouchant la cote z de
> trois matrices recopiées ? Que se passe-t-il, dans chacun des deux
> cas, si l'on change l'épaisseur des pièces ?

---

## 6. Valider hors ligne, avant de descendre sur la machine

```bash
python votre_programme.py          # sans --reel : le simulateur
```

Le modèle vous dit, **machine éteinte**, ce que la machine découvrirait
à vos dépens :

```python
print(robot.resume())
for a in robot.controler():
    print(" -", a)
```

> **Question 9.** Montez la pile : 4 pièces, puis 5, puis 6. À partir de
> combien `controler()` proteste-t-il, et sur quel axe ? Expliquez
> pourquoi c'est **la hauteur** qui finit par coincer, et non la
> distance.

**Ne descendez sur le robot qu'une fois les réserves levées.** Un cycle
validé hors ligne reste une hypothèse ; un cycle qui proteste hors ligne
est une certitude.

```bash
python votre_programme.py --reel
```

---

## 7. Pour aller plus loin

- **Déplacez le magasin** de quelques centimètres, mesurez sa nouvelle
  position, et changez **une seule matrice**. Le cycle doit continuer de
  fonctionner. Si vous avez dû toucher autre chose, vos cibles n'étaient
  pas vraiment exprimées dans le repère.
- **Posez le magasin en biais** et faites-le décrire par la partie
  rotation de `MAGASIN`.
- **Construisez autre chose qu'une tour** : deux colonnes et une pièce
  en travers. Le décalage devient une fonction de `n` à deux dimensions.

---

## Ce qu'il faut rendre

1. Le programme, commenté — chaque cote mesurée doit dire **d'où elle
   vient**.
2. Les réponses aux neuf questions.
3. La sortie de `robot.controler()` pour la pile la plus haute que vous
   ayez réussi à monter, et votre explication de ce qui l'a limitée.
