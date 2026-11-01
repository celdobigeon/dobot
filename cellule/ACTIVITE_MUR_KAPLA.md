# Bâtir un mur vrillé — activité de programmation

IUT, université de Bordeaux · Dobot Magician et sa ventouse
À lire avec le [manuel](README.md) et la [procédure de mise en route](PROCEDURE_CONVOYEUR.md)

---

## La structure à réaliser

![Mur de kapla vrillé : vue de dessus et vue isométrique](assets/mur_kapla.svg)

Des planches de **125 × 30 × 10 mm**, posées **deux par assise**, et
chaque assise **tournée de 75°** par rapport à la précédente. Six
assises, douze planches.

Vue de dessus, c'est une rosace. De côté, c'est un mur qui vrille.

> **Le robot prend toutes les planches au même endroit, dans la même
> orientation.** C'est donc lui qui les tourne, une par une, et c'est
> tout l'objet de l'activité : une cible n'est pas un point, c'est une
> **position *et* une orientation**.

---

## Ce qui est supposé acquis

Cette activité **continue** [Prendre et
poser](ACTIVITE_PICK_AND_PLACE.md). Tout ce qui suit y a été établi, et
n'est ici que rappelé :

| Acquis | Ce qu'on en reprend tel quel |
|---|---|
| mise en route | `python dobot.py mise-en-route`, puis la ligne `defauts effaces` |
| matrice de l'outil | `OUTIL`, mesurée puis vérifiée par une descente |
| repères de travail | `MAGASIN` et `CHANTIER`, par rapport à la base |
| la prise | `PRISE = MAGASIN @ DANS_LE_MAGASIN` |
| les dégagements | `PRISE @ HAUT`, avec `HAUT = translation(0, 0, −80)` |
| le lissage | lissé sur les dégagements, jamais sur un point de contact |

```python
L_OUTIL = ...                                     # votre mesure
OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)
robot.definir_repere_outil(OUTIL)

MAGASIN  = np.array(...)     # mesures au reglet, depuis la base
CHANTIER = np.array(...)     # le CENTRE du mur a batir

DANS_LE_MAGASIN = np.array([[1,  0,  0,         0],
                            [0, -1,  0,         0],
                            [0,  0, -1, EPAISSEUR],
                            [0,  0,  0,         1]], dtype=float)
PRISE = MAGASIN @ DANS_LE_MAGASIN
HAUT  = hg.translation(0, 0, -80)
```

Ce qui est **nouveau**, et qui fait toute l'activité : la dépose change
d'une planche à l'autre, en **hauteur** et en **orientation**.

---

## 1. La dépose : position **et** orientation

### 1.1 La hauteur

L'assise *n* repose sur les *n* précédentes :

```
z = (n + 1) × EPAISSEUR
```

C'est le **décalage d'empilement**, et il doit être un paramètre : des
planches de 8 mm ne doivent demander qu'une ligne à changer.

### 1.2 L'orientation

![D'où viennent le cosinus et le sinus](assets/cible_angle.svg)

L'assise *n* est tournée de `n × 75°`. Pour un angle `a` :

```python
c, s = np.cos(np.radians(a)), np.sin(np.radians(a))

dans_le_chantier = np.array([[c,  s,  0, ex],
                             [s, -c,  0, ey],
                             [0,  0, -1,  z],
                             [0,  0,  0,  1]], dtype=float)
```

Colonne 0 : la **longueur** de la planche. Colonne 1 : sa largeur,
**retournée** par le demi-tour autour de x. Colonne 2 : l'approche,
vers le bas. C'est `Rz(a) · Rx(180°)`, comme dans l'activité
précédente — seul `a` change désormais à chaque assise.

> **Question 1.** Vérifiez une fois pour toutes que c'est une rotation :
> colonnes unitaires, orthogonales deux à deux, déterminant **+1**.
> Le déterminant vaut-il encore `+1` quand `a` change ? Pourquoi
> était-ce prévisible ?

---

## 2. L'écart des deux planches d'une assise

Les deux planches d'une assise sont écartées de `±DECALAGE`,
**perpendiculairement** à leur longueur — donc le long de
`(−sin a, cos a)`.

**Et ce décalage n'est pas libre : il se déduit de l'angle.**

Une planche de l'assise du dessus croise celles du dessous à une
distance du centre qui vaut `DECALAGE / tan(a/2)`. Pour qu'elle repose
franchement, ce croisement doit rester à au moins une demi-largeur du
bout :

```
DECALAGE / tan(a/2)  ≤  LONGUEUR/2 − LARGEUR/2
```

d'où, en prenant l'égalité — le profil le plus large qui tienne :

```python
DECALAGE = (LONGUEUR - LARGEUR) / 2 * np.tan(np.radians(ANGLE) / 2)
```

> **Question 2.** Que vaut `DECALAGE` pour un angle de **90°** ?
> Retrouvez-vous une figure connue ? Expliquez pourquoi le carré
> classique du kapla est le cas particulier de cette formule.

> **Question 3.** Les deux planches d'une même assise ne doivent pas se
> chevaucher : il leur faut au moins une demi-largeur chacune. Écrivez
> cette condition, et **déduisez-en l'angle minimal**. Un quart de tour
> de 30° est-il possible avec ces planches ?

> **Question 4.** Pourquoi l'écart se prend-il le long de
> `(−sin a, cos a)` ? Que donnerait `(cos a, sin a)`, et à quoi
> ressemblerait le mur ?

---

## 3. La boucle

```python
robot.repli()
for assise in range(ASSISES):
    for planche in (0, 1):
        cible = depose(assise, planche)
        # prise  : approche, descente, aspiration, retrait
        # depose : le meme geste, a la hauteur et dans l'orientation voulues
robot.repli()
```

Une seule fonction `depose(assise, planche)` doit produire les douze
cibles. Si vous avez écrit douze matrices, recommencez.

> **Question 5.** Votre `depose` prend deux entiers et rend une matrice.
> Quels sont **tous** les nombres mesurés dont elle dépend ? Faites-en
> la liste, et dites pour chacun ce qu'il faudrait reprendre s'il
> changeait.

---

## 4. Voir les planches bouger

```python
robot.poser_piece(x=..., y=..., dims=(125, 30, 10), magasin=True)
```

La fenêtre dessine la planche et la fait **suivre la ventouse** : on la
voit quitter le magasin, traverser, et se poser sur le mur. Une prise
qui vise à côté se voit alors immédiatement.

> Rembobinez la lecture : les planches reviennent exactement où elles
> étaient. Rien n'est accumulé dans ce simulateur — tout est **fonction
> du temps**, y compris ce que le robot porte.

> Douze cycles couvrent la vue d'arcs bleus et de points jaunes :
> **Affichage ▸ Vue épurée** (Ctrl+Shift+V) ne laisse que la cellule et
> le mur.

---

## 5. Valider hors ligne, puis descendre sur la machine

```bash
python votre_programme.py          # sans --reel : le simulateur
```

```python
print(robot.resume())
for a in robot.controler():
    print(" -", a)
```

### Le poignet, et sa course

C'est **ici** que le mur vrillé se distingue d'un pick and place. Vous
verrez passer des réserves de ce genre :

```
[47 MOVJ] cible : J4 outil = -103.04 deg hors de [-90 ; 90]
```

Le poignet ne tourne que dans `[−90° ; +90°]`. Or **une planche est
symétrique** : la poser à 165° ou à −15° revient au même, elle occupe la
même place. On peut donc ajouter ou retirer des demi-tours librement.

Mais attention — et c'est le piège :

```
J4 = orientation − J1
```

Les orientations atteignables sont centrées sur **J1**, c'est-à-dire sur
la direction où pointe le bras, **et non sur zéro**. Ramener dans
`]−90 ; +90]` par réflexe enverrait 165° sur −15°, et le poignet à
−105° : hors course, pour exactement la même planche au même endroit.

> **Question 6.** Dans quel intervalle faut-il ramener les
> orientations ? Écrivez la fonction qui le fait.

> **Question 7.** Le mur peut être bâti dans n'importe quelle
> orientation de départ — le tourner en entier ne change pas sa forme.
> Montrez qu'un départ à **15°** laisse 15° de marge au poignet, là où
> un départ à 0° le met en butée. Combien d'assises peut-on empiler
> avant que la course du poignet ne suffise plus, et pourquoi ?

> **Question 8.** Le dégagement vaut 80 mm et non 90. Montez-le à 90,
> relancez `controler()`, et expliquez la réserve qui apparaît sur J2.
> Qu'est-ce que cela dit d'un dégagement « généreux » ?

**Ne descendez sur le robot qu'une fois les réserves levées.** Un cycle
validé hors ligne reste une hypothèse ; un cycle qui proteste hors ligne
est une certitude.

```bash
python votre_programme.py --reel
```

---

## 6. Pour aller plus loin

- **Changez l'angle** : 90°, 60°, 45°. Le décalage suit tout seul si
  vous l'avez écrit en fonction de l'angle. Quelle forme obtenez-vous à
  90° ?
- **Changez l'épaisseur** des planches dans une seule constante, et
  vérifiez que le mur monte toujours juste.
- **Posez le magasin en biais** et décrivez-le par la partie rotation de
  `MAGASIN`. Le reste du programme ne doit pas bouger.

---

## Ce qu'il faut rendre

1. Le programme, commenté — chaque cote mesurée doit dire **d'où elle
   vient**.
2. Les réponses aux huit questions.
3. La sortie de `robot.controler()`, et votre explication de ce qui
   limite la hauteur du mur.
