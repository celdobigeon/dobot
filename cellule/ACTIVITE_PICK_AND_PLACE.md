# Prendre et poser — activité de programmation

IUT, université de Bordeaux · Dobot Magician et sa ventouse
À lire avec le [manuel](README.md) et la [procédure de mise en route](PROCEDURE_CONVOYEUR.md)

---

## Le geste à réaliser

**Une pièce**, prise au **magasin** et posée sur la **structure**.
Rien de plus.

C'est le geste le plus courant de la robotique industrielle, et il
tient en six instructions. Toute la difficulté est ailleurs : **dire au
robot où aller, sans jamais l'y amener à la main.**

La pièce est posée dans une **autre orientation** que celle où elle a
été prise — c'est la seule différence entre les deux cibles, et c'est
déjà de quoi faire travailler les matrices.

> **Dans cette activité, toutes les cibles sont écrites par rapport à
> la base du robot.** C'est la façon la plus directe, et c'est celle
> qu'on pratique en premier. L'activité suivante — [bâtir une
> structure vrillée](ACTIVITE_STRUCTURE_KAPLA.md) — montrera ce qu'elle coûte, et
> introduira les **repères de programmation** pour y remédier. La
> dernière partie de ce sujet prépare le terrain : **gardez-la pour la
> fin.**

---

## Ce que vous allez apprendre

1. **Mesurer l'outil** et le déclarer sous forme de matrice.
2. **Écrire une cible** comme une matrice homogène 4×4, par rapport à
   la base du robot — position **et** orientation.
3. **Composer les approches et les retraits** avec `@`, dans le repère
   de l'outil.
4. **Le lissage** et son utilisation.

Toutes les cibles seront des **matrices homogènes 4×4**, construites à
partir de cotes mesurées au réglet. Aucun point ne sera appris en
amenant le bras à la main.

> **Pourquoi ne pas apprendre les points à la main ?** Parce qu'un
> point appris ne dit pas *d'où il vient*. Une cote mesurée au réglet
> se discute, se vérifie, se corrige. Un point appris, non : il faut
> tout recommencer.

---

## 1. Mettre le robot en route

```bash
cd cellule
python dobot.py mise-en-route
```

Attendez la ligne **`mise en route terminee : defauts effaces`**. C'est
elle, et non « prise d'origine terminée », qui dit que la machine est
prête.

**Avant de lancer** : dégagez la zone, amenez le bras à une posture
franche et **haute**, la base au milieu de sa course. Le bras parcourt
toute sa course pendant l'origine ; gardez la main près de
l'interrupteur.

> **Question 1.** Pourquoi faut-il refaire cette opération à chaque mise
> sous tension ? Que perd le contrôleur quand on l'éteint ?

---

## 2. Mesurer l'outil, et le déclarer en matrice

### 2.1 Mesurer

```bash
python mesurer_outil.py
```

Amenez la **pointe de la ventouse au contact du plan de travail**, à la
main, en **plusieurs endroits** de la table. Le script lit les angles,
calcule la hauteur de la bride et en déduit la longueur :

```
L = z_bride(q) − z_plan
```

> **Question 2.** Les mesures doivent donner la même longueur partout —
> une longueur d'outil ne dépend pas de l'endroit. Si elles s'écartent
> de plusieurs millimètres, qu'est-ce que cela révélerait ?

Le bras a été déplacé à la main : **refaites une prise d'origine** avant
la suite.

### 2.2 Les deux repères, côte à côte

![Repère de la base et repère de l'outil](assets/reperes_base_outil.svg)

Regardez la figure avant d'écrire quoi que ce soit : les deux repères
ont le **même `x`**, et leurs `y` et `z` sont **retournés**.

Le `z` de l'outil pointe vers le bas parce que c'est sa **direction
d'approche** : l'outil avance vers la pièce, et la pièce est en
dessous. Un repère de *table*, lui, a son `z` vers le haut.

Passer de l'un à l'autre, c'est donc un **demi-tour autour de `x`** —
noté `Rx(180°)` — puis une translation de `L` jusqu'à la pointe.

### 2.3 Déclarer

```python
L_OUTIL = ...        # votre mesure, en millimetres

OUTIL = np.array([[1,  0,  0,        0],
                  [0, -1,  0,        0],
                  [0,  0, -1, -L_OUTIL],
                  [0,  0,  0,        1]], dtype=float)

robot.definir_repere_outil(OUTIL)
```

Lisez-la **colonne par colonne** — c'est ainsi qu'une matrice homogène
se lit, et c'est exactement ce que dit la figure :

| Colonne | Ce qu'elle dit |
|---|---|
| <code>0&nbsp;:&nbsp;(1,&nbsp;0,&nbsp;0)</code> | l'axe **x** de l'outil, celui de la bride : inchangé |
| <code>1&nbsp;:&nbsp;(0,&nbsp;−1,&nbsp;0)</code> | son **y**, retourné par le demi-tour |
| <code>2&nbsp;:&nbsp;(0,&nbsp;0,&nbsp;−1)</code> | son **z**, vers le **bas** : la direction d'approche |
| <code>3&nbsp;:&nbsp;(0,&nbsp;0,&nbsp;−L)</code> | la **pointe**, L millimètres sous la bride |

> **Question 3.** Les trois premières colonnes forment la partie
> rotation. Vérifiez que c'est bien une rotation : colonnes unitaires,
> orthogonales deux à deux, déterminant **+1**. Que vaudrait le
> déterminant si l'on n'avait retourné **qu'un seul** axe, et
> qu'est-ce que cela décrirait alors ?

> **Question 4.** Pourquoi le `z` de l'outil descend-il, alors que celui
> de la table monte ? Que se passerait-il si on l'oubliait en écrivant
> un dégagement ?

### 2.4 Vérifier

Une mesure non vérifiée n'est pas une mesure. Faites descendre la
ventouse sur un plan de cote connue et comparez.

- écart **constant**, le même partout → erreur de longueur d'outil :
  corrigez `L_OUTIL` ;
- écart **qui change selon l'endroit** → ce n'est pas l'outil, c'est
  l'origine des axes : reprenez l'étape 1.

---

## 3. Les deux cibles exprimées dans la base

Le repère de la ventouse vient se confondre avec le repère de la cible
lors de l'exécution de la trajectoire. Une cible définit donc la
**position et l'orientation de l'outil à un instant t**.

### 3.1 La prise

Relevez au réglet, depuis le **centre de la base du robot**, où est la
pièce dans le magasin. Elle y est couchée à plat, le long de `x`.

```python
LONGUEUR, LARGEUR, EPAISSEUR = 125.0, 30.0, 10.0

PRISE = np.array([[1,  0,  0,      ...],      # x du magasin
                  [0, -1,  0,      ...],      # y du magasin
                  [0,  0, -1, EPAISSEUR],     # le DESSUS de la pièce
                  [0,  0,  0,        1]], dtype=float)
```

La partie rotation est celle de `OUTIL` : même demi-tour, pour la même
raison — la ventouse descend sur la pièce.

Le `z` mérite qu'on s'y arrête. Le plan du magasin est à 0, la pièce
fait `EPAISSEUR`, la ventouse se pose **dessus** : `z = EPAISSEUR`.

> **Question 5.** Ce seul nombre mélange deux informations : la hauteur
> du plan qui porte la pièce, et l'épaisseur de la pièce. Si vous
> glissiez une cale de 5 mm sous le magasin, quel nombre changeriez-vous
> et de combien ? Et si les pièces faisaient 8 mm ? **Rien dans la
> matrice ne dit laquelle des deux vous corrigez** — notez-le, la
> dernière partie y revient.

### 3.2 La dépose, et l'angle

![La matrice d'une cible orientée, en deux temps](assets/cible_angle.svg)

La pièce est posée sur la structure, tournée de `a`. **Composez la
cible plutôt que de la recopier** — c'est exactement ce que montre la
figure, dans le même ordre :

```python
DEPOSE = (hg.translation(..., ..., EPAISSEUR)   # ou elle se pose
          @ hg.rot_z(np.radians(a))             # 1. Rz(a)
          @ hg.rot_x(np.pi))                    # 2. Rx(180 deg)
```

Trois lignes, trois gestes : *on va là*, *on tourne de `a`*, *on
retourne l'outil pour qu'il approche par le dessus*.

Écrire les seize nombres à la main donne exactement la même matrice —
vérifiez-le plutôt que de le croire :

```python
c, s = np.cos(np.radians(a)), np.sin(np.radians(a))

assert np.allclose(DEPOSE, [[c,  s,  0,       ...],
                            [s, -c,  0,       ...],
                            [0,  0, -1, EPAISSEUR],
                            [0,  0,  0,         1]])
```

Mais la composition **dit ce qu'on fait**, là où la liste de nombres
demande d'être devinée. C'est elle qu'il faut écrire, et c'est pour
cela que la figure part de `Rz(a)`.

### Pourquoi la deuxième rotation n'est pas autour de `x`

C'est le point qui décide de tout, et il tient au **côté** où l'on
multiplie.

Dans `Rz(a) @ Rx(180°)`, la seconde rotation est écrite **à droite** :
elle s'effectue donc **dans le repère intermédiaire** — celui que
`Rz(a)` vient de produire, le repère `(u, v, w)` de la figure. Le
demi-tour se fait autour de **`u`**, et non autour du `x` de la base.

Ce qu'il laisse et ce qu'il retourne se lit alors tout seul :

| | avant | après le demi-tour autour de `u` |
|---|---|---|
| 1re colonne | <code>u&nbsp;=&nbsp;(cos&nbsp;a,&nbsp;sin&nbsp;a,&nbsp;0)</code> | **inchangée** : c'est l'axe du demi-tour |
| 2e colonne | <code>v&nbsp;=&nbsp;(−sin&nbsp;a,&nbsp;cos&nbsp;a,&nbsp;0)</code> | <code>−v&nbsp;=&nbsp;(sin&nbsp;a,&nbsp;−cos&nbsp;a,&nbsp;0)</code> |
| 3e colonne | <code>w&nbsp;=&nbsp;(0,&nbsp;0,&nbsp;1)</code> | <code>−w&nbsp;=&nbsp;(0,&nbsp;0,&nbsp;−1)</code> |

Voilà d'où vient le signe qui surprend.

**Multiplier à gauche donnerait autre chose** : `Rx(180°) @ Rz(a)`
tourne autour du `x` **fixe**, celui de la base, et ne donne pas la
même matrice. Essayez les deux :

```python
print(hg.rot_z(a) @ hg.rot_x(np.pi))    # autour de u : ce qu'on veut
print(hg.rot_x(np.pi) @ hg.rot_z(a))    # autour du x fixe : autre chose
```

> **À retenir.** À droite, on tourne dans le repère qu'on vient de
> construire ; à gauche, dans le repère de départ.

Il reste à savoir lire le résultat :

| Colonne | Ce qu'elle dit |
|---|---|
| <code>0&nbsp;:&nbsp;(c,&nbsp;s,&nbsp;0)</code> | l'axe x de l'outil : la **longueur** de la pièce |
| <code>1&nbsp;:&nbsp;(s,&nbsp;−c,&nbsp;0)</code> | son y, **retourné** par le demi-tour autour de `u` |
| <code>2&nbsp;:&nbsp;(0,&nbsp;0,&nbsp;−1)</code> | son z, vers le bas : la direction d'approche |
| <code>3</code> | où la pièce se pose |

Les composantes de la première colonne **sont** les projections `cos a`
et `sin a` sur les axes de la base — c'est tout ce que dit le premier
panneau de la figure.

C'est `Rz(a) · Rx(180°)` écrit en clair. **À `a = 0`, la partie rotation
redevient celle de la prise** : la prise est le cas particulier de la
dépose où l'on ne tourne rien. Vos deux cibles ne diffèrent donc que
par une position et un angle.

> **Question 6.** Dans `Rz(a)` seul, la deuxième colonne vaut
> `(−sin a, cos a)`. Ici elle vaut `(sin a, −cos a)`. D'où vient ce
> changement de signe, et sur quelle **autre** colonne agit le même
> retournement ? (Comparez les deux matrices de la figure.)

> **Question 7.** Posez `a = 180°`. La pièce occupe-t-elle la même
> place qu'à `a = 0` ? Et la matrice est-elle la même ? Qu'est-ce que
> cela dit d'une pièce symétrique ?

### 3.3 Une orientation n'est pas toujours atteignable

Le poignet ne tourne que dans `[−90° ; +90°]`, et surtout :

```
J4 = orientation − J1
```

Les orientations atteignables sont donc centrées sur **la direction où
pointe le bras**, et non sur zéro. Si la structure est droit devant le
robot (cap 90°), une dépose à `a = 0` demande au poignet exactement
`−90°` : la butée.

Lancez votre programme avec `a = 0` et regardez `robot.controler()`
protester, puis avec `a = 90` et voyez la réserve disparaître.

> **Question 8.** Expliquez, avec la relation ci-dessus, pourquoi
> `a = 90°` ne coûte rien au poignet alors que `a = 0` le met en butée.
> Une pièce étant symétrique, poser à `a` ou à `a + 180°` revient au
> même : cela vous sauve-t-il ici ? Pourquoi ?

---

## 4. Les approches et les retraits : le `@`

Les cibles sont écrites, mais on ne plonge pas droit sur une pièce. Il
faut **approcher par le dessus** et **se retirer par le dessus**.

Ces deux points-là ne se mesurent pas : ils se **composent** à partir de
la cible, **dans le repère de l'outil** :

```python
HAUT = hg.translation(0, 0, -80)     # 80 mm AU-DESSUS

robot.movj(PRISE @ HAUT)             # approche
robot.movl(PRISE)                    # descente
robot.ventouse(True)
robot.attendre(0.3)
robot.movl(PRISE @ HAUT)             # retrait
```

Le `−80` surprend : on monte, et le nombre est négatif. C'est que le `z`
de l'outil pointe vers le bas. Vérifiez-le plutôt que de le croire :

```python
assert (PRISE @ HAUT)[2, 3] > PRISE[2, 3], "le degagement descend !"
```

> **Question 9.** `PRISE @ HAUT` et `HAUT @ PRISE` ne donnent pas la
> même chose. Laquelle déplace l'outil **dans son propre repère** ?
> Essayez l'autre sens, regardez où part la cible, expliquez.

> **Question 10.** Le dégagement est le **seul** endroit de ce programme
> où vous composez deux matrices. Qu'est-ce qui le distingue des cotes
> que vous avez relevées au réglet ? Pourquoi n'avez-vous **pas** eu à
> le mesurer ?

---

## 5. Le lissage

Sans lissage, le robot **s'arrête à chaque point de passage** — y
compris aux points de dégagement, où il n'a rien à faire que passer.

```python
robot.movj(PRISE @ HAUT, lissage=70)      # pourcentage, de 0 a 100
```

`lissage` porte sur le raccord **avec l'instruction suivante**. Les deux
segments se **recouvrent** pendant une durée, sur laquelle la consigne
est le mélange pondéré des deux. À 100 %, le recouvrement vaut 49 % du
plus court des deux segments.

### 5.1 Ce que cela rapporte

```python
print(robot.resume())
```

La première ligne affiche les secondes gagnées. Relevez-les :

| `lissage` | durée du cycle |
|---|---|
| 0 % | … |
| 40 % | … |
| 70 % | … |
| 100 % | … |

### 5.2 Ce que cela coûte

**Un point lissé n'est plus atteint.** C'est le prix, et il se mesure :

```python
traj = robot.programme.construire(robot.robot, robot.q_depart)
P = np.asarray(traj.echantillonner(0.005)["P"])
print(np.min(np.linalg.norm(P - (PRISE @ HAUT)[:3, 3], axis=1)), "mm")
```

> **Question 11.** Relevez cet écart pour les quatre valeurs de lissage.
> Faites la même mesure sur le **point de prise**. Que trouvez-vous, et
> pourquoi ?

### 5.3 La règle

**On ne lisse jamais un point où l'outil agit.** La ventouse qui aspire
à côté de la pièce n'emporte rien. Dans le cycle ci-dessus, les
dégagements sont lissés et les deux points de contact restent à zéro.

Le modèle vous protège à moitié :

- il **refuse** de lisser autour d'une `attendre()`, parce qu'une
  temporisation demandée doit être tenue ;
- il **accepte** de lisser un point où vous commandez la ventouse. Une
  action ne demande aucun temps, donc elle ne neutralise rien. Lisser
  un point où l'on agit, c'est agir avant d'y être.

> **Question 12.** Lissez volontairement le point de prise à 100 %, et
> mesurez **où est la ventouse à l'instant où elle aspire** :
>
> ```python
> ev = [e for e in robot.programme.evenements if e.type == "OUTIL" and e.etat]
> T = robot.robot.mgd(traj.q(ev[0].t))
> print(np.linalg.norm(T[:3, 3] - PRISE[:3, 3]), "mm")
> ```
>
> Vous trouverez **quelques centièmes de millimètre**, et non les
> quelques millimètres annoncés. Expliquez pourquoi, sachant que le
> recouvrement est plafonné à 49 % du **plus court** des deux segments
> voisins, et qu'une action dure un battement de cil.
>
> Puis dites dans quel cas la même faute coûterait cher : quel genre
> d'instruction faudrait-il trouver après le point de prise pour que
> l'écart devienne dangereux ?

### 5.4 Sur la machine réelle

Ce contrôleur-ci **n'a pas de lissage en PTP**. À l'envoi, le pont
l'écrit noir sur blanc :

```
NOTE : le lissage n'existe pas en PTP sur le controleur ;
       le robot reel marquera un arret a chaque point.
```

Le simulateur vous montre donc ce que vous **gagneriez** avec un
contrôleur qui a des zones (ABB `zonedata`) ou du CNT (Fanuc). C'est
une mesure utile : elle chiffre ce que coûte un arrêt inutile.

> **Question 13.** Sur votre relevé du 5.1, quelle fraction du temps de
> cycle est perdue en arrêts ? Sur une journée de production à
> 3 000 cycles, cela fait combien ?

---

## 6. Voir la pièce bouger

Le simulateur peut dessiner la pièce et la faire **suivre la
ventouse** :

```python
robot.poser_piece(x=..., y=..., dims=(125, 30, 10), magasin=True)
```

C'est du **décor** : aucune instruction, aucune trajectoire, rien n'est
changé au programme. Mais une prise qui vise à côté se voit alors
immédiatement.

> Si la fenêtre se couvre d'arcs bleus et de points jaunes, le menu
> **Affichage ▸ Vue épurée** (Ctrl+Shift+V) ne laisse que la cellule.

---

## 7. Valider hors ligne, puis transférer sur la machine

```bash
python votre_programme.py          # sans --reel : le simulateur
```

```python
print(robot.resume())
for a in robot.controler():
    print(" -", a)
```

Le modèle vous dit, **machine éteinte**, ce que la machine découvrirait
à vos dépens.

**Ne transférez sur le robot qu'une fois la simulation validée.**

```bash
python votre_programme.py --reel
```

---

## 8. Ce que cette façon de faire vous coûtera

Votre programme marche. Il a pourtant un défaut, et il faut le voir
maintenant, pendant qu'il ne coûte encore rien.

**Le magasin a été déplacé de deux centimètres.** Cela arrive : on
libère la table, on change de poste, et les pièces ne sont plus tout à
fait au même endroit. Mesurez sa nouvelle position et corrigez votre
programme pour que le cycle fonctionne de nouveau.

> **Question 14.** Combien de nombres avez-vous dû reprendre, et dans
> combien de matrices ? Jusqu'ici, c'est supportable.
>
> Maintenant, **sans l'écrire**, répondez à ceci : l'activité suivante
> demande **douze** dépôts, à six hauteurs et six orientations
> différentes, tous définis par rapport au centre de la structure. Si ce centre
> bouge de deux centimètres, combien de nombres faudrait-il reprendre
> en écrivant les cibles comme ici ?

> **Question 15.** Reprenez la question 5. Dans `PRISE`, la cote `z`
> vaut `EPAISSEUR` parce que le plan du magasin est à zéro. Si le plan
> était à 60 mm, vous écririez `60 + EPAISSEUR` — **un seul nombre pour
> deux faits indépendants** : où est le magasin, et quelle est
> l'épaisseur de la pièce.
>
> Proposez une façon d'écrire la cible qui **sépare** ces deux faits,
> de sorte que déplacer le magasin et changer l'épaisseur des pièces
> ne touchent pas au même endroit du programme. Vous venez de décrire
> un **repère de programmation** — et c'est le sujet de la prochaine
> séance.

---

## Ce qu'il faut rendre

1. Le programme, commenté — chaque cote mesurée doit dire **d'où elle
   vient** et **par rapport à quoi**.
2. Les réponses aux quinze questions.
3. Le tableau du 5.1 et celui de la question 11, avec votre conclusion
   sur ce que le lissage achète et ce qu'il coûte.
4. Votre proposition de la question 15, même imparfaite. On la comparera
   à la solution au début de la séance suivante.
