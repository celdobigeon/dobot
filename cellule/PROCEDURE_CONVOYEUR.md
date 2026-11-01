# Mise en route de la cellule convoyeur

Procédure d'atelier — université de Bordeaux
Version du 30/10/2026 · à lire avec [la programmation hors ligne](PROCEDURE_HORS_LIGNE.md)

---

## 0. Pourquoi une procédure, et pas une liste de commandes

Chaque étape ci-dessous existe parce qu'un défaut **réel** s'est produit sans
elle, sur cette cellule, et a coûté du temps. L'ordre n'est pas une
convention : une étape faite trop tôt est une étape perdue, et pire, elle
donne un résultat qui a l'air juste.

| Étape | Ce qu'elle évite | Comment le défaut se manifeste |
|---|---|---|
| 1. dégager le bras | prise d'origine impossible | voyant rouge, « EXÉCUTION REFUSÉE » |
| 2. prise d'origine complète | repère faux | le bras va à côté, **de plus en plus loin** |
| 3. étalonnage **après** l'origine | point appris dans un repère mort | la ventouse plonge à côté d'une distance fixe |
| 4. étalonnage **dans les conditions du cycle** | décalage bord/centre | erreur d'une **demi-pièce**, toujours du même côté |
| 5. une seule pièce à la fois | détection sur une pièce déjà là | le tapis ne convoie pas, la prise est erratique |
| 6. **ne jamais pousser le bras à la main** | J1 perdu par le contrôleur | le bras garde son rayon et change d'**angle** |
| 7. partir d'une posture au **milieu** de la course | prise d'origine en butée | la base tourne, tape, et force sans s'arrêter |

> Le piège commun à toutes : **rien ne proteste**. Le programme s'exécute,
> le tapis tourne, le bras bouge. Seule la pièce non attrapée le dit, et
> elle ne dit pas pourquoi.

---

## 0 bis. Vérifier que le robot sait où est son bras

**À faire avant tout le reste**, et **sur chaque machine** : l'état des
capteurs d'angle ne se devine pas, et il ne se transporte pas d'un robot
à l'autre.

| Machine | Port, pont | Firmware | Capteurs d'angle |
|---|---|---|---|
| robot au convoyeur | COM6, CP210x | 3.7.0 | **J3 fautif, ~160°** |
| **robot 2** | COM9, CP210x | 3.7.0 | **J1 +21°, J3 +35°** à chaque réouverture |
| robot « 7 » | CH340 | — | sains |
| robot 5 | COM4, CH340 | 3.7.0 | sains |

### Ne relevez pas les capteurs juste après avoir déplacé le bras

Sur le robot 2, le 30/10/2026, quatre relevés de cinq lectures :

| Bras venait d'être déplacé à la main | Étendue J3 |
|---|---|
| oui | 20,8° |
| non | 0,4° |
| oui | 19,2° |
| non | 0,5° |

La corrélation est sans exception. **Bras immobile depuis un moment, ce
capteur se répète à un demi-degré près** ; relevé dans la foulée d'un
déplacement, il donne d'abord l'ancienne valeur puis la nouvelle — dans
un des essais, la bascule est arrivée à la **troisième** lecture, pas à
la première.

> **J'avais d'abord conclu à un capteur fautif**, et c'était faux : le
> protocole du relevé était en cause, pas la machine. C'est l'utilisatrice
> qui a vu le délai. La leçon vaut pour les deux sens — mesurer avant
> d'accuser la machine, et se méfier de sa propre mesure quand elle
> accuse.

Attendez donc que le bras soit posé et relâché **avant** de lancer
`../atelier/diag_capteurs_angle.py`.

### Le délai, mesuré

Vingt lectures enchaînées, bras relâché puis relevé aussitôt :

```
  t(s) |   J1     J2     J3
   0.6 |  21.1    9.0   29.7      <- périmée
   6.6 |   0.0    8.7   75.9
  12.6 |   0.0    9.3   76.3
   ...      ...                   <- et ainsi jusqu'à 20 lectures
```

**Une seule lecture fausse, sur deux axes à la fois, et tout stable
ensuite.** Le contrôleur rend d'abord une valeur périmée, puis la bonne
dès la connexion suivante — six secondes ici, qui sont le temps d'une
réouverture de liaison sur cette carte, pas forcément celui de
l'établissement lui-même.

> **Attention à ne pas en faire une règle trop simple.** « La première
> lecture ment » est faux en général : sur la machine du convoyeur, bras
> en équerre, c'est la première qui est juste (−2,5°) et les quatre
> suivantes qui mentent (−160°). Les deux motifs sont symétriques, et
> leur position ne les sépare pas.

### L'essai qui conclut : `--origine`

```bash
python ../atelier/diag_capteurs_angle.py COM9 --origine
```

Trois façons de donner une posture de référence au bras, et **deux sont
mauvaises** :

| Méthode | Pourquoi elle échoue |
|---|---|
| le placer **à la main** (équerre) | imprécis, et surtout **J1 est perdu** — la base n'a pas de capteur |
| lui **commander** le repli | n'arrive à destination que si le contrôleur sait d'où il part. Sur le robot 2, J3 faux de 34° : le bras est parti **en butée** |
| la **prise d'origine** | amène les axes sur leurs références mécaniques, sans rien supposer |

Seule la troisième ne demande aucune hypothèse. Le script fait donc
l'origine, lit les angles **dans la liaison qui vient de la faire**,
puis ferme, rouvre, et relit. L'écart est l'erreur du contrôleur,
mesurée sans intermédiaire.

Relevé sur le robot 2 le 30/10/2026, bras immobile :

| | J1 | J2 | J3 |
|---|---|---|---|
| référence, après l'origine | −21,5 | 15,2 | 30,1 |
| réouverture, 4 s plus tard | **0,0** | 15,4 | **65,0** |
| réouverture, 10 s plus tard | **0,0** | 15,5 | **65,2** |

**J2 juste, J1 faux de 21°, J3 faux de 35°**, et le bras n'a pas bougé.
C'est pourquoi un lancement réussissait et le suivant non : chacun
ouvre sa propre liaison.

> Notez que **J1 n'a pas de capteur du tout** — le contrôleur le compte
> en pas. Déclarer la posture d'équerre corrige J2 et J3, pas J1 : seule
> une prise d'origine le rétablit. C'est pourquoi `--equerre` fait les
> deux, **dans la liaison qui travaille**.

### Le vrai remède : UNE SEULE LIAISON, tenue par la fenêtre

Le défaut se produit **à l'ouverture de la liaison** : le contrôleur
relit ses capteurs à ce moment-là, et une fois sur deux il se trompe.
Tout ce qui ne rouvre jamais y échappe.

C'est pourquoi `test_prise.py`, lancé depuis la fenêtre du simulateur,
marche sans faute depuis toujours — y compris après un arrêt d'urgence,
une suspension ou un déplacement manuel. Il ne rouvre rien : la fenêtre
tient le port série du début à la fin de la séance.

**Et les exemples savent déjà passer par là.** `executer_sur_robot`
demande d'abord au simulateur s'il est ouvert et connecté ; si oui, il
lui confie le programme, qui part sur SA liaison. Le tapis et le capteur
font de même.

```bash
python main.py                      # 1. ouvrir la fenetre
                                    # 2. panneau Robot reel : CONNECTER,
                                    #    puis MISE EN ROUTE -- les deux
python examples/convoyeur.py --reel # 3. et autant de lancements qu'on veut
```

**Les deux gestes de l'étape 2, pas un seul.** La fenêtre ouvre elle
aussi une liaison, une fois, et cette lecture-là peut être fausse —
auquel cas elle l'est pour toute la séance. Ce qui la rattrape, c'est la
**prise d'origine contenue dans la mise en route** : faite dans la
liaison de la fenêtre, elle corrige la croyance du contrôleur quoi
qu'il ait lu à l'ouverture.

> Vérifié dans les deux sens le 31/10/2026 sur le robot 2. Connexion
> **sans** mise en route : le bras a perdu son origine dès le premier
> cycle. Connexion **avec** : trois pièces, trois prises bonnes, et cinq
> lancements consécutifs sans une faute auparavant.

La fenêtre ne guérit donc pas le capteur — elle **réduit l'exposition**
à une seule lecture au lieu d'une par lancement, et concentre le risque
en un instant unique. Mais cet instant-là, contrairement aux autres, est
**vérifiable** : l'étape 3 de l'assistant compare le modèle 3D au vrai
bras. S'ils ne coïncident pas, déconnectez, reconnectez, recommencez —
et ne lancez rien avant.

> **Le prix à payer, et il se mesure.** La passerelle ajoute son temps
> de transit : la détection→arrêt passe de 9 ms à **56–119 ms**, soit 3
> à 7 mm de bande au lieu de 0,6.
>
> `RETARD_ARRET` dans l'exemple le compense, déclaré **en secondes** et
> converti avec la vitesse du tapis. Mais il n'en corrige que la
> **médiane** : la dispersion, elle, reste, et c'est la ventouse qui
> l'absorbe. Avec une pince, ou des pièces plus petites, il faudrait
> fiabiliser la chaîne plutôt que la corriger.
>
> Les deux chemins, sur les relevés du 31/10/2026 :
>
> | | retards mesurés | étendue |
> |---|---|---|
> | direct | 9, 9, 10, 11, 11 ms | **2 ms** |
> | par la fenêtre | 56, 64, 65, 66, 66, 66, 66, 67, 67, 83, 101, 119 ms | **63 ms** |
>
> Le direct n'est pas seulement six fois plus rapide : il est trente
> fois plus régulier. C'est lui qu'il faudrait pour une pince — et donc
> une machine dont les capteurs ne mentent pas.

### Le contournement par `--equerre`, vérifié aussi

```bash
python examples/convoyeur.py --reel --equerre
```

Trois lancements séparés sur le robot 2, trois prises bonnes. Chacun
rouvre sa liaison, donc relit les capteurs fautifs, et se corrige
lui-même.

**Et les deux derniers l'ont fait sans déclarer l'équerre** : le bouton
« Passer » du dialogue, puis la seule prise d'origine. Si cela se
confirme, l'ingrédient actif est **l'origine dans la liaison qui
travaille**, pas la déclaration de posture — ce qui supprimerait le
placement à la main, donc la perte de J1 qu'il entraîne.

> **Deux succès ne font pas une démonstration.** Cette machine
> réussissait déjà une fois sur deux : deux réussites de suite arrivent
> par hasard une fois sur quatre. À confirmer sur cinq ou six
> lancements avant d'alléger la procédure.

### Et surtout : répétable ne veut pas dire juste

Dans ce relevé-là, les dix-neuf lectures stables annonçaient **J1 = 0,0
alors que le bras était physiquement à −21°**. La valeur répétée était
la fausse, et J3 était faux aussi.

Deux conséquences, et elles priment sur tout ce qui précède :

**Un essai sans posture connue ne prouve rien.** Il ne mesure que la
répétabilité ; un contrôleur qui se fixe sur une lecture fausse et s'y
tient passe pour sain. Seule l'**équerre** — où l'on sait que J2 et J3
valent zéro — permet de conclure.

**Et le relevé lui-même violait l'étape 6.** Pour mettre le bras dans
une posture d'essai, on l'avait poussé à la main — or la rotation de
base **n'a pas de capteur** : le contrôleur la compte en pas, et
moteurs relâchés ces pas sont perdus. J1 était donc faux *par
construction*, avant même la première lecture. Les trois relevés faits
ainsi ne mesuraient pas la machine, mais les dégâts de la méthode.

> Après tout déplacement manuel : **prise d'origine d'abord**, mesure
> ensuite. Dans cet ordre, et jamais l'inverse.

C'est l'**étiquette** qui identifie une machine. Ni le port, ni le pont
USB : le CH340 est partagé par plusieurs robots, et le port change d'une
prise à l'autre.

```bash
python ../atelier/diag_capteurs_angle.py COM6
```

Bras en équerre — arrière vertical, avant-bras horizontal — les angles
lus doivent valoir `J2 = 0` et `J3 = 0`. Le script relit cinq fois, en
rouvrant la liaison à chaque essai, et conclut. Relevé le 30/10/2026 sur quatre
connexions successives, bras immobile : `J3 = −2,5`, puis `−162,6`,
`−157,8`, `−161,8`. **La bonne valeur sort une fois sur quatre.**

Dans une même connexion la lecture ne bouge pas d'un dixième : l'angle
est lu **une seule fois à l'ouverture de la liaison**, puis compté en
pas. Une liaison ouverte sur une mauvaise lecture restera fausse
jusqu'à sa fermeture.

> **Ce que ça produit, et pourquoi c'est si trompeur.** Des prises
> décalées sans régularité, des alarmes de butée sans raison, des
> postures annoncées impossibles — et surtout des essais qui
> **réussissent**, ceux où la lecture était bonne. On cherche alors
> l'erreur dans le programme, les repères, les cotes. Partout sauf dans
> la machine.

### Le contournement, et pourquoi il doit être dans la même liaison

Le contrôleur relit ce capteur **à chaque ouverture de liaison**. Une
posture déclarée dans une autre liaison — par `dobot.py posture
equerre`, par exemple — est donc perdue dès que celle-ci se referme.

D'où la règle : **bras en équerre, et déclaration dans la liaison du
programme**.

```bash
python examples/prise_depose_matrices.py --reel --equerre
```

L'assistant s'ouvre déjà tout seul **quand des alarmes sautent**,
c'est-à-dire quand la lecture fautive sort des butées du contrôleur. Il
répare alors le repère au bon endroit, et le cycle est juste — c'est
pourquoi certains essais réussissent.

Mais une lecture fausse qui reste **plausible** ne déclenche rien. Le
cycle part sur un repère faux, en silence, et c'est le cas le plus
coûteux : on cherche l'erreur partout ailleurs. `--equerre` rend la
déclaration systématique au lieu de la laisser au hasard.

`executer_sur_robot` refuse en outre de partir si un axe annonce une
valeur hors de ses propres butées. Mais une erreur de 160° qui reste
*dans* les butées passe au travers : la déclaration en équerre reste la
seule sûre.

---

## 0 ter. Le firmware, et ce qu'une mise à jour emporte

**Il faut au moins la 3.7.0 pour lire le capteur du convoyeur.** Le
robot 5 était en 3.3.0 : sa diode s'allumait à chaque passage de pièce,
et le contrôleur rendait la **même valeur sur un port occupé que sur un
port vide** — quatre ports balayés, vingt entrées EIO, quatre formes de
trame essayées (`../atelier/diag_capteur_trame.py`), toutes figées. Rien ne dit
que la commande n'existe pas : ce contrôleur acquitte tout.

Passé en 3.7.0, le capteur répond du premier coup.

### Après une mise à jour, reprenez l'origine — et sachez ce qui a bougé

Un flashage n'est pas une mise sous tension : il réécrit la flash.
Relevé sur le robot 5 le 30/10/2026 :

- les **paramètres d'effecteur** sont passés de `xBias = 61` à **zéro**.
  Le pont les relit à chaque connexion et s'y adapte seul, mais le
  contrôleur annonce désormais des x inférieurs de 61 mm pour la même
  posture — ne vous y trompez pas en comparant des relevés ;
- le **point de parcage** se relit 61 mm plus court, et désigne le même
  point physique : c'est la même conversion, pas une valeur perdue.
  Inutile de relancer `../atelier/reparer_point_origine.py` ;
- `GetHOMEParams`, muet en 3.3.0, **répond** en 3.7.0 ;
- l'avis `0x00` s'efface de nouveau normalement.

**Sans prise d'origine après le flashage, le bras se décale sur tous
les axes.** C'est la panne qui suit immédiatement une mise à jour, et
elle ressemble à s'y méprendre à une erreur de repère.

---

## 0 quater. Deux pièges qui n'existent que sur le firmware 3.3.0

Relevés le 30/10/2026 sur le robot 5, et ils ont coûté une
après-midi à eux deux. Les deux sont **muets** : la machine est saine,
rien ne proteste, et rien ne part.

### L'avis de réinitialisation ne s'efface jamais

`alarme 0x00 : reinitialisation du controleur` reste affichée quoi qu'on
fasse — ni l'effacement, ni une prise d'origine menée à son terme ne la
font tomber. Sur le 3.7.0, elle tombait.

Ce n'est **pas un défaut**, c'est un avis : « j'ai été remis sous
tension, l'origine est à refaire ». Le code la laisse donc passer
(`liaison.alarmes_bloquantes`) et s'arrête sur tout le reste. Avant
cette correction, elle rouvrait l'assistant de mise en route **à chaque
commande**, sans qu'aucun geste puisse l'en sortir.

### Le point de parcage était enregistré hors course

Celui-ci est le plus retors de toute la mise au point. L'endroit où le
bras se gare **à la fin** de la prise d'origine est un nombre
**persistant, en flash**. Sur cette machine il valait J1 ≈ −135,2° pour
une butée à −135.

Conséquence : la prise d'origine réussissait, la base retrouvait bien sa
référence — et le bras finissait **hors de sa propre course**. Au
premier mouvement, `0x41 : J1 en butee negative`, et plus rien ne
partait. Le contrôleur disait la vérité ; c'est l'endroit où il se garait
qui était faux.

Et rien ne le montre : `GetHOMEParams` renvoie **zéro octet** sur ce
firmware. On ne peut pas relire la valeur fautive, seulement la
remplacer.

```bash
python ../atelier/reparer_point_origine.py COM4
```

Le bras va au repli, le script lit ce que le contrôleur annonce **de
lui-même** à cet endroit, et écrit cela. Aucune conversion de repère ne
s'interpose, donc aucune ne peut se tromper. À faire **une fois par
machine** ; vérifiez derrière par une prise d'origine, le bras doit se
garer au repli et `etat` annoncer J1 proche de zéro.

---

## 1. Dégager le bras

> ### La main près de l'interrupteur pendant la prise d'origine
>
> Sur une machine dont un capteur d'angle ment, **l'origine part d'un
> endroit que le contrôleur croit ailleurs**. Relevé sur le robot 2 :
> une fois elle est partie en butée, une autre fois le bras est
> **descendu trop bas en z** et il a fallu couper l'alimentation.
>
> Deux gestes qui l'évitent, et ils ne coûtent rien :
>
> - **avant la mise sous tension**, bouton de déverrouillage enfoncé,
>   placez le bras haut et franc — avant-bras bien relevé, loin du tapis
>   — et la base au milieu de sa course. Rien ne bouge à l'allumage ;
> - **pendant la prise d'origine**, gardez la main près de
>   l'interrupteur. Si le bras descend, coupez.
>
> Si elle plonge deux fois de suite, n'insistez pas : cherchez pourquoi.
> Une origine qui plonge finit par coûter une ventouse.

Si le voyant est **rouge**, ou si un cycle a été interrompu, le bras est
probablement contre une butée.

Bouton de la fourche maintenu, amenez-le à une posture franche : **bras
arrière vertical, avant-bras horizontal**. Relâchez.

**Et ramenez la base au milieu de sa course**, bras face à l'avant.
Partie d'une butée, elle n'a de la place que d'un côté : sur le robot 5,
la prise d'origine lancée depuis J1 = −135° a fait tourner la base
jusqu'à la butée, où elle a forcé sans s'arrêter. Repartie du milieu, la
même commande a trouvé son index du premier coup.

> Tant qu'il touche une butée, la prise d'origine échoue — et elle échoue
> en affichant « prise d'origine terminée », ce qui est trompeur. C'est la
> ligne suivante qui compte.

---

## 2. La prise d'origine, jusqu'au bout

```bash
python dobot.py mise-en-route
```

**Attendez la ligne `mise en route terminee : defauts effaces`.** C'est
elle, et non « prise d'origine terminée », qui dit que la machine est
prête. Coupée avant, les alarmes restent et tout ce qui suit échoue.

> Le bras bouge pendant cette étape. Dégagez la zone.

---

## 3. L'étalonnage du point de prise

```bash
python ../atelier/etalonner_convoyeur.py COM6
```

Trois fois : le tapis part, vous posez une pièce **bien en amont**, le
capteur l'arrête, et vous amenez la ventouse **au contact du centre** de
la pièce — **au clavier**, par pas de 1 à 25 mm.

> **Ne poussez jamais le bras à la main.** Le contrôleur relit deux
> capteurs d'angle, bras arrière et avant-bras ; la rotation de base n'en
> a pas, il la compte en pas. Moteurs relâchés, ces pas ne sont pas
> comptés : **J1 est perdu**, silencieusement, jusqu'à la prochaine prise
> d'origine. Mesuré ici : trois poses sur le même point fixe, même rayon
> à 1,4 mm, même hauteur à 0,3 mm, et **49° d'écart angulaire** — soit
> 241 mm. C'est la panne la plus coûteuse de toute la mise au point, et
> la plus silencieuse.

**À refaire après chaque prise d'origine.** Un point appris n'existe que
dans le repère où il a été appris ; reprendre l'origine le périme.

**Posez la pièce loin du capteur.** Posée juste devant, elle n'a pas le
temps d'être entraînée, et on étalonne le geste de la main plutôt que la
cellule.

### Lisez l'étendue

Le script affiche l'étendue sur les trois essais. Relevés sur cette
cellule : **1,5 à 3 mm**. C'est elle qui dit si la cellule tient la
répétabilité, **avant** de lui demander de produire.

| Étendue | Ce que ça veut dire |
|---|---|
| moins de 5 mm | la prise sera franche |
| 5 à 15 mm | la ventouse pardonne, une pince non |
| plus de 15 mm | ne continuez pas : reprenez la pose des pièces |

### Ce que le point appris absorbe

Et qu'il est donc inutile de mesurer séparément :

- la place et l'orientation réelles du tapis ;
- l'abscisse réelle du capteur le long de la bande ;
- le retard de la chaîne de scrutation ;
- le décalage entre le **bord avant** de la pièce, que le capteur voit, et
  son **centre**, que la ventouse vise — une demi-pièce, soit 12 mm sur des
  pièces de 25 ;
- la hauteur de la bande et l'écrasement de la ventouse.

> Un point appris vaut mieux qu'un point calculé sur des cotes relevées au
> décimètre. C'est ainsi qu'on procède sur une vraie cellule.

---

## 4. Le cycle

```bash
python examples/convoyeur.py --reel
```

Posez **une pièce à la fois**, bien en amont, et laissez le capteur
l'arrêter.

Le programme affiche à chaque cycle le retard entre la détection et
l'arrêt du tapis, et récapitule à la fin. Relevés ici : **5 à 11 ms**, soit
moins d'un millimètre de bande. Si la pièce dépasse de plusieurs
centimètres, **ce n'est pas le programme** — le récapitulatif le dit.

---

## 5. Si la prise rate

Dans cet ordre, du plus probable au moins probable.

**Reprenez l'étape 3.** Neuf fois sur dix, l'étalonnage est périmé : une
prise d'origine, une coupure, un tapis déplacé.

**Vérifiez où va le bras**, sans qu'il descende :

```bash
python ../atelier/diag_vise_juste.py COM6
```

Il monte 70 mm au-dessus du point appris et mesure l'écart à la consigne.
S'il annonce 0 mm, le bras obéit : c'est la **cible** qui est fausse, donc
l'étalonnage. S'il annonce plusieurs millimètres, c'est la chaîne de
commande, et le problème est ailleurs que dans le convoyeur.

**Mesurez la dérive du repère** si le doute persiste :

```bash
python ../atelier/diag_derive_origine.py --poser COM6     # enregistre un point fixe
python ../atelier/diag_derive_origine.py COM6             # y revient
```

Choisissez un **angle franc** de la cellule, pas une surface plane.
Mesurez **deux fois de suite sans rien changer** : ce premier écart n'est
pas une dérive, c'est la fidélité de votre geste. Rien en dessous ne
compte.

---

## 6. Les autres diagnostics

À n'utiliser que si l'étape 5 n'a rien donné. Chacun est né d'une fausse
piste de la mise au point, et chacun ne répond qu'à une question.

| | |
|---|---|
| `../atelier/diag_capteur.py` | sur quel port le capteur répond (GP ou EIO) |
| `../atelier/diag_capteur_vivant.py` | le capteur se déclenche-t-il seul, et où bascule-t-il |
| `../atelier/diag_convoyeur.py` | le moteur du tapis reçoit-il ses commandes |
| `../atelier/diag_depassement.py` | la pièce dépasse-t-elle le capteur, et selon quelle loi |
| `../atelier/diag_file_tapis.py` | un arrêt immédiat passe-t-il quand la file tourne |
| `../atelier/diag_arret.py` | le capteur, l'arrêt et le glissement, séparés |

---

## 7. Deux particularités de cette cellule

**Le capteur se déclenche sur le CENTRE de la pièce, pas sur son bord
avant.** On avait écrit le contraire, en raisonnant : le faisceau voit
d'abord le bord qui arrive, donc la ventouse doit viser une demi-pièce
en amont. Essai du 31/10/2026 : la ventouse se posait une demi-pièce
trop en amont, exactement. Le raisonnement était propre et il était
faux — faisceau large, ou surface réfléchissante insuffisante avant le
milieu du cube. La cible vise donc le capteur, plus la dérive du retard,
et rien d'autre.

**Le capteur est un infrarouge sur GP4.** La file du contrôleur ne sait pas
attendre sur un port GP : le cycle est donc **mené depuis le PC**, morceau
par morceau, et le robot reste attaché à l'ordinateur pendant tout le
cycle. Un capteur tout-ou-rien sur une entrée EIO rendrait la machine
autonome — c'est un choix d'implantation, pas un détail de câblage.

> **Et on l'a déjà.** Mesuré le 30/10/2026 par
> `../atelier/diag_capteur_autonome.py` : la broche du capteur se lit **aussi en
> entrée numérique**, et elle bouge **fonction infrarouge éteinte**. Ce
> n'est donc pas la commande 138 qui la pilote — l'entrée vaut pour
> elle-même, et la file du contrôleur peut l'attendre.
>
> | Port GP | Entrée numérique |
> |---|---|
> | GP4 | **EIO 7** |
> | GP5 | EIO 5 |
>
> **C'est fait**, et le cycle tourne PC débranché :
>
> ```bash
> python examples/convoyeur.py --reel --autonome
> ```

### Les trois choses que le cycle autonome a coûtées

Elles valent d'être connues avant de l'écrire ailleurs.

**La polarité est inversée.** Pièce présente : GP4 vaut `True`, EIO 7
vaut `False`. On attend donc `etat=False`. Garder `True` par habitude
retourne le cycle — le robot attendrait que la pièce *parte*.

**`TRIG` attend un NIVEAU, pas un front.** Une pièce déjà devant le
capteur satisfait la condition aussitôt : au premier essai, tout le
cycle s'est déroulé **dans le vide**, sur une pièce restée là du cycle
précédent, et la file est allée jusqu'au bout sans rien attendre. On
écrit donc un front avec **deux attentes** — capteur libre, puis pièce
qui arrive.

> Et la morale du détour : j'ai d'abord accusé `TRIG` de ne pas
> bloquer, parce que ce contrôleur acquitte n'importe quelle trame.
> `../atelier/diag_attente_file.py` a montré qu'il bloquait très bien. **Mesurer
> avant d'accuser la machine** — c'est la troisième fois de la journée
> que le défaut était chez nous.

**La file ne tient que 32 commandes, et on n'en remplit que 24** —
la marge de travail, ramenée à la baisse si le contrôleur refuse. Un
cycle en demande 23 : **il passe d'un cheveu, et deux ne passeront
jamais**. Trois cycles demandés, 63 commandes, 39 de trop : refusé.

Ce qui déborde est gardé par le PC et envoyé pendant l'exécution. Le
programme démarre alors, il a l'air de bien partir, et il n'est **pas**
autonome. `deposer_sur_robot` refuse plutôt que de le laisser croire.
Une cellule autonome se dimensionne à la file de son automate.

> Vérifié le 30/10/2026 sur le robot 5 : **trois lancements, trois
> prises bonnes**, liaison série refermée à chaque fois.

**Le pont USB est un CP210x** sur la machine au convoyeur. La carte reste
muette un instant après la mise sous tension : la première connexion
échoue parfois en « aucune réponse en 1,0 s ». Relancez, la seconde
passe. Le robot 5, lui, est en CH340 et n'a pas ce défaut — une raison
de plus de ne jamais identifier une machine par son pont.

---

## 8. Ce qui reste ouvert

**La dérive du repère entre deux prises d'origine** reste à mesurer
proprement. Celle qu'on a observée venait du guidage manuel, pas de
l'origine elle-même : il faudra refaire la mesure en ne déplaçant le bras
qu'au clavier, et c'est seulement alors qu'on saura si l'origine est
répétable.

**L'axe linéaire** n'est pas monté. Sa direction, `+y`, est une hypothèse
déclarée comme telle dans le code, avec la procédure pour la mesurer.

**Deux façons de déclarer l'outil qui ne commandent pas la même
hauteur.** C'est le défaut le plus gênant de ceux qui restent, parce
qu'il est invisible à la lecture :

```python
robot.definir_repere_outil(Tz(-89.3) @ Rx(180))   # geo.lt reste au catalogue (62)
robot.regler_longueur_outil(89.3)                 # geo.lt = 89,3
```

Les deux produisent le **même repère outil**. Mais c'est `geo.lt`, et
non le repère outil, qui entre dans la conversion du pont
(`z_ctrl = z_tcp + lt − d1 + zBias`) : 27,3 mm d'écart à la commande.

**D'où vient ce nombre, exactement** (relevé le 01/11/2026) :

```
89,3  la longueur mesurée, déclarée par la matrice
62,0  la longueur de la ventouse au CATALOGUE, que geo.lt conserve
----
27,3  l'écart — et c'est toute l'explication
```

Ce n'est donc ni l'origine, ni la pose à l'équerre. Celles-ci déplacent
bien la hauteur absolue, et c'est pour cela qu'il faut revérifier
`L_OUTIL` à chaque séance — mais elles ne peuvent pas produire **cet**
écart-là : les deux déclarations se font dans la même séance, avec la
même origine, et ne diffèrent que par une ligne de code.

**Et on ne le corrige pas, délibérément.** Aligner `geo.lt` sur le
repère déclaré est une ligne, et elle rendrait les deux chemins
identiques — mais elle déplacerait de 17 à 27 mm la cote commandée par
des programmes **réglés sur la cellule** : le convoyeur, la structure
de kapla. Ils plongeraient ou survoleraient dès la première exécution.
C'est à la machine de dire laquelle des deux valeurs est la bonne.

Depuis le 01/11/2026, **le pont le signale au téléversement** :

```
ATTENTION : outil declare a 89.3 mm, mais geo.lt vaut 62.0.
La cote envoyee au controleur differe de -27.3 mm de ce que montre
le simulateur -- verifiez la hauteur par une descente avant de
lancer le cycle.
```

**L'essai qui tranchera**, et il tient en dix minutes : monter le
porte-stylo — dont la longueur catalogue, 70 mm, est aussi celle qu'on
déclare, donc sans écart — puis le descendre sur la feuille.
`L_OUTIL` est alors réglé sans ambiguïté. Remonter ensuite la ventouse,
refaire la même descente, et comparer la valeur qui écrit juste à la
valeur mesurée au réglet. Si elle vaut 62, c'est `geo.lt` qui a raison
et la mesure géométrique compte autre chose ; si elle vaut 89,3, c'est
la conversion qu'il faut corriger.

Les exemples s'en tiennent tous à la première forme, celle qui est
validée sur la machine. Tant que l'essai n'est pas fait, **ne mélangez
pas les deux déclarations dans un même programme**.
