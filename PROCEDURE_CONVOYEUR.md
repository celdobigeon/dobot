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

> Le piège commun à toutes : **rien ne proteste**. Le programme s'exécute,
> le tapis tourne, le bras bouge. Seule la pièce non attrapée le dit, et
> elle ne dit pas pourquoi.

---

## 0 bis. Vérifier que le robot sait où est son bras

**À faire avant tout le reste**, et c'est nouveau : le capteur d'angle
**J3** de cette machine est fautif.

```bash
python diag_capteurs_angle.py COM6
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

## 1. Dégager le bras

Si le voyant est **rouge**, ou si un cycle a été interrompu, le bras est
probablement contre une butée.

Bouton de la fourche maintenu, amenez-le à une posture franche : **bras
arrière vertical, avant-bras horizontal**. Relâchez.

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
python etalonner_convoyeur.py COM6
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
python diag_vise_juste.py COM6
```

Il monte 70 mm au-dessus du point appris et mesure l'écart à la consigne.
S'il annonce 0 mm, le bras obéit : c'est la **cible** qui est fausse, donc
l'étalonnage. S'il annonce plusieurs millimètres, c'est la chaîne de
commande, et le problème est ailleurs que dans le convoyeur.

**Mesurez la dérive du repère** si le doute persiste :

```bash
python diag_derive_origine.py --poser COM6     # enregistre un point fixe
python diag_derive_origine.py COM6             # y revient
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
| `diag_capteur.py` | sur quel port le capteur répond (GP ou EIO) |
| `diag_capteur_vivant.py` | le capteur se déclenche-t-il seul, et où bascule-t-il |
| `diag_convoyeur.py` | le moteur du tapis reçoit-il ses commandes |
| `diag_depassement.py` | la pièce dépasse-t-elle le capteur, et selon quelle loi |
| `diag_file_tapis.py` | un arrêt immédiat passe-t-il quand la file tourne |
| `diag_arret.py` | le capteur, l'arrêt et le glissement, séparés |

---

## 7. Deux particularités de cette cellule

**Le capteur est un infrarouge sur GP4.** La file du contrôleur ne sait pas
attendre sur un port GP : le cycle est donc **mené depuis le PC**, morceau
par morceau, et le robot reste attaché à l'ordinateur pendant tout le
cycle. Un capteur tout-ou-rien sur une entrée EIO rendrait la machine
autonome — c'est un choix d'implantation, pas un détail de câblage.

**Le pont USB est un CP210x.** La carte reste muette un instant après la
mise sous tension : la première connexion échoue parfois en « aucune
réponse en 1,0 s ». Relancez, la seconde passe.

---

## 8. Ce qui reste ouvert

**La dérive du repère entre deux prises d'origine** reste à mesurer
proprement. Celle qu'on a observée venait du guidage manuel, pas de
l'origine elle-même : il faudra refaire la mesure en ne déplaçant le bras
qu'au clavier, et c'est seulement alors qu'on saura si l'origine est
répétable.

**L'axe linéaire** n'est pas monté. Sa direction, `+y`, est une hypothèse
déclarée comme telle dans le code, avec la procédure pour la mesurer.
