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

> Le piège commun à toutes : **rien ne proteste**. Le programme s'exécute,
> le tapis tourne, le bras bouge. Seule la pièce non attrapée le dit, et
> elle ne dit pas pourquoi.

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
capteur l'arrête, vous amenez la ventouse **au contact du centre** de la
pièce, à la main.

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

**La dérive du repère entre deux prises d'origine** n'a pas été mesurée
contre un point fixe. Si elle se confirme, l'étape 3 devient obligatoire à
chaque mise sous tension — ce qu'elle est déjà par prudence.

**L'axe linéaire** n'est pas monté. Sa direction, `+y`, est une hypothèse
déclarée comme telle dans le code, avec la procédure pour la mesurer.
