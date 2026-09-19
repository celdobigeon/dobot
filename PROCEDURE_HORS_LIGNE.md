# Programmation hors ligne du Dobot Magician

Procédure d'atelier — université de Bordeaux
Version du 18/09/2026 · à lire avec le [manuel](README.md), sections 8 et 10

---

## 0. Le principe, et ce qu'il suppose

Programmer **hors ligne**, c'est écrire et valider le cycle sur le modèle
géométrique, machine éteinte ou débranchée, puis ne descendre sur le robot
qu'une fois le programme sûr. Le robot n'est plus un outil de mise au point :
il est le dernier maillon.

Cela ne fonctionne que si le modèle décrit **cette** machine-là. Toute la
partie 2 ci-dessous ne sert qu'à cela : établir les trois grandeurs qui
séparent le modèle de la cellule réelle.

| Ce qui doit correspondre | Où c'est réglé | Erreur si on l'oublie |
|---|---|---|
| position des montages sur la table | `repere_travail(...)` dans le script | le robot va à côté |
| longueur réelle de l'outil | `robot.regler_longueur_outil(...)` | erreur **constante** sur z |
| zéro des axes de la machine | prise d'origine ou recalage | erreur **variable** selon la posture |

> Un cycle validé hors ligne reste une hypothèse tant qu'il n'a pas tourné
> une fois à vitesse réduite, main sur l'arrêt d'urgence. La programmation
> hors ligne supprime les tâtonnements, pas la première exécution prudente.

---

## 1. Préparer le poste (sans robot)

1. Vérifier l'installation :

```bash
python main.py --verifier
```

   Affiche les versions, charge le modèle 3D et contrôle la cohérence
   MGD/MGI sur 2000 configurations tirées au hasard. Tout doit passer avant
   d'écrire une ligne.

2. Ouvrir le simulateur seul si l'on veut la fenêtre tout de suite :

```bash
python main.py
```

3. Dans VS Code, les quatre configurations de `.vscode/launch.json` sont
   prêtes : *Simuler le fichier ouvert*, *Simuler mon_programme.py*,
   *Ouvrir le simulateur*, *Diagnostic de l'installation*.

---

## 2. Relever la cellule

À faire **une fois par montage**, et à refaire dès qu'on déplace un magasin.

### 2.1 Les repères de travail

Mesurer, sur la table, la position de chaque montage par rapport à la base du
robot, puis l'écrire en tête du script — une ligne par montage :

```python
MAGASIN = repere_travail(x=215, y=-95, z=25)
PALETTE = repere_travail(x=205, y=75, z=25, rz=20)
```

Toutes les cibles s'expriment ensuite **dans** ces repères. Déplacer
physiquement la palette ne demandera alors de corriger qu'une seule ligne —
c'est tout l'intérêt de la méthode.

### 2.2 La longueur réelle de l'outil

Les longueurs du catalogue (ventouse 62 mm, pince 78 mm, stylo 70, laser 56)
sont **nominales** ; un montage particulier s'en écarte, et l'écart se
retrouve entièrement sur z.

1. amener la **pointe au contact** du plan de travail ;
2. lire le `z` annoncé par le modèle ;
3. l'ajouter à la longueur courante ;
4. recommencer en **deux ou trois endroits** bien séparés de la table.

Même écart partout → c'est l'outil, on l'inscrit dans le script, après le
constructeur :

```python
robot = Pilote(nom="cycle", outil="ventouse")
robot.regler_longueur_outil(79.3)     # APRES monter_outil, qui remet 62
```

Écart variable → ce n'est pas l'outil, c'est le zéro articulaire : passer au
§ 5 avant d'aller plus loin.

> Sur la ventouse du labo, la valeur mesurée est **79,3 mm** et non 62.

### 2.3 Affectation des entrées / sorties

Noter quel port EIO porte quoi (voyant, vérin, capteur de présence) et le
déclarer en commentaire dans le script. Les sorties sont des ordres, les
entrées des informations venues de la cellule.

---

## 3. Écrire le programme

Fichier de travail : [`mon_programme.py`](mon_programme.py). L'ordre de
rédaction est celui de ses sections, et il n'est pas arbitraire — on va du
plus stable au plus volatil.

| § | Contenu |
|---|---|
| 1 | repères de travail |
| 2 | cibles, exprimées dans ces repères |
| 3 | réglages : vitesses, accélérations, lissage, outil, E/S |
| 4 | contrôle des cibles |
| 5 | mouvement |
| 6 | contrôle du mouvement |

Une cible n'est pas un triplet de nombres mais la **situation** d'un repère :

| Fonction | Construit | Axe z |
|---|---|---|
| `repere_travail(x, y, z, rz)` | `Trans · Rz` | vers le **haut** — un plan de travail |
| `cible(x, y, z, rz, repere=…)` | `Trans · Rz · Rx(180°)` | vers le **bas** — l'approche de l'outil |

Trois règles de rédaction qui évitent la plupart des erreurs :

- **post-multiplier, c'est se déplacer dans le repère de l'OUTIL** ; son axe
  z pointant vers le bas, un dégagement vers le haut s'écrit
  `hg.translation(0, 0, -85)` ;
- pour fixer une **orientation**, utiliser l'argument `rz=` et non une
  composition : `cible(…, rz=20) @ rot_z(30°)` donne r = −10°, pas +50° ;
- les coordonnées omises sont conservées : `robot.movl(z=25)` descend à la
  verticale sans répéter x et y.

Enfin, l'approche et le dégagement se font toujours **à la verticale de la
cible**, en `MOVL`, et à vitesse réduite pour la descente. Le transfert entre
deux postes se fait en `MOVJ` ou `MOVC`, plus haut.

---

## 4. Valider hors ligne

C'est le cœur de la méthode. Rien ne doit partir sur la machine avant que
ces quatre contrôles soient au vert.

### 4.1 Les cibles, avant tout mouvement

La section 4 de `mon_programme.py` vérifie chaque cible sur quatre points :
matrice homogène valide, orientation réalisable, atteignabilité dans les
butées, et surtout `MGD(MGI(T)) = T`.

```
PRISE                  x=  215.0 y=  -95.0 z=  25.0 r=   0.0   ok  (ecart MGD/MGI 6e-14)
case(1)                x=  196.8 y=   97.6 z=  25.0 r=  20.0   ok  (ecart MGD/MGI 9e-14)
```

L'écart doit tomber vers 10⁻¹⁴ mm. Une erreur de repère se voit **ici**, pas
en regardant le robot partir de travers.

### 4.2 Le résumé du programme

```python
print(robot.resume())      # durées calculées, avertissements, butées
```

N'ouvre aucune fenêtre. Lire les avertissements un par un : butée frôlée,
point hors d'atteinte, attente qui neutralise le lissage de ses voisins.

### 4.3 Le mouvement échantillonné

La section 6 mesure la distance minimale du TCP à chaque cible **sur la
trajectoire** :

```
03 MOVL  descente lente                  0.00 mm   atteinte
06 MOVL  remontee                        1.28 mm   coin coupe, lissage 80 %
```

Les cibles sans lissage doivent être atteintes à **0,00 mm**. Celles qui en
portent sont volontairement manquées de 1 à 2 mm : c'est le lissage, et
c'est voulu — mais il faut que ce soit voulu **là**, et pas ailleurs.

### 4.4 L'animation

`F5` dans VS Code. Si un simulateur est déjà ouvert, le programme lui est
envoyé et la fenêtre remonte au premier plan avec le nouveau cycle chargé.
Regarder le cycle en entier au moins une fois, en 3D, avec l'enveloppe de
travail affichée.

**Critère de sortie de cette étape :** aucun avertissement de butée, aucune
collision, aucune cible hors d'atteinte, cycle visuellement conforme.

---

## 5. Étalonner la machine

Machine sous tension, **avant** tout téléversement. Les deux robots du labo
ne demandent pas la même chose.

| | robot n° 1 | robot n° 2 |
|---|---|---|
| pont USB | CH340 | Silicon Labs CP210x |
| port usuel | COM4 | COM6 |
| capteurs | **J2 fautif** (erreur 19,66 · cos J2) | sains |
| mise en référence | **recalage manuel**, à chaque séance | prise d'origine normale |

### 5.1 Diagnostic, qui ne fait rien bouger

```bash
python calibrer_robot.py --capteurs
```

Affiche firmware, alarmes, pose dans les deux repères, point d'origine
enregistré et écart de cohérence. **N'écrit rien, ne bouge pas.**

### 5.2 Robot n° 2 (COM6) — prise d'origine

```bash
python calibrer_robot.py --port COM6 --origine
```

**Le robot bouge.** Dégager la zone. À refaire après chaque mise sous
tension et dès que le bras a été déplacé à la main.

### 5.3 Robot n° 1 (COM4) — recalage sans prise d'origine

Sa prise d'origine part en butée : elle s'appuie sur le capteur fautif. On
contourne, en profitant de ce que le contrôleur **garde la posture qu'on lui
déclare** et compte ensuite ses pas.

1. bouton de la fourche enfoncé, placer le bras à la main : **bras arrière
   vertical, avant-bras horizontal** ;
2. relâcher, puis déclarer :

```bash
python calibrer_robot.py --recaler
```

   ou, dans l'interface : panneau *Robot réel* → **Déclarer la posture
   d'équerre**. Rien ne bouge.

> À refaire **à chaque mise sous tension**. La précision devient celle du
> placement : cinq mises à la verticale se sont étalées sur 3,5°. Pour de la
> répétabilité, pousser doucement le bras contre sa **butée arrière** plutôt
> que viser la verticale à l'œil.

### 5.4 Vérifier le zéro — le seul test décisif

Cocher le **miroir** dans le panneau *Robot réel*, mettre le bras réel dans
une posture reconnaissable et comparer au modèle 3D. S'ils diffèrent, les
zéros ont dérivé et rien ne doit être lancé.

> `coherence()` ne suffit pas : il compare la pose et les angles renvoyés par
> la machine, qui viennent des **mêmes** compteurs — ils restent d'accord
> entre eux même quand le zéro est faux.

---

## 6. Transférer et exécuter

Deux voies. La première est celle du TP.

### 6.1 Depuis le simulateur — panneau *Robot réel*

1. **Connecter** sur le bon port. La connexion est en lecture seule : rien
   ne bouge.
2. Régler le **ratio de vitesse** à **30 %** pour la première exécution.
3. **Téléverser le programme** — remplit la file **sans rien lancer**. Un
   programme dont le contrôle signale une butée ou une collision est
   **refusé** ici : c'est le dernier filet.
4. Dégager la cellule. Main sur l'arrêt d'urgence (bouton rouge, ou
   `Ctrl+Maj+Espace`).
5. **Lancer.** Là, le robot bouge.

> **La file se consomme.** Chaque commande exécutée disparaît ; après un
> cycle il n'en reste rien. *Lancer* la remplit de lui-même quand elle est
> épuisée : on peut donc enchaîner les cycles sans retéléverser.

En script, la même séquence :

```python
from dobot_sim.reel import PontReel

pont = PontReel(robot, journal=print)
pont.connecter("COM6")           # lecture seule
pont.definir_ratio(30)
pont.televerser(programme, q0)   # remplit la file
pont.demarrer()                  # LA, le robot bouge
```

### 6.2 Export vers DobotStudio

*Fichier ▸ Exporter un script Dobot* produit un script `DobotDllType`
utilisable sans le simulateur. Utile pour archiver un cycle ou le confier à
un poste équipé de DobotStudio.

```python
robot.exporter_python("cycle_reel.py")   # même chose depuis un script
robot.enregistrer("cycle.json")          # programme éditable, relisable
```

> À l'export, les cibles du repère objet sont converties en coordonnées
> **absolues** — le contrôleur ne connaît pas cette notion — et un
> avertissement est inséré si un repère outil déporté est en place, le
> contrôleur ne gérant qu'un déport vertical.

---

## 7. Première exécution, puis montée en vitesse

1. cycle complet à **30 %**, **à vide** (sans pièce, sans aspiration) ;
2. contrôler visuellement les approches : la descente s'arrête-t-elle au bon
   niveau ? Si elle est trop haute ou trop basse d'une valeur **constante**,
   c'est la longueur d'outil (§ 2.2) ; si l'écart **change selon le poste**,
   c'est le zéro (§ 5) ;
3. cycle avec pièce, toujours à 30 % ;
4. remonter la vitesse par paliers — 50 %, puis 70 % — en surveillant les
   dépassements dans les virages lissés.

Toute correction se fait **dans le script**, puis on repasse par le § 4.
On ne retouche jamais un point sur la machine sans le répercuter dans le
programme : ce serait perdre le bénéfice du hors-ligne.

---

## 8. Fin de séance

1. arrêt d'urgence ou fin de cycle propre ;
2. fermer la fenêtre du simulateur — la fermeture déclenche l'arrêt d'urgence
   avant de fermer le port ;
3. mettre la machine hors tension ;
4. enregistrer le programme (`cycle.json`) et le script dans le dossier du
   binôme.

> Le recalage du robot n° 1 est **perdu à la mise hors tension**. Le § 5.3
> est à refaire au début de la séance suivante.

---

## Dépannage

| Symptôme | Cause probable | Que faire |
|---|---|---|
| le bras réel n'est pas là où le modèle le montre | zéro dérivé | § 5.2 ou § 5.3 |
| la prise d'origine part en butée | capteur J2 fautif (robot n° 1) | § 5.3, ne pas insister |
| descente trop basse / trop haute, **partout pareil** | longueur d'outil | § 2.2 |
| écart qui **change selon le poste** | zéro articulaire | § 5 |
| le programme s'arrête en cours, barre plantée | alarme : la file est gelée | *Effacer les défauts*, puis retéléverser |
| *Lancer* ne fait rien | file vide, cycle déjà consommé | recliquer *Lancer* : il la remplit |
| le port refuse de s'ouvrir | une autre fenêtre le tient | fermer l'autre simulateur ou DobotStudio |
| plusieurs connexions d'affilée échouent (COM6) | pont CP210x rouvert trop vite | attendre quelques secondes entre deux essais |
| cible refusée « axe d'outil non vertical » | orientation inclinée demandée | contrainte du parallélogramme : corriger la cible |

Pour situer l'instruction fautive à partir de la barre de progression : la
file compte **deux commandes par mouvement** (les paramètres de vitesse, puis
le PTP). Un programme de 7 mouvements fait 14 entrées, soit ≈ 14 % par
instruction.

---

## Fiche récapitulative — à cocher

**Hors ligne, sans robot**

- [ ] `python main.py --verifier` passe
- [ ] repères de travail mesurés et écrits en tête du script
- [ ] longueur d'outil mesurée, écart vérifié **constant** en 2–3 endroits
- [ ] contrôle des cibles : tout `ok`, écart MGD/MGI ≈ 10⁻¹⁴
- [ ] `robot.resume()` : aucun avertissement non expliqué
- [ ] contrôle du mouvement : cibles sans lissage atteintes à 0,00 mm
- [ ] cycle regardé en entier dans la vue 3D

**Sur machine**

- [ ] bon port identifié (COM4 = CH340, COM6 = CP210x)
- [ ] `--capteurs` : firmware lu, aucune alarme
- [ ] mise en référence faite (prise d'origine, ou posture d'équerre déclarée)
- [ ] miroir 3D : la posture affichée correspond à la vraie
- [ ] ratio de vitesse à 30 %
- [ ] cellule dégagée, arrêt d'urgence à portée de main
- [ ] téléversement accepté (aucune butée, aucune collision)
- [ ] premier cycle à vide, à 30 %
