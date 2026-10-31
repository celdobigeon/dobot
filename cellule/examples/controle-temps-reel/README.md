# Commande temps réel du Dobot

Avec `Pilote`, on **décrit** un programme, puis on le joue d'un bloc. Avec
`Commande`, **chaque ligne bouge le robot tout de suite**, et ne rend la main
qu'une fois le mouvement **fini**. On peut donc lire la pose juste après,
réagir à une touche, à un capteur — demain à une caméra.

```python
from dobot_sim import Commande, cible

with Commande(simulateur=True, robot=False, vitesse=40) as robot:
    robot.movj(cible(220, 0, 80))       # le script attend ici l'arrivée
    print(robot.pose())                 # (x, y, z, r), lue sur la machine
```

Les scripts de ce dossier présentent les fonctionnalités **une par une**.
Lisez-les dans l'ordre : ils sont courts et commentés. Aucun ne bouge le
robot sans que vous ayez tapé quelque chose au clavier.

## Choisir ce qui bouge

Chaque script choisit sa cible par la constante `CIBLE`, en tête du
fichier, ou par un argument :

```
python examples/controle-temps-reel/02_mouvements_simples.py              simulateur
python examples/controle-temps-reel/02_mouvements_simples.py robot        robot seul
python examples/controle-temps-reel/02_mouvements_simples.py les-deux     les deux
```

| cible | ce qu'il faut | ce qui se passe |
|---|---|---|
| `simulateur` | le simulateur **ouvert** (`python main.py`) | la fenêtre anime chaque commande |
| `robot` | le Dobot branché, simulateur **fermé** | le script ouvre lui-même le port série |
| `les-deux` | le simulateur ouvert (ou `lancer_simulateur=True`) | la fenêtre tient la liaison, exécute sur le robot, et le modèle 3D suit le vrai bras |

Un port série ne s'ouvre qu'une fois : en mode `robot`, le simulateur doit
être fermé ; en mode `les-deux`, c'est lui qui parle au robot.

## Les scripts

| | fonctionnalité |
|---|---|
| `01_choisir_la_cible.py` | choisir la cible, lire `pose()` et `angles()` — **aucun mouvement** |
| `02_mouvements_simples.py` | `movj`, `movl`, `repli` : chaque appel attend la fin |
| `03_cible_au_clavier.py` | taper « x y z » : le robot y va ; zone de travail ; refus attrapé sans quitter |
| `04_pas_a_pas_clavier.py` | avancer de ± N mm en x, y, z au clavier ; régler le pas ; `pas_max` |
| `05_ventouse_et_es.py` | ventouse, sorties et entrées tout-ou-rien |
| `06_plusieurs_cibles.py` | plusieurs cibles avec leurs durées ; `continu=True` contre `continu=False` |
| `07_arc_et_lissage.py` | arcs de cercle (`movc`) et coins arrondis (`lissage`) |
| `08_simulateur_et_robot.py` | simulateur **et** robot ; le script lance le simulateur lui-même |
| `09_garde_fous.py` | zone, `pas_max`, refus, Ctrl+C |

## Les garde-fous

Une commande part tout de suite : il n'y a pas de programme à relire avant.
Avant **chaque** mouvement, depuis la posture **lue** sur la machine :

- la cible est-elle **atteignable** (MGI), **dans les butées**, et le chemin
  entier évite-t-il la table, l'embase et le bras lui-même ?
- reste-t-on dans la **zone** autorisée, `zone=((xmin, xmax), (ymin, ymax), (zmin, zmax))` ?
- la cible est-elle à moins de `pas_max` mm de la précédente ?

Un refus affiche une **alerte rouge** et lève `ErreurExecution` : rien n'est
parti. Pour continuer malgré tout, attrapez-la :

```python
try:
    robot.movl(cible(x, y, z))
except ErreurExecution:
    print("refusé, on continue")
```

Un **avertissement jaune** (durée impossible à tenir, vitesse excessive)
laisse partir le mouvement : le robot fait au mieux.

Robot en défaut — c'est le cas après chaque mise sous tension : l'assistant
de **mise en route** s'ouvre, et rien ne part tant qu'un défaut reste.

**Ctrl+C** pendant un mouvement : **arrêt d'urgence**, puis le script s'arrête.

## Plusieurs cibles, sans s'arrêter

```python
robot.movl([(A, 1.5), (B, 0.8), C])      # (cible, durée en s) ou cible seule
robot.movc([(via1, fin1, 2.0), (via2, fin2)])
robot.movl([A, B, C], lissage=60)        # coins arrondis
robot.movl([A, B, C], continu=False)     # arrêt à chaque cible
```

Par défaut (`continu=True`), le robot enchaîne les cibles **sans s'arrêter**,
grâce au mode « CP » du contrôleur. Ce mode a été mesuré sur la machine :

- chaque portion reçoit sa vitesse, calculée pour tenir sa durée ;
- l'accélération limite ce qui est faisable ; dans une courbe serrée ou un
  angle vif, le robot ralentit ;
- l'outil ne tourne pas en mode continu : un segment qui change `r` est fait
  en point à point, et le script l'annonce.

Avant de partir, la commande **prévoit** la durée de chaque portion. Si une
durée demandée ne peut pas être tenue, un avertissement jaune donne la durée
réaliste — et c'est elle que joue le simulateur.

## Sur le vrai robot : précautions

- Pour un premier essai, dans cet ordre : `01` en mode `les-deux` (aucun
  mouvement : vérifiez que la pose lue est juste et que le modèle 3D se
  superpose au bras), puis `02` en mode `les-deux`, zone dégagée : vous voyez
  dans la fenêtre ce que fait le bras. **Seulement ensuite** `06` et `07`, qui
  emploient le mode continu, moins éprouvé que le point à point.
- Gardez la main près du bouton d'arrêt du robot, et le terminal au premier
  plan pour Ctrl+C.
- Les positions des exemples sont prévues pour la **ventouse** (62 mm). Avec
  un autre outil, vérifiez-les dans le simulateur d'abord.
- Vérifiez ce qui est branché sur les ports EIO avant `05`.
- La pompe à air : relâcher la ventouse ou ouvrir la pince la fait souffler,
  puis tourner. `pince(False)` la coupe d'elle-même 1 s après l'ouverture,
  sans bloquer le script ; `ventouse(False)` la laisse tourner, sauf avec
  `couper=True`. `couper_pompe()` l'arrête tout de suite — une pièce tenue
  tombe. C'est l'outil déclaré (`outil="pince"`) qui décide de la commande
  envoyée, pas le nom de la méthode.
