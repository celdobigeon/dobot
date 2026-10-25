# Simulateur Dobot Magician

Logiciel de pilotage **virtuel** d'un Dobot Magician à partir de son modèle CAO.
Tout est construit sur les **matrices homogènes 4×4** : modèle géométrique
direct, modèle inverse, jacobienne, et programmation de trajectoires
**articulaires** et **cartésiennes**.

---

## 1. Installation

Python 3.8 ou plus récent.

```powershell
pip install -r requirements.txt
python main.py
```

Contrôle de l'installation, sans interface graphique :

```powershell
python main.py --verifier
```

Cette commande vérifie les dépendances, teste la cohérence MGD/MGI sur
2000 configurations tirées au hasard, compare la jacobienne analytique aux
différences finies, et liste les pièces du modèle 3D effectivement trouvées.

Sous Linux, `./installer_linux.sh` crée l'environnement virtuel `.venv` et
y installe les dépendances, **sans droits administrateur** ; il termine par
`python main.py --verifier`, qui dit aussi si le port du robot est
accessible depuis ce compte. Si ce n'est pas le cas, seul l'administrateur
peut y remédier, une fois par poste : `sudo ./admin/configurer_poste_linux.sh`
(paquets système, port série ouvert à tous les comptes, retrait de brltty).

> Si `PyOpenGL` est absent ou si la machine n'a pas de pilote OpenGL
> exploitable, le simulateur bascule automatiquement sur deux projections 2D
> (vue méridienne et vue de dessus). Toutes les autres fonctions restent
> identiques. Un vrai contexte OpenGL est essayé au démarrage : s'il ne
> répond pas, la raison est affichée dans le terminal et dans *Aide ▸ A
> propos*. La variable `DOBOT_VUE=2d` (ou `3d`) impose l'une ou l'autre vue.
>
> Sous Linux en session Wayland (Ubuntu par défaut), la fenêtre passe par
> XWayland : le simulateur règle lui-même `QT_QPA_PLATFORM=xcb` et
> `PYOPENGL_PLATFORM=glx`, faute de quoi la vue 3D resterait vide.

---

## 2. Organisation

```
APPS_DOBOT/
├── main.py                    point d'entrée  (--verifier pour le diagnostic)
├── mon_programme.py           VOTRE script : F5 dans VS Code et le robot bouge
├── requirements.txt
├── .vscode/                   configurations de lancement prêtes à l'emploi
├── dobot_sim/
│   ├── liaison.py            protocole série officiel du Magician
│   ├── reel.py               pont simulateur → machine réelle
│   ├── passerelle.py         un script envoie son programme au simulateur ouvert
│   ├── homogeneous.py         algèbre SE(3) : matrices homogènes, DH, SLERP
│   ├── robot.py               MGD, MGI, jacobienne, butées, espace de travail
│   ├── trajectory.py          lois horaires + segments MOVJ / MOVL / MOVC
│   ├── program.py             programme éditable, JSON, export robot réel
│   ├── pilote.py              API de script (movj, movl, movc, simuler…)
│   ├── commande.py            commande temps réel : chaque appel bouge tout de suite
│   ├── alertes.py             alertes du terminal, ErreurExecution
│   ├── outils.py              catalogue d'outils : ventouse, pince, stylo, laser
│   ├── es.py                  entrées / sorties tout-ou-rien (20 ports EIO)
│   ├── urdf.py                lecture de la description ROS officielle + STL
│   ├── meshes.py              chargement des maillages, OBJ de secours
│   ├── model_config.json      choix de la source 3D et couleurs
│   ├── viewer3d.py            vue OpenGL
│   ├── viewer2d.py            vue de secours en projections
│   └── ui.py                  interface graphique
├── fiches/
│   ├── fiche_interface.html   ce que fait chaque commande de la fenêtre
│   ├── fiche_programmation.html  exemples de programmation, support de TP
│   └── fiche_api.html         référence : définition de chaque commande
├── examples/
│   ├── exemple_matrices.py    matrices homogènes, MGD, MGI, jacobienne
│   ├── exemple_pick_place.py  trajectoires et export
│   └── controle-temps-reel/   Commande, pas à pas et au clavier (01 à 09)
├── tests/
│   ├── test_kinematics.py     modèle, trajectoires, outils, repères
│   ├── test_protocole.py      protocole série et pont vers la machine réelle
│   └── test_commande.py       commande temps réel, robot et fenêtre simulés
├── Modele3D/                  OBJ d'origine (source de secours)
└── Magician & Accessories-2D&3D files/
                               dossier constructeur : URDF officiel, table DH,
                               cotes d'effecteur, STEP des accessoires
```

---

## 3. Modèle géométrique

### Paramètres (mm)

| Symbole | Valeur | Signification | Source |
|---------|--------|---------------|--------|
| `d1` | 139   | hauteur de l'axe d'épaule au-dessus du plan de pose | URDF : 56,9 + 82,1 |
| `l1` | 135   | bras arrière : épaule → coude | URDF **et** table DH |
| `l2` | 147   | avant-bras : coude → poignet | URDF **et** table DH |
| `l3` | 59,7  | déport horizontal poignet → axe de l'outil | table DH (`a₄`) |
| `lt` | 0     | déport vertical de la pointe d'outil | dépend de l'outil monté |

Ces valeurs viennent des documents constructeur du dossier
`Magician & Accessories-2D&3D files`. La table de Denavit-Hartenberg
publiée par Dobot y figure ; elle est reprise dans `robot.DH_CONSTRUCTEUR` :

| L | θᵢ | αᵢ | dᵢ | aᵢ |
|---|----|----|----|----|
| 1 | θ₁ | −π/2 | 8 | 0 |
| 2 | θ₂ | 0 | 0 | 135 |
| 3 | θ₃ | 0 | 0 | 147 |
| 4 | θ₄ | −π/2 | 0 | 59,7 |

> Elle décrit le **bras seul** : la hauteur du socle n'y figure pas, et son
> repère 0 n'est pas le nôtre (d'où le `d₁ = 8`). Nous gardons la chaîne
> explicite ci-dessous, qui part du plan de pose — mais les longueurs, elles,
> sont bien celles de cette table.

### Variables articulaires

| Axe | Plage | Convention |
|-----|-------|------------|
| J1 | −135° … +135° | rotation de la colonne autour de la verticale |
| J2 | 0° … 85° | bras arrière, **0° = vertical**, positif vers l'avant |
| J3 | −10° … 95° | avant-bras, **0° = horizontal**, positif vers le bas |
| J4 | −90° … +90° | rotation de l'outil autour de la verticale |

### Chaîne de transformations

```
T01 = Rz(q1)                        base       -> colonne
T12 = Tz(d1) . Ry(q2)               colonne    -> bras arrière
T23 = Tz(l1) . Ry(q3 - q2 + 90°)    bras arr.  -> avant-bras
T34 = Tz(l2) . Ry(-90° - q3)        avant-bras -> poignet
T45 = Tx(l3) . Rz(q4)               poignet    -> axe outil
T5T = repère OUTIL                  axe outil  -> TCP (z dirigé vers le bas)
      par défaut Tz(-lt) . Rx(180°)

T0_TCP = T01 . T12 . T23 . T34 . T45 . T5T
```

La somme des rotations autour de Y vaut exactement zéro :

```
q2 + (q3 - q2 + 90°) + (-90° - q3) = 0
```

C'est la traduction algébrique du **parallélogramme** du Magician : l'axe de
l'outil reste vertical quelles que soient les valeurs de J2 et J3. Le robot
n'a donc que 4 degrés de liberté utiles : `x, y, z` et l'orientation `r`
autour de la verticale, avec `r = J1 + J4` (même convention que l'API Dobot).

### Modèle géométrique inverse

Solution analytique fermée :

```
q1 = atan2(y, x)
rp = hypot(x, y) - l3                zp = z + lt - d1
D  = hypot(rp, zp)
cos β = (l1² + D² - l2²) / (2 l1 D)          loi des cosinus
q2 = recentrer( atan2(rp, zp) ∓ β )
q3 = recentrer( atan2(l1 cos q2 - zp, rp - l1 sin q2) )
q4 = recentrer( r - q1 )
```

**Deux subtilités**, l'une et l'autre vérifiées par les tests.

*Le choix de branche.* Le triangle épaule–coude–poignet admet deux
solutions, selon le côté de la droite épaule–poignet où se place le coude.
En notant `γ = 90° + q3 − q2` la flexion du coude, la solution physique est
celle du signe de `sin γ` : « coude haut » pour `γ ∈ ]0°, 180°[`. Or les
butées du Magician autorisent `γ ∈ [−5°, 185°]`, donc deux minces bandes
d'hyperextension et d'hyperflexion ne sont atteignables qu'avec l'autre
branche. Le mode `coude="auto"` (défaut) prend la solution coude haut et ne
bascule que si elle viole les butées ; aux deux singularités les deux
solutions se confondent, le raccordement est donc continu. `coude="haut"`
ou `"bas"` force une branche.

*Le recentrage des angles.* Bras complètement replié, `β` vaut 180° et
`atan2(rp, zp) − β` sort de `]−180°, 180°]` : sans recentrage le MGI renvoie
par exemple −356° au lieu de +4°.

### Plage utile en z

Le point de référence (`lt = 0`) est au niveau de l'**axe du poignet**, pas
à la pointe de l'outil. Avec les butées ci-dessus, il descend au plus bas
vers z ≈ 3 mm, et seulement autour de ρ ≈ 200 mm :

| rayon ρ | 150 | 180 | 200 | 240 | 280 | 300 |
|---------|-----|-----|-----|-----|-----|-----|
| z min   | 80  | 16  | 4   | 12  | 32  | 49  |
| z max   | 158 | 190 | 230 | 295 | 276 | 260 |

Renseignez `Geometrie.lt` (ou `robot.avec_outil(65)`) avec la longueur de
votre ventouse ou de votre stylo pour que `z` désigne la pointe : c'est
ainsi que l'on atteint la table.

### Jacobienne

`[vx vy vz ωz]ᵀ = J(q) · [q̇1 q̇2 q̇3 q̇4]ᵀ`, forme analytique validée par
différences finies. `manipulabilite()` renvoie la mesure de Yoshikawa
`√det(J Jᵀ)`, qui s'effondre au voisinage des singularités (TCP sur l'axe
vertical, bras tendu ou complètement replié).

---

## 4. Trajectoires

Deux notions **indépendantes**, combinables librement.

**La loi horaire** donne l'abscisse curviligne normalisée `s(t) ∈ [0,1]` :

| Nom | Comportement |
|-----|--------------|
| `lineaire` | vitesse constante, échelons de vitesse aux extrémités |
| `trapeze` | accélération constante, palier, freinage |
| `polynome5` | `10u³ − 15u⁴ + 6u⁵`, vitesse **et** accélération nulles aux bouts |
| `sinus` | loi en cosinus surélevé, accélération continue |

**Le segment** donne la géométrie du chemin :

| Type | Espace interpolé | Chemin du TCP |
|------|------------------|---------------|
| `MOVJ` | angles articulaires | courbe quelconque, jamais singulier |
| `MOVL` | matrice homogène (position linéaire + SLERP) puis MGI | **droite** |
| `MOVC` | arc de cercle par 3 points, orientation en SLERP | **arc** |

### Vitesse, accélération, lissage

Chaque instruction se cadence de deux façons.

**Durée imposée** — on donne le temps, la vitesse en découle.

**Vitesse + accélération** — on donne `v_max` et `a_max`, la durée se
**déduit** de la longueur du chemin. C'est le mode par défaut. Chaque loi
horaire connaît le maximum de `s'` et de `s''` sur une course unitaire, d'où

```
T ≥ K_V · L / v_max        et        T ≥ √( K_A · L / a_max )
```

| Loi | `K_V` | `K_A` | durée pour L = 200 mm à 100 mm/s |
|-----|-------|-------|----------------------------------|
| `lineaire` | 1 | — | 2,00 s |
| `trapeze` | *adapté* | *adapté* | 2,25 s |
| `sinus` | π/2 ≈ 1,571 | π²/2 ≈ 4,935 | 3,14 s |
| `polynome5` | 15/8 = 1,875 | 10/√3 ≈ 5,774 | 3,75 s |

Le trapèze est traité à part : sa durée `L/v + v/a` et sa part
d'accélération sont calculées pour saturer **exactement** les deux
contraintes (il bascule tout seul en triangle si `L < v²/a`).

> **Attention aux unités.** Un `MOVJ` interpole des *angles* : sa vitesse est
> en deg/s, et elle **ne borne pas** la vitesse du TCP, qui dépend du bras de
> levier. Un `MOVL` ou un `MOVC` interpole des *millimètres* : là, la consigne
> est bien la vitesse de l'outil. C'est pourquoi `Pilote` a deux jeux de
> réglages séparés.

**Lissage** — l'arrondi du raccord avec l'instruction suivante, de 0 à 100 %.
Deux segments consécutifs se **recouvrent** pendant une durée `tb`, et sur
cette fenêtre la consigne est le mélange pondéré des deux :

```
q(t) = (1 − w)·q_i(t) + w·q_i+1(t),     w quintique, de 0 à 1
```

`w` ayant une dérivée nulle aux deux bords, la vitesse reste continue : le
robot **coupe le coin** au lieu de s'arrêter. C'est le principe des *zones*
d'ABB ou du *CNT* de Fanuc. Sur un angle droit à 120 mm/s :

| lissage | 0 % | 20 % | 40 % | 60 % | 80 % | 100 % |
|---------|-----|------|------|------|------|-------|
| coin coupé (trapèze) | 0,0 mm | 0,4 | 1,7 | 3,7 | 6,5 | 10,0 mm |
| durée | 2,70 s | 2,61 | 2,51 | 2,41 | 2,32 | 2,22 s |

Le point de passage n'est donc plus atteint exactement : c'est le prix du
lissage, et la durée totale diminue d'autant. Une `ATTENTE` n'est jamais
absorbée par un lissage. Mesuré sur le cycle de `mon_programme.py` : **35,1 s
→ 28,8 s**, sans que la vitesse maximale augmente d'un seul mm/s.

> **Le lissage n'existe que dans le simulateur.** Sur le robot réel, chaque
> `movl` est une commande point à point qui s'arrête à sa cible. Pour un
> mouvement sans arrêt **aussi sur la machine**, utilisez `trajet`.

**Trajet continu** — `robot.trajet([P1, P2, P3], vitesse=80)` relie les
points par des segments de droite, **sans s'arrêter** aux points
intermédiaires, et passe exactement par chacun. Sur le robot réel, c'est le
mode continu (CP) du contrôleur. Le simulateur en reproduit les vitesses :
départ et arrivée à l'arrêt, croisière à `vitesse`, et ralentissement dans
les coins, d'autant plus fort qu'ils sont vifs. Sur un carré de 60 mm à
80 mm/s, le robot passe chaque angle droit à environ 27 mm/s.

```python
robot.movj(cible(200, 0, 50))                 # orienter l'outil AVANT
robot.trajet([cible(200, 60, 50), (260, 60, 50),
              (260, 0, 50), (200, 0, 50)], vitesse=80)
```

Trois limites, celles du mode CP de la machine :
- l'outil ne tourne pas pendant un trajet : un point qui change `r` est
  refusé ;
- la vitesse ne dépasse pas 125 mm/s, la plus haute essayée sur le robot ;
- `duree=` et `profil=` ne s'appliquent pas.

Dans le programme, un trajet devient une suite de `MOVL` marqués « continu
-> » ; la case *enchaîner sans arrêt* de l'éditeur fait la même chose.

`Trajectoire.controler()` échantillonne le mouvement complet et signale les
dépassements de butées, les points hors d'atteinte, les **collisions** et
les vitesses excessives.

### Butées frôlées

Une pose peut être **dans** les butées et refusée par la machine. Le modèle
travaille sur la table nominale ; les butées d'un exemplaire réel n'y sont
jamais exactement, et une prise d'origine approximative suffit à consommer
les quelques degrés qui restent.

`butees_frolees(q)` signale donc les articulations à moins de **5°** de leur
butée — seuil réglable par `Limites.marge_alerte`, mis à 0 pour le
désactiver. C'est un avertissement **ambre**, pas un refus : la pose reste
réalisable pour le modèle.

```python
robot.marges(q)            # distance de chaque axe a sa butee, en degres
robot.butees_frolees(q)    # ceux qui passent sous le seuil
```

> **Le cas qui a motivé ce contrôle.** La cible `(175, 0, 0)` avec la
> ventouse exige `J3 = 90,05°` pour une butée à 95 : **4,95° de marge**. Le
> simulateur l'acceptait sans un mot ; le vrai contrôleur l'a refusée et
> s'est mis en défaut au milieu du programme. Deux autres pièges du même
> genre : une cible sur l'axe **+y** avec `r = 0` impose `J4 = −90°`,
> exactement sur la butée, et une cible sur **−y** impose `J4 = +90°`.

### Collisions

Le bras est confronté à trois volumes, dont les dimensions sont **mesurées
sur les maillages du modèle CAO officiel** (boîtes englobantes des pièces de
`magician_description`) :

| volume | dimensions | d'où elles viennent |
|--------|-----------|---------------------|
| **plan de pose** | z = 0 | le plan sur lequel l'embase repose |
| **embase** | boîte 158 × 158 mm, haute de 58 mm | `magician_base` : x, y ∈ [−79 ; 79], z ∈ [0 ; 57,9] |
| **colonne** | cylindre r = 69,5 mm, haut de 176 mm | `magician_link1` : r ≤ 69,5, z ∈ [57 ; 176,2] |

```python
robot.collisions(q)          # liste de messages, vide si la pose est saine
robot.collisions(q, marge=10)  # alerte 10 mm AVANT le contact
robot.defauts(q)             # butees + collisions, dans cet ordre
```

### Le bras contre lui-même

Chaque maillon est traité comme une **capsule** — un segment entouré d'un
rayon — et les paires **non adjacentes** sont confrontées deux à deux. Les
paires adjacentes se touchent par construction à leur articulation commune,
les tester n'aurait aucun sens.

Les rayons viennent de la demi-largeur en y des maillages CAO : bras arrière
22 mm, avant-bras 15, déport du poignet 12, outil 15.

> **Le cas qui compte sur ce robot.** Quand l'avant-bras se replie, le
> **déport du poignet revient sur le bras arrière** : contact à
> `J3 − J2 = 77,03°`, jusqu'à 22,6 mm d'interpénétration à fond de course.
> Le jeu ne dépend **que** de cette différence — étendue mesurée 0,000 mm
> sur toute la plage de J2 —, conséquence directe du parallélogramme.
>
> C'est la raison d'être de `Limites.couplage`, laissé à `None` faute de
> valeur mesurée. Le contrôle géométrique le rend inutile à régler à la
> main, et il suit l'outil monté.

> **Un simulateur qui laisse passer ce mouvement ment**, puisque la machine
> réelle ne peut pas l'exécuter. Il était détecté nulle part avant : la
> première version du modèle affirmait que les butées suffisaient à l'éviter.
> C'est faux, et le balayage de l'espace articulaire le montre.

Seul le squelette **mobile** est testé — coude, poignet, axe outil, TCP. Le
bras arrière est exclu volontairement : il est solidaire de la colonne, la
butée J2 l'empêche d'atteindre table comme embase, et l'inclure ferait
signaler l'épaule elle-même, qui vit légitimement *à l'intérieur* de la
colonne (l'axe d'épaule est à 139 mm, la colonne monte à 176).

Ces trois interférences sont **réellement atteignables** dans les butées —
vérifié par balayage de l'espace articulaire :

| | atteignable ? |
|---|---|
| plan de pose | seulement **avec un outil** : bride nue, le point le plus bas reste à +3,8 mm ; ventouse −58 mm ; pince −74 mm |
| colonne | oui, à 33 mm de l'axe sous l'épaule |
| embase | oui, jusqu'à 33 mm du centre avec la pince |

> **Ce que ce modèle ne couvre pas.** C'est une enveloppe simple, pas un
> moteur de collision : ni la forme exacte des maillages, ni les servos qui
> dépassent latéralement à l'épaule, ni aucun obstacle extérieur au robot.
> `Encombrement` expose toutes les cotes, modifiables.

---

## 5. L'interface

Un index et quatre fiches HTML autonomes, à ouvrir dans un navigateur et à
garder à côté du simulateur pendant la séance :

| Fiche | Pour | Contenu |
|-------|------|---------|
| [`fiche_index.html`](fiches/fiche_index.html) | aide-mémoire | l'index : ce qu'on écrit, sous forme matricielle, avec un renvoi vers les fiches 2 et 3 |
| [`fiche_interface.html`](fiches/fiche_interface.html) | pendant la séance | chaque panneau, chaque case et chaque bouton de la fenêtre |
| [`fiche_programmation.html`](fiches/fiche_programmation.html) | support de TP | vocabulaire (TCP, pose, MGD, MGI), fabrication d'une cible, deux schémas, les erreurs expliquées, cinq exercices |
| [`fiche_api.html`](fiches/fiche_api.html) | ressource | la définition de chaque ligne de code : signature, paramètres, types, unités, valeur renvoyée, index |
| [`fiche_commande.html`](fiches/fiche_commande.html) | robot réel, scripts interactifs | la commande temps réel (`Commande`) : les trois montages, les neuf exemples commentés, la référence, le mode continu et ses mesures, les garde-fous, ce qui reste à vérifier |

| Zone | Contenu |
|------|---------|
| **Centre** | vue 3D : maillages CAO, repères des liaisons, squelette, chemin théorique, trace réelle du TCP, enveloppe de l'espace de travail |
| **Gauche** | dock **Commande** : quatre sections **visibles en même temps**, chacune repliable par sa case à cocher : **Articulaire** (J1…J4, curseurs bornés aux butées), **Cartésien** (X, Y, Z, Rz, MGI appliqué en continu), **Outil** (ventouse, pince, stylo, laser) et **Repères** (repères outil et objet, lus en X, Y, Z, Rz, saisis en matrice 4×4) ; état des butées, portée, manipulabilité |
| **Droite** | dock **Programme** : barre d'édition en pictogrammes, liste des instructions, formulaire **repliable** (type, cible et point de passage en `(X, Y, Z, Rz)`, bouton *Saisir* pour la matrice 4×4, **cadence**, **lissage**, loi horaire) et lecteur : lire / pause, arrêter, pas à pas, facteur de vitesse, barre de temps |
| **Bas** | onglet **Courbes** (angles et vitesse du TCP en fonction du temps), onglet **Entrées / Sorties** (20 voyants de sortie, 20 boutons d'entrée), onglet **Journal** — et **Matrices homogènes** (les six matrices élémentaires, le produit, la jacobienne, det J), **masqué à l'ouverture**, qu'on appelle par le menu *Fenêtres* |

### Le panneau ne porte plus d'explications

Tout ce qui était écrit en gris dans le panneau **Commande** a quitté
l'écran : l'appel du MGI, le déport `lt` de l'outil, le rôle des deux
repères, la contrainte `Rx = 180° / Ry = 0°`, la description du catalogue.
Rien n'est perdu — chaque texte a une nouvelle adresse, choisie pour qu'on
le lise au moment où il sert :

| Texte retiré | Où il se lit maintenant |
|---|---|
| MGI et espace atteignable | infobulle du bouton *Appliquer le MGI* |
| longueur d'outil → déport `lt` | infobulle de *outil monté* et de *longueur* |
| description de l'outil (« kit ventouse… ») | infobulle de *outil monté*, elle suit l'outil choisi |
| `Rx = 180° / Ry = 0°`, non commandables | infobulle du curseur **Rz** |
| rôle des repères outil et objet | **dans le dialogue** qui sert à les modifier |

Les trois fiches HTML de [`fiches/`](fiches/) portent la version longue de
tous ces textes.

#### Un seul bloc de lecture, dans la section Cartésien

*Où est le bras* et *pourquoi il n'est pas ailleurs* sont une seule question :
les deux réponses tiennent dans le même bloc, placé **sous les curseurs
cartésiens**, à côté des commandes qui les produisent.

Il montre **soit l'un, soit l'autre** :

| Quand | Ce qu'il montre |
|---|---|
| le MGI a échoué | **seulement** le motif du refus : le point refusé, la distance épaule-poignet demandée, la plage admissible, et le rappel que le robot reste à sa position précédente |
| sinon | le TCP `(x, y, z, r)`, la portée horizontale, la manipulabilité, puis *butées dépassées* ou *collision* s'il y a lieu, sinon *configuration admissible* |

Le TCP disparaît pendant un refus parce qu'il décrirait la pose **tenue**, pas
celle dont on parle : les lire l'une sous l'autre ferait cohabiter deux
situations dans le même bloc. Il revient dès que le MGI aboutit.

Le bloc **s'encadre de rouge** dès qu'il a une mauvaise nouvelle à annoncer,
et redevient neutre à la pose suivante. Le verdict vert ne s'affiche jamais
en même temps qu'un défaut — ce serait se contredire.

Auparavant les deux étaient dissociés : la ligne d'état en bas du panneau, le
motif du MGI dans un encadré séparé. Le message écrasait même l'état, si bien
que le TCP disparaissait au moment précis où on cherchait à comprendre où le
bras se trouvait.

### Un code couleur unique

**x rouge, y vert, z bleu**, partout : le trièdre du repère de base dans la
vue 3D (étiqueté, et dessiné par-dessus la scène comme un gizmo de CAO), les
repères des liaisons, ceux des cibles, et les étiquettes des curseurs
cartésiens — **X (avant)**, **Y (gauche)**, **Z (hauteur)**. `Rz (outil)`
est en bleu clair : c'est une rotation *autour de* z, et c'est la seule
orientation commandable. Une note grisée rappelle sous les curseurs que
`Rx = 180°` et `Ry = 0°` sont imposés par le parallélogramme.

Le signalement d'une consigne irréalisable se fait alors par le **fond** de
l'étiquette, pour ne pas écraser la couleur de l'axe.

### Des pictogrammes plutôt que des mots

Les instructions portent leur **dessin**, dans la liste comme dans le
catalogue d'ajout, et ce dessin est le **chemin parcouru** — c'est-à-dire
exactement ce qui distingue les trois mouvements :

| | |
|---|---|
| `MOVJ` | courbe en pointillés — chemin libre, on interpole les angles |
| `MOVL` | segment de droite |
| `MOVC` | arc passant par trois points |
| `ATTENTE` | horloge |
| `OUTIL` | ventouse (tige + cuvette) |
| `SORTIE` / `ENTREE` | flèche sortant du, ou entrant dans le contrôleur |

Les pictogrammes sont **dessinés par le programme** (`_icone()` dans
[ui.py](dobot_sim/ui.py)) : aucun fichier d'icône à installer, et ils
suivent la couleur du thème.

Deux barres d'outils remplacent les neuf boutons en toutes lettres qui
imposaient à eux seuls la largeur du panneau : **édition** (ajouter,
dupliquer, supprimer, monter, descendre) au-dessus de la liste,
**exécution** (lire/pause, arrêter, pas précédent, pas suivant, facteur de
vitesse) en dessous. La flèche du `+` ouvre le catalogue des sept types.

### Faire de la place aux courbes

Le formulaire *Instruction sélectionnée* est **repliable** par sa case à
cocher : déplié il occupe ≈ 660 px de haut, replié 37. **Fenêtres ▸
Agrandir les courbes** (`Ctrl+Maj+C`) fait les deux gestes d'un coup —
courbes au premier plan, plus de la moitié de la hauteur, formulaire
replié. **Disposition par défaut** revient en arrière.

### L'agencement des panneaux

Chaque panneau latéral démarre à la largeur dont son contenu a **besoin** —
son `minimumSizeHint`, pas son `sizeHint` —, bornée à 32 % de la fenêtre
pour que la vue 3D garde sa place.

> **La nuance a son importance.** Une première version visait la largeur
> *minimale du dock*, une constante choisie à la main. Résultat : huit
> champs de saisie des curseurs se retrouvaient 250 px hors du panneau,
> derrière une barre de défilement — donc impossibles à cliquer. Un
> panneau qu'on ne peut pas utiliser ne fait pas gagner de place.

> **Et le `sizeHint` n'est pas le bon critère non plus** : c'est la largeur
> *préférée*, qu'une seule étiquette de texte long fait exploser. Sur ce
> projet, une annotation collée aux valeurs du repère objet portait la
> ligne à 74 caractères en police fixe et imposait **614 px** à tout le
> panneau. Elle est passée sur sa propre ligne, et les formats d'affichage
> ont été resserrés à 34 caractères.

**En hauteur**, chaque panneau haut est enveloppé d'une zone de
défilement. Sans cela sa hauteur minimale devient celle de la **fenêtre** :
*Programme* en exigeait 954 px et *Robot réel* 637, ce qui portait le
minimum de la fenêtre à **1284 px** pour un écran qui en offre 1020. Elle
débordait par le bas et l'affichage se chevauchait. Le minimum est
retombé à **473 px**, et une fenêtre tient désormais sur n'importe quel
écran.

**Ce qui partage un jeu d'onglets** se choisit sur l'usage, pas sur la
proximité :

| | |
|---|---|
| **Commande** | seul à gauche — les curseurs doivent rester visibles |
| **Programme** et **Robot réel** | à droite, en onglets : on édite un programme *ou* on pilote la machine, jamais les deux en même temps |
| bas | Matrices, Courbes, E/S, Journal |

*Robot réel* était initialement en onglet avec *Commande* : piloter le bras
à la main demandait alors de basculer sans cesse entre les curseurs et
l'arrêt d'urgence.

### Pas à pas et inspection

Trois façons de parcourir un programme sans le rejouer en entier :

- **Pas suivant / Pas précédent** (`Ctrl+→`, `Ctrl+←`) avancent d'une
  instruction et s'arrêtent **sur sa cible**, pas sur son point de départ ;
- **cliquer une ligne** du programme y amène directement le robot, si
  *se positionner sur la ligne sélectionnée* est coché ;
- le menu **Exécution** regroupe tout cela, plus `Espace` pour lancer ou
  mettre en pause.

Éditer une instruction ne déplace jamais le robot : seule une sélection
faite à la main le fait.

Les deux jeux de curseurs restent en permanence d'accord : bouger un axe
articulaire met à jour x, y, z, r par le MGD, et inversement par le MGI.
C'est la façon la plus directe de faire sentir la différence entre les deux
espaces — et de voir la portée et la manipulabilité s'effondrer quand on
approche d'une singularité.

Quand un point demandé sort de l'espace atteignable, le robot **ne bouge
pas**, les quatre axes cartésiens passent en rouge et le bandeau d'état
indique la cause (par exemple : *distance épaule-poignet = 284,3 mm,
admissible [12 ; 282]*). Le bouton **Resynchroniser** recopie la pose
courante du robot dans les curseurs.

### *Appliquer le MGI pendant le glissement*

Cochée, le modèle suit le curseur en temps réel — chaque pixel parcouru
appelle le MGI, ce qui coûte **2,3 ms** et reste parfaitement fluide.
Décochée, la consigne n'est appliquée qu'au **relâchement**, sans passer
par les poses intermédiaires.

> **Ce que cette case ne sert PAS à faire.** Elle a longtemps été présentée
> comme permettant « de composer une consigne complète avant de la
> valider ». C'est faux : en mode continu, la valeur d'un curseur est
> conservée même quand le MGI refuse le point, si bien que la composition
> aboutit de toute façon. Passer par (50, 0, 0), inatteignable, puis amener
> y à 250 mène bien à (50, 250, 0).

> **Et décocher fut un temps dangereux.** Curseur cartésien relâché avec le
> **pilotage** du robot actif, le modèle n'avait pas bougé et la machine
> recevait la configuration **précédente** — elle partait donc quelque part
> sans rapport avec le curseur qu'on venait de tirer. Le relâchement
> applique désormais le MGI avant d'envoyer quoi que ce soit.

Un **programme** dont les cibles sortent de l'espace atteignable, ou qui
envoie le bras dans la table, est signalé par un **bandeau** au-dessus de la
vue 3D dès le chargement ; le détail reste dans le *Journal*. Deux niveaux,
car tous les avertissements ne pèsent pas pareil :

| | |
|---|---|
| **rouge** | le robot ne peut pas y aller — butée dépassée, point hors d'atteinte, **collision**, programme non compilable |
| **ambre** | le mouvement se fait, mais avec une réserve — arc dégénéré exécuté en droite, survitesse |

Il disparaît dès que le programme redevient sain.

### Le modèle ne franchit pas ses butées

Une pose hors course est **refusée avant d'être appliquée** : le bras reste
où il est, le bandeau rouge nomme l'axe fautif et sa plage, et le détail
part au Journal. Montrer le modèle dans une posture que la machine ne
prendrait jamais donnerait une fausse idée de ce qui est réalisable — c'est
précisément ce qu'un simulateur de TP doit éviter : **une trajectoire
validée à l'écran doit passer sur la table.**

Le garde-fou est unique et tient en un point, `_appliquer_q()`, que tous les
chemins traversent : curseurs, MGI cartésien, formulaire d'instruction,
pas à pas, lecture du programme. Les afficheurs sont ensuite remis en phase
avec la pose **conservée**, de sorte qu'un champ poussé trop loin revient
tout seul à la valeur tenue.

Deuxième verrou, en amont : les champs de cible d'une instruction sont
**bornés à la course de leur axe** en mode articulaire — taper 180 sur J1
donne 135. `_bornes_cible()` les rouvre en mode cartésien, où c'est le MGI
qui tranche, et garde Rz sur un tour. L'ordre compte : borner *après* avoir
écrit écrêterait la valeur qu'on vient d'écrire, et un X de 220 mm
deviendrait un J2 de 85.

Deux exceptions, voulues :

| Source | Pourquoi elle passe |
|--------|---------------------|
| **miroir** | il lit les angles de la *machine*. Son rôle est de montrer ce qu'elle fait, y compris hors de la table du modèle — la brider mentirait sur l'état réel. |
| **arrêt de sécurité décoché** | c'est le mode « laissez la faute se produire », demandé pour l'observer en démonstration. |

### Arrêt de sécurité

À la lecture, l'animation **s'arrête net** en cas de collision. Le robot
**reste sur la pose fautive** : c'est en la voyant qu'on comprend ce qui
cloche. Une boîte d'alerte non modale nomme l'instant, l'instruction et le
défaut.

`Exécution ▸ Arrêt de sécurité` (coché par défaut) gouverne les deux
mécanismes — le refus des poses hors butées et l'arrêt de l'animation.
**Décocher** laisse la faute se produire et se dérouler, ce qui reste utile
en démonstration.

### Ce qui est visible à l'ouverture

Six panneaux sur sept. **Matrices homogènes est masqué** : c'est un panneau
de cours, qu'on appelle quand on en a besoin — *Fenêtres ▸ Matrices
homogènes*, ou <kbd>Ctrl</kbd>+<kbd>Maj</kbd>+<kbd>R</kbd> qui rouvre tout.
La place du bas revient aux **Courbes**, qui est l'onglet actif.

Les deux commandes du menu *Fenêtres* sont bien distinctes, et le restent :

| Commande | Effet |
|---|---|
| **Disposition par défaut** | restaure la photo prise à l'ouverture, **visibilité comprise** — les matrices redeviennent donc masquées |
| **Tout réafficher** <kbd>Ctrl</kbd>+<kbd>Maj</kbd>+<kbd>R</kbd> | rouvre *tous* les panneaux fermés et les ramène devant, matrices incluses |

### Rouvrir une fenêtre fermée

> **Un panneau tabulé rouvert revient en arrière-plan de son groupe** : Qt
> le déclare visible tout en le parquant hors de l'écran, à (−986, −1473)
> sur la machine d'essai. On le voyait donc disparaître en croyant l'avoir
> rouvert. Il est maintenant **levé** après avoir été montré.
>
> *Disposition par défaut* ne redimensionne plus à la main : elle restaure
> une **photographie** de la disposition (`saveState`), prise au premier
> affichage — avant, Qt n'a ni calculé les géométries ni fixé l'onglet
> actif. Tailles, groupes d'onglets, onglet actif et visibilité reviennent
> d'un coup.

Les six panneaux (**Commande**, **Programme**, **Matrices homogènes**,
**Courbes**, **Entrées / Sorties**, **Journal**) sont détachables et se
ferment par leur croix. Le menu **Fenêtres** les liste toutes, cochées ou
non : un clic rouvre celle qu'on a fermée par mégarde, et **Tout réafficher**
(`Ctrl+Maj+R`) les ramène toutes d'un coup. Fermer *Commande* ou *Programme*
n'est donc plus un aller simple.

Menu **Affichage** : maillages CAO ou modèle simplifié, repères des
liaisons, squelette, grille, espace de travail, vues normalisées, calibrage
du modèle 3D.

**Cibles du programme** affiche un marqueur à chaque point visé.
**Repères des cibles** y ajoute le trièdre de la situation commandée : on
voit alors l'orientation `r` de chaque cible, et l'axe z dirigé vers le bas
qui est la direction d'approche. Masquer les marqueurs masque aussi les
trièdres.

**Zoom avant / arrière** (`Ctrl++`, `Ctrl+-`) et **Recadrer** (`Ctrl+0`)
doublent la molette, plus commode au pavé tactile ; les touches `+`, `-` et
`0` agissent aussi directement dans la vue. Le zoom est borné entre 90 et
2600 mm de distance, et **changer d'angle de vue le conserve** — seul
*Recadrer* le remet à zéro.

Menu **Fichier** : programme en JSON, export **script Dobot** (API officielle
`DobotDllType`, celle de DobotStudio) et export **CSV** des consignes
échantillonnées.

---

## 6. Le modèle 3D

### Source principale : la description ROS du constructeur

Le dossier `Magician & Accessories-2D&3D files` contient un paquet ROS
`magician_description` : un URDF décrivant la chaîne cinématique et **un STL
par corps**. C'est de loin la meilleure source, et c'est celle utilisée par
défaut (`"source": "urdf"` dans `model_config.json`).

Son avantage décisif : **les maillages sont déjà exprimés dans le repère de
leur corps**. Il n'y a plus aucune pose d'assemblage à identifier — le sujet
qui occupait la version précédente disparaît.

Le raccordement est **automatique et auto-vérifié**. Pour chaque corps de
l'URDF, on cherche le repère `k` de notre chaîne tel que

```
C = inverse(T_k(q)) · Tz(d1) · T_urdf(q)
```

soit **constant** — testé sur trois configurations très différentes. Le corps
est rattaché au repère qui minimise la variation ; si aucun ne convient, il
est signalé et ignoré plutôt que placé au hasard.

Résultat mesuré :

```
magician_base          repère 0   9630 facettes  (écart 0,0e+00)
magician_link1         repère 1   5294 facettes  (écart 1,1e-16)
magician_link2         repère 2   4242 facettes  (écart 2,2e-16)
magician_link3         repère 3   3130 facettes  (écart 1,4e-14)
magician_link4         repère 4    828 facettes  (écart 5,7e-14)
magician_link2_surface repère 2    898 facettes  (écart 2,2e-16)
magician_link3_surface repère 3   2112 facettes  (écart 1,4e-14)
magician_servos        repère 1   1560 facettes  (écart 1,1e-16)
magician_base_surface  repère 0    288 facettes  (écart 0,0e+00)
```

Un écart de 10⁻¹⁴ mm, c'est la précision machine : **notre chaîne
cinématique et celle publiée par Dobot décrivent le même robot**. Ce n'est
pas qu'un chargeur de maillages, c'est une validation croisée du modèle.

Les couleurs sont lues corps par corps dans l'URDF, et surchargeables dans
`model_config.json` (les carters de moteur et les capots, noirs sur la
machine, y sont explicitement assombris).

### Source de secours : les OBJ d'origine

`"source": "obj"` bascule sur les maillages du dossier `Modele3D`, recalés à
la main sur une pose d'assemblage identifiée géométriquement
(`pose_assemblage_deg`, rotation, translations). Le repli est **automatique**
si l'URDF est absent. Cette voie reste documentée dans `model_config.json` :
elle est plus détaillée par endroits (le logo DOBOT en relief), mais son
recalage n'est juste qu'à quelques millimètres près.

Les OBJ utilisés sont ceux de
`Modele3D/DobotMagician_obj（separate moving parts）/` — cinq pièces mobiles
séparées, environ 50 000 facettes au total, mises en cache en `.npz` au
premier lancement.

### Comment le recalage est fait

Les fichiers CAO décrivent le robot dans **une seule pose d'assemblage**.
Pour l'animer, chaque pièce doit être exprimée dans le repère de son propre
corps. Si `q0` est la configuration de la pose d'assemblage :

```
M_k = inverse(T0k(q0)) · Trans(t) · Rz(α)
```

Les valeurs sont dans `dobot_sim/model_config.json` :

| Paramètre | Valeur | Origine |
|-----------|--------|---------|
| `rotation_z_deg` | 90° | le bras pointe vers −y dans le repère CAO |
| axe J1 | x = −51, y = −4,5 mm | plan de symétrie et centre de la colonne |
| plan de pose | z = −19,2 mm | dessous du socle |
| `pose_assemblage_deg` | (0 ; 25,17 ; 28,96 ; 0) | identifiée sur la géométrie |

> **À savoir.** Le socle mesure 158 × 158 mm dans le modèle, ce qui
> correspond exactement au Magician : l'échelle est donc sûre (millimètres).
> En revanche la pose d'assemblage `q0` a été **identifiée** à partir de la
> géométrie des maillages (plan de symétrie, contact colonne/bras, position
> du support d'effecteur), et non lue dans un fichier d'assemblage. Le
> recalage est cohérent à quelques millimètres près. Si un décalage visuel
> vous gêne, **Affichage ▸ Calibrer le modèle 3D…** permet d'ajuster `q0`,
> la rotation et les translations pièce par pièce, puis d'enregistrer.
>
> Ce recalage n'affecte **que l'affichage**. La cinématique, elle, repose sur
> les paramètres constructeur (138 / 135 / 147 / 61 mm) et reste exacte.

### Couleurs

Les OBJ livrés sont découpés en groupes (`g`), et ce découpage sépare le
corps blanc des pièces noires. `model_config.json` affecte donc une couleur
**par groupe** :

| Fichier | Groupe | Ce que c'est | Couleur |
|---------|--------|--------------|---------|
| colonne | `object_1` | les deux carters de moteur (traverse toute la largeur, y −93 → +84) | noir |
| bras arrière | `object_2` | la bielle du parallélogramme | noir |
| effecteur | — | support d'outil | noir |
| tout le reste | — | corps de la machine | blanc |

Le nuanceur livré avec pyqtgraph n'a que 0,2 d'ambiant et une seule source :
un corps blanc y apparaît gris foncé. `viewer3d.py` en enregistre donc un
autre, à éclairage frontal plus lumière haute, qui rend le blanc de la
machine. Un repli automatique sur `shaded` est prévu si la compilation
échoue.

### Enveloppe de travail

**Affichage ▸ Espace de travail** balaie le contour du domaine articulaire
autour de l'axe J1, entre les butées de J1 : on voit le volume atteignable
et, très nettement, le secteur mort de 90° à l'arrière. L'enveloppe est
**recalculée à chaque changement d'outil**, puisque `lt` la décale.

**Affichage ▸ Basculer modèle CAO / simplifié** affiche un robot construit
uniquement à partir de `d1, l1, l2, l3` : par construction toujours en accord
parfait avec la cinématique, il sert de référence pour juger du recalage.

---

## 7. Outils et entrées / sorties

### Le catalogue d'outils

| Outil | Longueur | État commandé | Commande sur le robot réel |
|-------|----------|---------------|----------------------------|
| `aucun` | 0 mm | — | bride nue |
| `ventouse` | 62 mm | aspiration / relâchée | `SetEndEffectorSuctionCup` |
| `pince` | 78 mm | fermée / ouverte | `SetEndEffectorGripper` |
| `stylo` | 70 mm | — | aucun actionneur |
| `laser` | 56 mm | allumé / éteint | `SetEndEffectorLaser` |

Le point essentiel n'est pas le décor : c'est que **la longueur de l'outil
devient le déport `lt` du modèle géométrique**. Une fois l'outil monté, `z`
désigne sa **pointe** et non plus la bride — c'est ainsi qu'on vient
effleurer la table. Le MGI change donc, et l'enveloppe de travail avec lui.

### Mesurer la vraie longueur de votre outil

Les longueurs ci-dessus sont **nominales**. Un montage particulier — cuvette
différente, rallonge, embase imprimée — décale toute la coordonnée z, et le
robot descend trop bas ou pas assez.

La mesure tient en trois gestes : amenez la **pointe au contact** du plan de
travail, lisez le `z` annoncé par le modèle, ajoutez-le à la longueur
courante.

Puis inscrivez-la dans le programme, juste après le constructeur :

```python
robot = Pilote(nom="cycle", outil="ventouse")
robot.regler_longueur_outil(79.3)      # APRES monter_outil, qui remet 62
```

> **La longueur voyage avec le programme.** Elle est inscrite dans le
> `Programme` et sérialisée en JSON, parce que le nom seul ne suffit pas :
> sans elle, le simulateur et le robot réel rechargeraient la valeur du
> catalogue et le `z` ne désignerait plus le même point. Un JSON antérieur,
> qui n'en porte pas, revient proprement au catalogue.

> **Relever des poses au panneau.** Le X, Y, Z du panneau *Cartésien*
> désigne la pointe de l'outil, calculée avec la longueur **en vigueur à
> l'instant de la lecture** — celle de la ligne *longueur* de la section
> Outil. Un programme chargé impose la sienne : après avoir lancé un script
> à 79,3 mm, le panneau reste à 79,3 mm. Recopier ces valeurs dans un
> script réglé sur 62 mm enverrait le robot 17,3 mm plus bas. **Même
> longueur au relevé et dans le script**, et le point est le même, que la
> longueur soit exacte ou non.

> **Pourquoi pas un fichier de configuration machine ?** Essayé, puis
> retiré. Faire lire une longueur mesurée par `outils.outil()` la rendait
> globale : le modèle, les exemples et six tests unitaires changeaient de
> résultat selon la machine. Le catalogue reste donc **nominal et pur**, et
> la mesure appartient au programme — c'est-à-dire à l'endroit où elle est
> visible et versionnée. Pour un TP, mettez la ligne dans le squelette que
> les étudiants recopient.

> Sur la machine d'essai, pointe au contact, le modèle annonçait **+17,3 mm**
> : la ventouse montée était 17,3 mm plus longue que les 62 mm du catalogue.

> **Vérifiez que c'est bien l'outil.** Une erreur de longueur d'outil est
> **constante** : elle vaut pareil partout sur la table. Un défaut d'origine
> articulaire, lui, **varie avec la configuration** — à J2 = 71°, un degré
> d'erreur déplace le z de 2,2 mm. Refaites donc le contact en deux ou trois
> endroits bien séparés. Même valeur partout → c'est l'outil, le réglage
> ci-dessus suffit. Valeur différente → c'est l'origine qu'il faut reprendre.

```python
robot = Pilote(outil="ventouse")     # lt = 62 mm
robot.movl(cible(220, 0, 5))         # la VENTOUSE se pose à 5 mm de la table
robot.actionner(True)                # aspiration
```

> **Maillages paramétriques, pas importés.** Les STEP d'outils fournis
> (`ventouse.stp`, `Support_outil.stp`) sont des B-Rep exacts, sans
> facettes : les tesseller demanderait un noyau géométrique. Les outils
> sont donc construits par assemblage de primitives, aux dimensions Dobot à
> quelques millimètres près. Vous pouvez les retoucher dans
> `dobot_sim/outils.py` ; seule la **longueur** entre dans le calcul.

### Entrées / sorties tout-ou-rien

Le contrôleur expose 20 ports EIO. Le simulateur modélise le tout-ou-rien,
qui couvre l'essentiel des TP : commander un vérin ou un voyant, attendre
l'acquittement d'un capteur, se synchroniser avec un convoyeur.

| Instruction | Effet |
|-------------|-------|
| `SORTIE` port, état | met `EIOn` à 0 ou 1 → `SetIODO` |
| `ENTREE` port, état, délai | **suspend** le programme jusqu'à ce que `EIOn` prenne l'état voulu → boucle sur `GetIODI` |

```python
robot.sortie(1, True, "voyant de marche")
robot.attendre_entree(5, True, timeout=15.0, "capteur magasin")
```

La distinction est celle du terrain : **une sortie est un ordre, une entrée
est une information venue de la cellule**. Dans le panneau *Entrées /
Sorties*, les sorties sont des voyants pilotés par le programme, et les
entrées des boutons sur lesquels **vous** cliquez — vous jouez le rôle des
capteurs. À la lecture, le robot se **fige** sur chaque `ENTREE` jusqu'à ce
que vous basculiez le port, puis repart ; passé le délai, il continue et le
journal le signale.

> Conséquence à connaître : une attente **neutralise le lissage** de ses
> voisins. On ne peut pas arrondir un angle autour d'un arrêt. Le rapport de
> `mon_programme.py` le signale explicitement.

L'export vers le robot réel configure le multiplexage des ports utilisés
(`SetIOMultiplexing`, 2 = sortie, 3 = entrée) et traduit l'attente en
vidage de la file de commandes suivi d'une scrutation de `GetIODI`.

---

## 8. Écrire un programme dans VS Code

Ouvrez [`mon_programme.py`](mon_programme.py) et appuyez sur **F5**. La
fenêtre de simulation s'ouvre et le robot joue le mouvement. C'est tout.

### Le simulateur déjà ouvert reçoit le programme

Presser `F5` dans VS Code démarre un **nouveau processus** Python. Il ne
voit pas la fenêtre déjà ouverte — et s'il en ouvrait une seconde, le port
série, tenu par la première, lui serait refusé. Il fallait donc fermer le
simulateur avant chaque exécution.

`simuler()` frappe maintenant à la porte avant tout : si un simulateur
tourne, il lui **envoie** le programme et s'arrête là.

```
programme envoye au simulateur deja ouvert (4 instructions)
```

La fenêtre garde sa liaison série, ses réglages et sa place à l'écran ;
elle remonte au premier plan avec le nouveau programme chargé. Trois cas,
dans cet ordre :

| | |
|---|---|
| un simulateur tourne dans un **autre** processus | on lui envoie le programme |
| un simulateur tourne dans **ce** processus (console, menu *Fichier*) | on y charge le programme |
| il n'y en a pas | on ouvre une fenêtre, comme avant |

Le transport est un **socket local** — tube nommé sous Windows, socket UNIX
ailleurs. Pas de port TCP : rien à ouvrir dans un pare-feu, aucune
collision possible. Le nom du tube porte celui de l'utilisateur, pour que
deux comptes sur la même machine ne se marchent pas dessus. Un tube
orphelin, laissé par un simulateur fermé brutalement, n'est retiré
**qu'après** avoir vérifié que personne ne répond derrière.

`DOBOT_SIM_SERVICE` permet de choisir un autre nom : deux simulateurs
indépendants cohabitent alors sur le même compte — deux postes de TP sur
une machine partagée, ou une recette qui ne doit pas parler à la fenêtre
ouverte à côté.

### Les cibles sont des matrices homogènes

Une cible n'est pas un triplet de nombres : c'est la **situation** d'un
repère, position et orientation ensemble. Deux constructeurs :

| Fonction | Construit | Axe z |
|----------|-----------|-------|
| `repere_travail(x, y, z, rz)` | `Trans · Rz` | vers le **haut** — un plan de travail |
| `cible(x, y, z, rz, repere=…)` | `Trans · Rz · Rx(180°)` | vers le **bas** — la direction d'approche de l'outil |

```python
import numpy as np
from dobot_sim import Pilote, cible, repere_travail
from dobot_sim import homogeneous as hg

# Deux repères posés sur la table. Déplacer physiquement le magasin ne
# demandera de changer qu'une seule ligne.
MAGASIN = repere_travail(x=215, y=-95, z=25)
PALETTE = repere_travail(x=205, y=75, z=25, rz=20)     # palette de biais

PRISE = cible(repere=MAGASIN)                          # matrice 4×4
def case(i):                                           # coordonnées locales
    return cible(x=0, y=i * 24, z=0, repere=PALETTE)

# Dégagement de 85 mm vers le haut. Post-multiplier, c'est se déplacer dans
# le repère de l'OUTIL ; son axe z pointant vers le bas, monter s'écrit
# avec un z négatif.
DEGAGEMENT = hg.translation(0, 0, -85)

robot = Pilote(vitesse=140, acceleration=600,          # MOVL/MOVC : mm/s
               vitesse_articulaire=50)                 # MOVJ      : deg/s

robot.repli()
robot.movj(PRISE @ DEGAGEMENT, lissage=70)
robot.movl(PRISE, vitesse=50, profil="trapeze")        # descente lente
robot.ventouse(True)
robot.attendre(0.3)
robot.movl(PRISE @ DEGAGEMENT, lissage=80)
robot.movc(via=cible(265, 0, 135), T=case(0) @ DEGAGEMENT, vitesse=180)
robot.movl(case(0))
robot.ventouse(False)
robot.repli()

print(robot.resume())        # durées, avertissements, butées
robot.simuler()              # ouvre la fenêtre et lance l'animation
```

**Pré- ou post-multiplication**, c'est toute la différence. La palette étant
inclinée de 20°, un décalage de 30 mm selon x ne donne pas le même point
selon le repère où on l'exprime :

```
case(0)                              (205,0 ; 75,0 ; 25)
case(0) @ Trans(30, 0, 0)            (233,2 ; 85,3 ; 25)   repère OUTIL
Trans(30, 0, 0) @ case(0)            (235,0 ; 75,0 ; 25)   repère BASE
```

Une cible dont l'axe d'outil n'est pas vertical est **refusée** avec un
message explicite, plutôt que silencieusement projetée : c'est une
contrainte mécanique du parallélogramme, pas une approximation.

> **Un piège à connaître.** L'axe z de l'outil pointant vers le bas,
> `cible(…, rz=20) @ rot_z(30°)` donne **r = −10°**, pas +50° : la rotation
> post-multipliée s'applique autour d'un axe inversé. C'est mathématiquement
> juste — et instructif — mais pour fixer une orientation, utilisez
> l'argument `rz=` plutôt qu'une composition.

Trois points de confort :

- **les coordonnées omises sont conservées** — `robot.movl(z=25)` descend à
  la verticale sans qu'on répète `x` et `y`, et `robot.movl(PRISE, z=60)`
  part d'une matrice en forçant une composante ;
- **chaque commande accepte** `vitesse`, `acceleration`, `lissage`, `profil`,
  `duree` et `commentaire`, qui surchargent les réglages du constructeur ;
  passer `duree=` bascule cette instruction en durée imposée ;
- `robot.resume()` affiche le programme, les durées réellement calculées et
  les avertissements **sans rien ouvrir** — pratique dans le terminal.

> Le programme enregistré stocke `(x, y, z, r)` et non la matrice : sur ce
> robot l'orientation est imposée par la mécanique, les deux formes sont
> donc équivalentes et le JSON reste lisible. `Instruction.matrice()`
> reconstruit la matrice à la demande.

### Deux formes, deux usages

| | forme |
|---|---|
| **lecture** — panneaux, résumé d'un repère, formulaire d'instruction | `(X, Y, Z, Rz)` |
| **écriture** — *Modifier* un repère, *Saisir* une cible, script Python | matrice homogène 4×4 |

Une situation **s'écrit en matrice homogène 4×4**. C'est la forme demandée
aux étudiants, et c'est la seule que les dialogues de saisie acceptent :
`EditeurMatrice` propose seize cases, dont les quatre de la dernière ligne
sont montrées mais figées à (0, 0, 0, 1) — les cacher laisserait croire que
la matrice est 3×4.

La **lecture** reste simplifiée : `(X, Y, Z, Rz)` suffit à relire une pose
d'un coup d'œil. Rx et Ry n'y figurent pas, le parallélogramme les impose —
180° et 0 pour une cible et pour le repère outil, 0 et 0 pour le repère
objet. La contrainte est rappelée dans l'infobulle du curseur **Rz**, le
seul qu'elle concerne. Le dialogue de saisie, lui, relit la matrice à chaque
chiffre tapé et affiche ce qu'elle signifie en `(X, Y, Z, Rz)` — ou le motif
du refus.

`vers_xyzrpy(T)` et `depuis_xyzrpy(X, Y, Z, Rx, Ry, Rz)` font la conversion
dans les deux sens, en millimètres et degrés.

#### Une rotation tapée au clavier est arrondie

cos 45° s'écrit 0,7071 dans une case à quatre décimales, pas 0,70710678.
L'écart est minuscule — 7·10⁻⁶ — mais il suffit à faire échouer le contrôle
d'orthonormalité, qui travaille à 10⁻⁶. Refuser une matrice juste pour cette
raison n'aurait aucun sens : `_orthonormaliser()` la **projette sur la
rotation la plus proche** par décomposition en valeurs singulières, le
dialogue annonce de combien elle a bougé, et ne refuse qu'au-delà de
`ECART_ROTATION` = 10⁻³. Là, ce n'est plus un arrondi mais une erreur :
taper 0,7 au lieu de 0,7071 donne un écart de 7·10⁻³, refusé avec le motif.

> Effet de bord corrigé au passage : le bouton *Saisir* écrivait la rotation
> `r` dans le champ **Rx** et jamais dans **Rz** — les quatre valeurs rendues
> par `decomposer_cible()` étaient copiées dans les quatre premiers champs
> d'une grille qui en comptait six. La grille en compte maintenant quatre,
> dans l'ordre X, Y, Z, Rz, et la valeur arrive au bon endroit.

### Repère outil et repère objet

Deux repères, donnés en matrice 4×4 aussi bien par le script que par la
section **Repères**, qui les relit en `(X, Y, Z, Rz)`. Ce ne sont pas des
commodités d'affichage : ils entrent dans le modèle géométrique.

| Repère | Va de | Rôle | Contrainte |
|--------|-------|------|------------|
| **outil** | axe de rotation de l'outil → pointe | déport **latéral** et rotation propre, que la seule longueur ne décrit pas | axe z vers le bas |
| **objet** | base → plan de travail | les cibles marquées `repere="objet"` y sont exprimées | axe z vers le haut |

```python
robot.definir_repere_outil(                     # pince déportée de 18 mm
    hg.composer(hg.translation(18, -6, -78), hg.rot_x(np.pi)))

robot.definir_repere_objet(repere_travail(205, 75, 25, rz=20))
robot.movl(cible(0, 48, 0), repere="objet")     # coordonnées locales
```

Le MGI retire d'abord le repère outil — le triangle se résout sur **l'axe
outil**, pas sur la pointe — puis résout comme avant. Avec le repère par
défaut cela revient exactement à remonter de `lt`, donc rien ne change pour
un outil classique ; avec un déport latéral, l'aller-retour MGD/MGI sur la
pointe reste exact à 2·10⁻¹³ mm.

Déplacer physiquement le montage ne demande alors de changer **qu'une
matrice** : toutes les cibles exprimées dans le repère objet suivent.

> Une **inclinaison** est refusée dans les deux cas, avec un message :
> le parallélogramme maintient l'axe d'outil vertical, un plan de travail
> incliné rendrait toutes ses cibles inatteignables.

> À l'export, les cibles du repère objet sont converties en **coordonnées
> absolues** — le contrôleur Dobot ne connaît pas cette notion — et un
> avertissement est inséré si un repère outil déporté est en place, car le
> contrôleur ne gère qu'un déport vertical.

### Tester ses cibles avant de bouger

`mon_programme.py` contient deux sections de contrôle, à garder ou à
supprimer selon l'usage. La première vérifie les **matrices** avant même de
construire le mouvement — une erreur de repère se voit là, pas en regardant
le robot partir de travers :

```
4. CONTROLE DES CIBLES
  PRISE                  x=  215.0 y=  -95.0 z=  25.0 r=   0.0   ok  (ecart MGD/MGI 6e-14)
  case(1)                x=  196.8 y=   97.6 z=  25.0 r=  20.0   ok  (ecart MGD/MGI 9e-14)
  ...
  inverse(PALETTE) @ case(i)  ->  coordonnees locales
    case(1) : [ 0. 24.  0.]   (attendu [0 24 0])
  cible inclinee de 25 deg        : correctement refusee
```

Chaque cible est contrôlée sur quatre points : matrice homogène valide,
orientation réalisable, atteignabilité dans les butées, et surtout
`MGD(MGI(T)) = T` — le robot rejoint-il *exactement* la situation demandée ?
L'écart tombe à 10⁻¹⁴ mm. L'inversion du repère de travail redonne les
coordonnées locales, ce qui valide d'un coup la chaîne de composition.

La seconde section mesure la distance minimale du TCP à chaque cible **sur
la trajectoire échantillonnée** :

```
6. CONTROLE DU MOUVEMENT
    03 MOVL  descente lente                  0.00 mm   atteinte
    06 MOVL  remontee                        1.28 mm   coin coupe, lissage 80 %
    07 MOVC  transfert en arc                1.94 mm   coin coupe, lissage 80 %
```

Les cibles sans lissage sont atteintes à **0,00 mm**, celles qui en portent
sont volontairement manquées de 1 à 2 mm. C'est la vérification de bout en
bout : matrice → MGI → trajectoire → position réelle de l'outil.

Le dossier `.vscode/` contient quatre configurations de lancement :
*Simuler le fichier ouvert*, *Simuler mon_programme.py*, *Ouvrir le
simulateur* et *Diagnostic de l'installation*.

### Depuis le simulateur déjà ouvert

**Fichier ▸ Exécuter un script Python…** (Ctrl+R) exécute un script dans la
fenêtre courante, et **F5** rejoue le dernier. `simuler()` détecte alors la
fenêtre existante et y charge le programme au lieu d'en ouvrir une seconde :
le même fichier fonctionne dans les deux sens. C'est la boucle de travail la
plus rapide — on modifie le script dans VS Code, on revient au simulateur, F5.

### Sans interface

```python
traj = robot.trajectoire()            # compile sans rien afficher
e = traj.echantillonner(0.01)         # t, Q, dQ, P, V, R
print(e["V"].max(), "mm/s")
print(robot.controler())              # butées, points hors d'atteinte
robot.exporter_python("cycle_reel.py")
```

---

## 9. Utilisation comme bibliothèque

```python
from dobot_sim import DobotMagician, homogeneous as hg

robot = DobotMagician()

T = robot.mgd([30, 20, 35, -15])      # modèle géométrique direct
print(hg.formater(T, 3, "T0_TCP"))

q = robot.mgi(T)                      # modèle géométrique inverse
x, y, z, r = robot.pose(q)            # pose au format Dobot

J = robot.jacobienne(q)               # modèle cinématique
print(robot.depassements(q))          # butées violées, liste vide si tout va bien
```

Trajectoire cartésienne rectiligne :

```python
from dobot_sim.trajectory import SegmentLineaire, ProfilTrapeze, Trajectoire

q0 = robot.mgi_pose(220, -90, 60, 0)
T1 = robot.pose_vers_matrice(220, -90, 5, 0)
seg = SegmentLineaire(robot, q0, T1, ProfilTrapeze(1.0, 0.3))

traj = Trajectoire(robot, [seg])
e = traj.echantillonner(0.02)         # t, Q, dQ, P, V, R
print(traj.controler())               # avertissements éventuels
```

---

## 10. Le robot réel

Trois voies : **exécuter depuis son script**, **exporter un script**, ou
**piloter directement** depuis le simulateur.

### Exécuter depuis son script : `executer_sur_robot`

À la place de `robot.simuler()`, ou en plus :

```python
robot.executer_sur_robot(vitesse=50)     # en % de la vitesse nominale
```

Le robot part **tout de suite**, et l'appel rend la main **à la fin du
mouvement** : un script peut donc enchaîner plusieurs programmes, et
décider entre deux. `vider_programme()` repart d'un programme vide — même
outil, même repère, départ à la posture où le précédent a laissé le bras :

```python
robot.movj(PRISE @ HAUT)
robot.movl(PRISE)
robot.executer_sur_robot(vitesse=50)

robot.vider_programme()                  # sans cela, tout serait rejoué
robot.ventouse(True)
robot.movl(PRISE @ HAUT)
robot.executer_sur_robot(vitesse=80)
```

Le chemin se choisit tout seul :

- **simulateur ouvert et connecté au robot** — il tient le port série :
  le programme lui est confié par la passerelle et il l'exécute **sans
  confirmation**, miroir actif ; le script suit l'avancement ;
- **sinon** — le script se connecte lui-même et **garde la liaison
  ouverte** d'un appel à l'autre ; elle se ferme à la fin du script.

Garde-fous : chaque programme est **contrôlé avant tout envoi** (butées,
collisions) ; un robot **en défaut** — c'est le cas après chaque mise sous
tension — ouvre l'**assistant de mise en route**, et rien ne part tant
qu'un défaut subsiste ; **Ctrl+C** déclenche un **arrêt d'urgence**.

En cas d'échec — butée, collision, robot en défaut, liaison perdue — une
**alerte rouge encadrée** s'affiche dans le terminal, et l'exception
`ErreurExecution` **arrête le script** : un programme suivant partirait
d'une posture qui n'est pas celle prévue. Pour gérer l'échec soi-même :

```python
from dobot_sim import ErreurExecution
try:
    robot.executer_sur_robot(vitesse=50)
except ErreurExecution:
    ...                                  # le robot est arrêté, à vous
```

Les avertissements non bloquants (survitesse…) s'affichent en jaune avant
le départ, sans empêcher l'exécution.

### Commande temps réel : `Commande`

`Pilote` décrit un programme puis le joue d'un bloc. `Commande` est l'autre
paradigme : **chaque appel bouge tout de suite**, et ne rend la main qu'une
fois le mouvement **fini**. C'est ce qu'il faut pour un script interactif —
une cible tapée au clavier, plus tard une caméra :

```python
from dobot_sim import Commande, ErreurExecution, cible

with Commande(simulateur=True, robot=True, vitesse=40,       # mm/s
              zone=((170, 270), (-90, 90), (30, 140))) as robot:
    robot.movj(cible(220, 0, 80))              # rend la main à l'arrivée
    x, y, z, r = robot.pose()                  # LUE sur la machine
    robot.movl([(cible(220, 60, 80), 1.5),     # plusieurs cibles, chacune
                (cible(260, 60, 80), 0.8)])    # avec sa durée, SANS arrêt
    try:
        robot.movl(cible(400, 0, 80))
    except ErreurExecution:
        print("refusée : rien n'est parti")
```

Mêmes verbes et mêmes cibles que `Pilote` (`movj`, `movl`, `movc`, `repli`,
`ventouse`, `sortie`, `entree`…). Trois cibles matérielles :

| | |
|---|---|
| `robot=True, simulateur=False` | le script ouvre lui-même le port série |
| `simulateur=True, robot=False` | la fenêtre ouverte anime chaque commande |
| `simulateur=True, robot=True` | la fenêtre tient la liaison et exécute sur le robot, miroir actif ; le script attend la fin du **robot** |

`lancer_simulateur=True` ouvre `main.py` dans un processus séparé si aucune
fenêtre ne répond.

**Garde-fous**, vérifiés avant chaque mouvement depuis la posture **lue** :
MGI, butées et collisions sur tout le chemin, `zone=`, `pas_max=` ; un refus
lève `ErreurExecution` sans rien envoyer. Robot en défaut : assistant de mise
en route. Ctrl+C : arrêt d'urgence.

**Plusieurs cibles s'enchaînent sans arrêt** grâce au mode CP du contrôleur
(`SetCPParams` / `SetCPCmd`), dont les règles ont été mesurées sur la machine
(`essai_cp.py`) : la vitesse de croisière est le `junctionVel` de
`SetCPParams`, mis en file entre deux portions ; l'accélération et la
courbure plafonnent la vitesse ; l'outil ne tourne pas — un segment qui le
fait tourner repasse en point à point, et c'est annoncé. La trajectoire est
construite par le moteur du simulateur, échantillonnée tous les 5 mm, et
chaque durée est **prévue** avant le départ : ce qui ne peut pas tenir est
signalé en jaune, avec la durée réaliste. `continu=False` force le point à
point.

Exemples pas à pas, au clavier : [`examples/controle-temps-reel/`](examples/controle-temps-reel/).

### Exporter un script DobotStudio

`Fichier ▸ Exporter un script Dobot` produit un script utilisant
`DobotDllType`, l'API livrée avec DobotStudio :

```python
dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 220, -90, 5, 0, isQueued=1)
dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=1)
dType.SetARCCmd(api, [265, 0, 130, 0], [220, 90, 110, 45], isQueued=1)
```

### Piloter directement : le panneau *Robot réel*

`dobot_sim/liaison.py` implémente le **protocole série officiel** en Python
pur, d'après `Dobot-Communication-Protocol-V1.1.5.pdf` fourni à la racine.
Ni DLL, ni DobotStudio : seul `pyserial` est requis, et le module marche sur
Windows, Linux et macOS.

```
0xAA 0xAA │ Len │ ID │ Ctrl │ Params… │ Checksum
```

`Len = 2 + n` (la charge utile compte l'ID et le Ctrl) ; `Ctrl` porte
`rw` en bit 0 et `isQueued` en bit 1 ; le checksum est le **complément à
deux** de la somme de la charge utile ; les paramètres sont en
**little endian**. Liaison 115200 bauds, 8N1.

Les commandes **en file** (`isQueued = 1`) sont empilées dans le contrôleur
et renvoient un index sur 64 bits. C'est ce mécanisme qui permet de
téléverser tout un programme puis de le lancer : on compare l'index courant
du contrôleur à celui de la dernière commande envoyée pour suivre l'avancée.

| instruction | commande envoyée |
|---|---|
| `MOVJ` cartésien / articulaire | `SetPTPCmd` mode `MOVJ_XYZ` / `MOVJ_ANGLE` |
| `MOVL` | `SetPTPCmd` mode `MOVL_XYZ` |
| `MOVC` | `SetARCCmd` (point de passage + arrivée) |
| `ATTENTE` | `SetWAITCmd` (ms) |
| `SORTIE` | `SetIOMultiplexing` puis `SetIODO` |
| `ENTREE` | `SetTRIGCmd` — la file n'a pas d'attente conditionnelle, **TRIG en tient lieu** |
| `OUTIL` | `SetEndEffectorSuctionCup` / `Gripper` / `Laser` selon l'outil monté |
| vitesse, accélération | `SetPTPJointParams` ou `SetPTPCoordinateParams`, **mises en file** pour prendre effet au bon moment |

### Le repère du contrôleur n'est pas celui du modèle

Mesuré sur une machine réelle (firmware 3.7.0), `GetPose` annonce :

```
rho_ctrl = l1·sin(J2) + l2·cos(J3) + xBias
z_ctrl   = l1·cos(J2) − l2·sin(J3) + zBias
```

Deux différences, toutes deux vérifiées au millième :

- le contrôleur place son origine **à hauteur d'épaule** : son `z` ne
  contient pas `d1`. L'écart mesuré valait exactement **−139,000 mm** ;
- son déport horizontal est le `xBias` de `EndEffectorParams`, pas le
  `l3 = 59,7 mm` de la table DH. Sur la machine d'essai `xBias = 61,0 mm`,
  d'où **1,30 mm** d'écart radial.

D'où la conversion, exacte dans les deux sens :

```
rho_ctrl = rho_tcp − l3 + xBias      z_ctrl = z_tcp + lt − d1 + zBias
```

`lt` y figure parce que le `z` du simulateur désigne la **pointe** de
l'outil, quand celui du contrôleur se rapporte au plan du poignet tant que
`zBias` vaut 0. Le pont **lit** `xBias` à la connexion et le signale dans le
journal ; il ne modifie **jamais** la calibration de la machine.

Les **angles**, eux, ne demandent aucune conversion : les quatre valeurs
renvoyées par `GetPose` sont exactement celles du modèle. C'est pourquoi le
miroir temps réel les utilise plutôt que les coordonnées.

### Les deux sens de circulation

| case | sens | |
|---|---|---|
| **miroir** | robot → modèle | interroge la machine à 5 Hz et applique ses angles au modèle : le robot virtuel reproduit le mouvement du vrai, en direct. Un aller-retour série coûte **10,2 ms** |
| **pilotage** | modèle → robot | chaque curseur **relâché** envoie le robot sur la pose affichée |

Elles sont **exclusives** : cocher l'une décoche l'autre. Si le modèle
suivait la machine pendant que la machine suit le modèle, les curseurs se
battraient avec la main de l'opérateur.

### Piloter la machine à la main

Le **pilotage** fait des quatre curseurs articulaires et des quatre
curseurs cartésiens une console de mouvement pour le robot réel.

Ce qui part sur le fil, ce sont toujours les **angles** — `MOVJ_ANGLE` —
et ce quel que soit le curseur manipulé, y compris cartésien. Le simulateur
a déjà résolu le MGI, appliqué la longueur d'outil et vérifié les butées :
transmettre sa configuration articulaire garantit que le bras reproduit
exactement la posture montrée en 3D, sans qu'aucune conversion de repère
puisse s'y glisser.

Quatre garde-fous :

| | |
|---|---|
| **au relâchement seulement** | `valeurChangee` part à chaque pixel parcouru et noierait la machine ; le pilotage s'accroche à `valeurValidee`, émis au relâchement du curseur ou à la validation du champ |
| **le dernier ordre gagne** | chaque envoi coupe puis vide la file avant d'y déposer le mouvement : un relâchement **remplace** le précédent au lieu de s'empiler derrière |
| **pose fautive refusée** | butée dépassée ou collision : rien ne part, et le bandeau le dit |
| **confirmation à l'activation** | et la case se décoche d'elle-même à la déconnexion |

La vitesse reste bridée par le ratio du panneau — 30 % par défaut.

### Le badge dans la vue 3D

Quand une **machine** est au bout du fil, cela doit se voir là où l'on
regarde — pas seulement dans un panneau latéral qu'on a peut-être replié.
Un badge se superpose donc au coin de la vue 3D, à trois degrés :

| | |
|---|---|
| **bleu** — *ROBOT REEL CONNECTE* | la liaison est ouverte, rien ne bouge |
| **ambre** — *ROBOT REEL PILOTE* | les curseurs commandent la machine |
| **rouge** — *EN MARCHE* / *EN DEFAUT* | le programme s'exécute, ou la file est arrêtée sur une alarme |

### Forcer les sorties à la main

Les sorties ne sont plus de simples voyants : **cliquez-les pour les
forcer**. C'est la façon la plus courte de vérifier qu'une électrovanne, un
voyant ou un relais répond — sans écrire de programme ni lancer de cycle.

Cochez *« répercuter les sorties sur le robot réel »* pour que le clic
agisse aussi sur la machine. Le port est configuré en sortie
(`SetIOMultiplexing`) au premier usage, puis chaque bascule part en
`SetIODO` **immédiat** — pas en file : on veut l'effet tout de suite.

> Une confirmation prévient que le matériel branché va s'actionner. Et
> quand c'est le **programme** qui écrit une sortie pendant une lecture,
> rien ne part sur le fil : seul un clic de votre part commande la machine.

> **L'effecteur n'est pas sur les EIO.** Vérifié sur machine : actionner les
> vingt sorties une à une n'a **aucun effet** sur la ventouse. Ventouse,
> pince et laser ont leur propre circuit et leurs propres commandes —
> `SetEndEffectorSuctionCup`, `…Gripper`, `…Laser`. Les ports EIO servent
> aux périphériques de la cellule : voyants, vérins, convoyeur, capteurs.

### Actionner l'effecteur à la main

Le bouton **Actionner** de la section *Outil*, panneau Commande, commande
aussi la machine dès que le **pilotage** est coché. Il envoie la commande
dédiée à l'outil monté — ventouse, pince ou laser —, en **immédiat** et non
en file : on veut entendre la pompe tout de suite, pas à la fin d'un cycle.
Un outil passif comme le porte-stylo laisse le bouton grisé.

C'est la façon de vérifier qu'une ventouse aspire, sans écrire de programme.

### Lire les vraies entrées

Cochez *« lire les entrées sur le robot réel »* et les vingt boutons cessent
d'être des interrupteurs : ils deviennent les **témoins** des capteurs de la
cellule. On voit un détecteur de présence basculer en direct, avant de
lancer un cycle qui l'attend avec une instruction `ENTREE`.

> **Comment les vingt ports sont rafraîchis.** Les lire tous coûterait
> vingt allers-retours, soit 200 ms — bien trop à chaque tour. Les ports
> **attendus par le programme** sont donc lus à chaque rafraîchissement,
> cinq fois par seconde ; les autres sont balayés à tour de rôle, trois à
> la fois, et font le tour complet en une seconde environ.

Un port que vous avez forcé en **sortie** n'est jamais reconfiguré en
entrée : les deux usages cohabitent sans se marcher dessus. Et la
déconnexion rend la main aux interrupteurs.

### L'alerte s'efface quand la pose redevient saine

Un arrêt de sécurité laisse un bandeau rouge. Une fois le bras ramené dans
ses butées — au curseur, au pas à pas ou à la main — ce bandeau ne décrit
plus rien : il **s'efface tout seul**, sans qu'il faille recompiler le
programme.

Les avertissements du **programme**, eux, survivent : ils parlent de la
trajectoire, pas de l'instant. Le bandeau y revient une fois l'alerte de
pose retirée.

### Recaler les capteurs d'angle

Symptôme : **le bras réel n'est pas là où le modèle le montre** — typiquement
le bras arrière nettement plus reculé — et la prise d'origine **échoue en
butée mécanique**. Le contrôleur vise ce qu'il *croit* être la posture
d'origine ; avec un zéro décalé, il pousse jusqu'à la butée.

Le protocole nomme la cause (section 1.16) :

> *Angle sensors of forearm and rear arm may have a **static offset** due to
> angle sensor welding, machine status, and so on.*

Deux commandes le corrigent, et **aucune ne fait bouger le bras** — elles ne
changent que la façon dont les capteurs sont lus :

| commande | ID | portée |
|---|---|---|
| `ResetPose` | 11 | déclare les angles vrais à la posture actuelle ; immédiat, **perdu à la mise hors tension** |
| `AngleSensorStaticError` | 140 | écrit l'offset **persistant** des deux capteurs |

**Posture de référence.** La convention du contrôleur a été vérifiée sur
machine réelle — `z = l1·cos(J2) − l2·sin(J3)`, donc :

- **J2 = 0° : bras arrière VERTICAL**
- **J3 = 0° : avant-bras HORIZONTAL**

C'est celle du modèle, ce qui rend la référence vérifiable à l'équerre.

```powershell
python calibrer_robot.py --capteurs          # lit tout, N'ECRIT RIEN
python calibrer_robot.py --recaler           # ResetPose, sans mouvement
python calibrer_robot.py --erreur-statique   # recalage persistant
python calibrer_robot.py --restaurer 0.25,-0.5   # revient en arriere
```

Déverrouillez les moteurs (bouton de la fourche), placez le bras à
l'équerre, relâchez, puis lancez la commande.

> **Mesuré sur firmware 3.7.0 : `AngleSensorStaticError` ne s'applique pas
> à chaud.** La valeur est bien acceptée et relue telle quelle (+5,0000),
> mais les angles annoncés ne bougent pas d'un millième. Le firmware la lit
> au démarrage. La vérification exige donc une **coupure d'alimentation**,
> et la procédure se fait en deux temps.

> **Le sens de l'offset n'est pas documenté** et la sonde à chaud ne peut
> pas le révéler, puisqu'elle n'a aucun effet. On écrit, on redémarre, on
> regarde : si l'écart retombe vers 0, c'était le bon sens ; s'il **double**,
> on recommence avec `--signe -1`. La valeur précédente est toujours
> affichée avant écriture, et `--restaurer` la remet.

**Décalage constant ou défaut d'échelle ?**

Avant d'ouvrir la machine, il vaut la peine de savoir ce qu'on cherche.
Relevez l'angle vrai à trois ou quatre postures bien réparties — niveau,
équerre, inclinomètre de téléphone — et laissez l'ajustement trancher :

```powershell
python calibrer_robot.py --mesure 0       # bras vertical
python calibrer_robot.py --mesure 30
python calibrer_robot.py --mesure 60
python calibrer_robot.py --bilan          # hors ligne, n'ouvre pas le port
```

`--bilan` confronte les relevés à **trois** modèles d'erreur, qui ne se
réparent pas de la même façon :

| modèle | forme | cause | remède |
|---|---|---|---|
| **constant** | `lu = vrai + b` | un cran de trop sur un accouplement indexé | remontage |
| **échelle** | `lu = a·vrai + b` | le capteur ne couvre pas la bonne plage | ni remontage ni offset |
| **biellette** | `lu = vrai + A·cos + B·sin + C` | capteur entraîné par une tringlerie, pas monté dans l'axe | ni l'un ni l'autre |

Le bilan affiche le résidu de chacun et désigne le meilleur. Pour un
décalage constant, il cherche en plus s'il tombe sur un nombre entier de
crans (16, 15, 18, 20 ou 24 dents).

> **Deux points ne suffisent pas, et le bilan le dit.** Une droite passe
> toujours exactement par deux points : son résidu nul ne prouve rien, une
> erreur sinusoïdale donnerait les mêmes deux points. Il en faut **trois**
> pour séparer constant et échelle, **quatre** bien répartis pour tester la
> biellette. Le bilan refuse de conclure en deçà.

> **Leçon d'un cas réel.** Deux mesures prises toutes les deux près de la
> verticale donnaient 22,13° et 22,68°, soit 22,41° — à 0,095° de
> 360/16 = 22,500°. La coïncidence semblait décisive et a conduit à un
> démontage. Une troisième mesure, à 90°, a montré une erreur **nulle** :
> le défaut dépendait de l'angle, aucun cran n'était en cause, et le
> démontage était inutile. **Mesurez sur toute la plage avant d'ouvrir
> quoi que ce soit.**

**Séquence complète**

```powershell
# bras a l'equerre, moteurs deverrouilles
python calibrer_robot.py --capteurs           # mesure l'ecart
python calibrer_robot.py --erreur-statique    # ecrit (aucun mouvement)
#   -> couper l'alimentation, attendre 5 s, rallumer
#   -> remettre le bras a l'equerre
python calibrer_robot.py --capteurs           # l'ecart doit etre retombe a ~0
python calibrer_robot.py --origine            # LE ROBOT BOUGE
```

> **N'envoyez aucun mouvement entre l'écriture et la vérification.** Si le
> signe est inversé, les angles annoncés sont faux du double et le bras
> partirait droit en butée.

### Se passer de la prise d'origine

Quand un capteur d'angle est fautif, la prise d'origine **ne peut pas
aboutir** : elle s'appuie justement sur ce capteur et part en butée. Il
existe pourtant une issue, et elle tient à une propriété du contrôleur,
**vérifiée sur firmware 3.7.0** :

```
on declare J2 = +25.718 (soit +5 deg), sans rien bouger
   t=0.5 s   J2  +25.718
   ...
   t=4.0 s   J2  +25.718
   -> la croyance TIENT
```

Le contrôleur **garde ce qu'on lui déclare** et ne relit pas ses capteurs.
Les moteurs étant des pas-à-pas, il suffit donc de lui dire une fois où est
le bras : il compte ses pas à partir de là, et le capteur défaillant ne sert
plus à rien.

| | |
|---|---|
| **bouton** | *Declarer la posture d'equerre*, panneau Robot réel |
| **ligne de commande** | `python calibrer_robot.py --recaler` |

Placez le bras à la main — bras arrière vertical, avant-bras horizontal,
bouton de la fourche enfoncé — puis déclarez. Rien ne bouge.

> **Trois limites à connaître.** C'est un contournement, pas une
> réparation : le capteur reste faux. Le réglage est **perdu à chaque mise
> hors tension**, donc à refaire à chaque séance. Et la précision du robot
> devient celle de **votre placement** : sur la machine d'essai, cinq mises
> à la verticale successives se sont étalées sur 3,5°. Pour un TP c'est
> jouable, pour de la précision non.

> **Astuce de répétabilité.** Plutôt que de viser la verticale à l'équerre
> chaque fois, repoussez doucement le bras contre sa butée arrière : une
> butée mécanique est bien plus reproductible qu'un placement à l'œil.
> Mesurez une bonne fois l'angle qu'elle représente, puis déclarez cette
> valeur avec `--reference`.

### Mise en route sans le simulateur : `mise_en_route.py`

```
.venv/bin/python mise_en_route.py
```

ouvre le même assistant de mise en route que la fenêtre du simulateur
(posture de référence, prise d'origine, défauts effacés), seul, dans une
petite fenêtre. À lancer après chaque mise sous tension, fenêtre du
simulateur fermée, avant les scripts qui pilotent le robot directement
(`Commande(robot=True, simulateur=False)`, étalonnage de la caméra…).

Options : `--sans-posture` passe l'étape 1 (posture déjà déclarée depuis la
mise sous tension) ; `--equerre` amène d'abord le bras en équerre **en mode
manuel** (base immobile), puis ouvre l'assistant à la prise d'origine. C'est
ce qui débloque un robot dont la file ne répond plus après un défaut de
planification (alarme 0x12) : aucun mouvement programmé ne part alors, le
mode manuel si. Le même geste existe dans l'assistant (étape 2, « Amener
d'abord le bras en équerre »).

### Le robot en ligne de commande : `dobot.py`

Une commande, une action, sans ouvrir le simulateur (fenêtre fermée) :

```
.venv/bin/python dobot.py etat                   pose, angles, alarmes (ne bouge rien)
.venv/bin/python dobot.py pince ouvrir           ou fermer
.venv/bin/python dobot.py ventouse on            ou off
.venv/bin/python dobot.py pompe off              couper la pompe
.venv/bin/python dobot.py repli                  posture de repos
.venv/bin/python dobot.py equerre                bras arrière vertical, avant-bras horizontal
.venv/bin/python dobot.py aller 220 0 80         MOVJ vers (x, y, z), r gardé
.venv/bin/python dobot.py aller 220 0 80 30 --lineaire
.venv/bin/python dobot.py deplacer 0 0 20        RELATIF : monter de 20 mm
.venv/bin/python dobot.py deplacer 10 -5 0 --dr 15
.venv/bin/python dobot.py effacer                effacer les alarmes
.venv/bin/python dobot.py mise-en-route --equerre
.venv/bin/python dobot.py etalonner --reprendre  étalonnage caméra -> robot
```

`deplacer dx dy dz` bouge la pointe d'outil de (dx, dy, dz) mm **dans le
repère de la base**, en ligne droite (`--articulaire` pour un MOVJ) ;
`--dr` tourne l'outil. Les coordonnées sont celles du simulateur. Options
communes, **avant** la commande : `--longueur-outil 78`, `--vitesse 30`
(mm/s, deg/s), `--outil pince`, `--port /dev/...`. Exemple :

```
.venv/bin/python dobot.py --longueur-outil 78 --vitesse 20 deplacer 0 0 -10
```

### Prise d'origine

> **Séquence de début de séance, validée sur la machine (firmware 3.7.0).**
> Le contrôleur annonçait J2 = −9° et J3 en butée, alarmes 0x43 et 0x44,
> alors que le bras était dans sa course.
>
> 1. **Déclarer la posture d'équerre** (bouton de la fourche enfoncé, bras
>    arrière vertical, avant-bras horizontal) : J2 et J3 deviennent justes.
>    `ResetPose` ne porte que ces deux angles — **J1 reste faux**.
> 2. **Prise d'origine** : elle retrouve J1. Précédée de l'équerre, elle
>    **aboutit**, alors qu'elle partait en butée sans elle.
> 3. Contrôler au **miroir** que le modèle colle au bras, puis *Effacer les
>    défauts*.
>
> Le bouton **Mise en route…** du panneau *Robot réel* enchaîne ces trois
> étapes dans l'ordre, en n'ouvrant chacune qu'une fois la précédente
> faite. Il s'ouvre de lui-même quand le robot est en défaut à la
> connexion — ce qui est le cas après chaque mise sous tension.

Le Magician retrouve ses repères par une **prise d'origine** : le bras
parcourt toute sa course, remet ses compteurs à zéro, puis se parque sur le
point enregistré dans `HOMEParams`. Elle est nécessaire après chaque mise
sous tension, et dès que le bras a été **déplacé à la main**.

**Comment savoir si elle manque ?** Le contrôleur ne le dit pas toujours :
sur la machine d'essai, aucune alarme n'était levée alors que le bras venait
d'être déplacé à la main. Trois indices, du plus fiable au plus faible :

1. **le miroir 3D** — mettez le bras réel dans une posture reconnaissable et
   regardez le modèle. S'ils diffèrent, les zéros ont dérivé. C'est le test
   décisif, et c'est immédiat ;
2. des **alarmes de butée** (0x40 à 0x49) alors que le bras est
   visiblement dans sa course : le contrôleur le croit ailleurs ;
3. le robot va systématiquement « à côté » de ses cibles, d'un décalage
   constant.

> **Ce que `PontReel.coherence()` ne fait pas.** Il confronte la pose et les
> angles renvoyés par `GetPose` : cela vérifie que le contrôleur et le
> modèle décrivent la **même géométrie** — utile pour valider `xBias` — mais
> **pas** que les zéros sont justes. Les deux valeurs viennent des mêmes
> compteurs : elles restent d'accord entre elles même quand le zéro a
> dérivé. Sur la machine d'essai il annonçait 0,000 mm alors que la prise
> d'origine était bien nécessaire.

En ligne de commande :

```powershell
python calibrer_robot.py                 # diagnostic seul, RIEN ne bouge
python calibrer_robot.py --origine       # lance la prise d'origine
```

Le diagnostic affiche firmware, alarmes, pose dans les **deux** repères,
point d'origine enregistré et écart de cohérence. La séquence de prise
d'origine coupe la file avant de la vider, efface les alarmes, bride les
vitesses à 30 %, puis suit les angles jusqu'à stabilisation — et force
l'arrêt si le délai est dépassé.

### Quand la machine se met en défaut

Une alarme du contrôleur **gèle la file** : le programme s'arrête en cours
de route et la barre de progression reste plantée. Le panneau l'annonce
maintenant par un encart rouge nommant l'alarme et le pourcentage d'arrêt,
et le bouton **Effacer les défauts** acquitte puis vide la file.

Pour retrouver l'instruction fautive, la barre suffit : la file compte
**deux commandes par mouvement** (les paramètres de vitesse, puis le PTP).
Un programme de 7 mouvements fait donc 14 entrées, et chaque instruction
vaut ≈ 14 %.

> Le protocole V1.1.5 ne contient **pas** la table des alarmes : sa section
> 1.5 renvoie au *Dobot ALARM document*, absent du projet. `resume_alarmes()`
> traduit les codes qu'il connaît et signale les autres comme tels, sans les
> deviner. Certains ne s'acquittent pas : ils décrivent un état, pas un
> événement, et ne partent qu'après une mise hors tension puis une prise
> d'origine.

### Garde-fous

Le simulateur ne se connecte jamais tout seul, et rien ne bouge sans un
geste explicite :

| | |
|---|---|
| **ratio de vitesse** | 30 % par défaut ; bride toutes les vitesses du contrôleur |
| **téléversement** | remplit la file **sans rien lancer** ; un programme dont le contrôle signale une butée ou une collision est **refusé** |
| **relance** | *Lancer* suffit : la file se consomme en s'exécutant, elle est donc **remplie à nouveau** quand elle est épuisée |
| **ordre critique** | `StopExec` **avant** `Clear` : si la file était restée démarrée, chaque commande téléversée partirait dès son arrivée |
| **confirmation** | toute action qui fait bouger la machine passe par une boîte d'alerte |
| **arrêt d'urgence** | bouton rouge et `Ctrl+Maj+Espace` : coupe le mouvement en cours et vide la file |
| **fermeture** | quitter la fenêtre déclenche l'arrêt d'urgence avant de fermer le port |

> **La file se consomme.** Le contrôleur n'est pas une mémoire de
> programme : chaque commande exécutée disparaît de la file. Après un
> cycle, il n'y reste rien — relancer sans téléverser ne ferait
> strictement rien. Le bouton *Lancer* remplit donc la file de lui-même
> quand elle est épuisée : il veut dire « joue ce programme », et on peut
> l'enchaîner autant de fois qu'on veut. Le bouton *Téléverser* reste là
> pour qui veut séparer les deux temps.

### La file est courte, et elle ne le dit pas

Elle tient une **trentaine de commandes**. Un programme plus long n'y entre
pas d'un coup : le pont en dépose une partie, garde le reste, et l'écoule
pendant l'exécution, au rythme où la file se vide. Une boucle de dix cycles
passe donc sans que rien n'ait à être découpé à la main.

> **Comment la place libre est connue.** Le protocole V1.1.5 a **supprimé**
> `GetQueuedCmdLeftSpace` de sa documentation — c'est écrit dans son
> historique, page 3. Le **firmware la sert pourtant toujours** : relevé sur
> 4.0.3, il répond **32** file vidée et **0** file pleine. Le pont
> l'interroge donc, et ne se rabat sur l'arithmétique des index —
> `dernier_index − index_courant`, plafonnée à 24 — que si le firmware reste
> muet. Deux places restent gardées libres : entre la réponse et l'envoi
> suivant, une commande partie d'ailleurs a pu s'intercaler.

> **La panne qui a conduit à ce réglage.** Une boucle de dix cycles —
> 130 instructions, 150 commandes — s'arrêtait à **21 %**, soit
> 32 commandes, sans alarme ni message. Le pont détectait une file pleine à
> une **réponse vide** (tableau 198 du protocole : *« queuedCmdIndex or
> Empty »*). La machine, pleine, répond en fait un **index valide et jette
> la commande** : le pont croyait avoir tout envoyé, ne gardait rien à
> renvoyer, et le robot s'arrêtait au bout de ce que la file contenait
> vraiment. Mesuré au banc sur firmware 4.0.3 : la 33ᵉ commande renvoie
> huit octets d'allure normale, et **le même index que la 32ᵉ**. Même
> comportement, mesuré séparément, sur une seconde machine en firmware
> 3.7.0 : 32 commandes acceptées, puis l'index reste figé.
>
> C'est cet **index figé** qui sert désormais de signal : une commande en
> file dont l'index ne progresse pas est tenue pour refusée, et renvoyée
> plus tard. Vérifié sur la machine — `FilePleine` se lève à la 33ᵉ, pile.
>
> Le contrôleur simulé des tests, lui, refusait franchement — il avait été
> écrit d'après cette hypothèse. Les 31 tests du protocole passaient donc
> au vert pendant que la machine s'arrêtait au quart du programme. Un
> contrôleur simulé plus fidèle a été ajouté depuis : file bornée qui se
> vide avec le temps, index de la commande **en cours**, et les deux
> comportements possibles quand elle est pleine.

Si un firmware a une file plus courte, le pont s'en aperçoit — refus franc
ou index figé, les deux sont détectés — et réduit sa marge de lui-même, en
le signalant dans le journal.

```python
from dobot_sim.reel import PontReel

pont = PontReel(robot, journal=print)
pont.connecter("COM4")            # lecture seule : rien ne bouge
pont.definir_ratio(30)
pont.televerser(programme, q0)    # remplit la file
pont.demarrer()                   # LÀ, le robot bouge
pont.televerser(programme, q0)    # a refaire avant chaque nouveau cycle
pont.demarrer()
```

---

## 11. Tests

```powershell
python -m unittest discover -s tests -v
```

283 tests, dont **38 sur le protocole et le pont vers la machine réelle**
(`tests/test_protocole.py`), exécutés sans matériel grâce à un **port série
simulé** qui rejoue le comportement du contrôleur : cadrage conforme à
l'exemple du protocole, checksum, `Ctrl`, endianness, refus des trames
invalides, index de file, traduction de chaque type d'instruction, ordre
arrêt-puis-vidage, refus d'un programme fautif, et exactitude de la
conversion de repère.

`tests/test_commande.py` (65 tests) exerce la commande temps réel sans
matériel : un contrôleur simulé qui **bouge** — file bornée à 32 commandes,
index figé quand elle est pleine, angles qui suivent les commandes
exécutées —, une fausse passerelle, et la **vraie fenêtre** hors écran
branchée sur ce contrôleur. Résolution des cibles, refus (MGI, butées,
collision, zone, `pas_max`) sans qu'aucune commande ne parte, commandes CP et
PTP générées dans le bon ordre, `junctionVel` mis en file, repli en point à
point quand l'outil tourne, échantillonnage, prévision des durées, mise en
route, Ctrl+C.

Le reste couvre : identités des matrices homogènes, aller-retours RPY /
quaternion / axe-angle, conformité des deux conventions de
Denavit-Hartenberg, verticalité de l'outil sur tout l'espace articulaire,
aller-retour MGD/MGI sur 1000 configurations, cas du bras replié et des deux
branches du triangle, jacobienne analytique contre différences finies,
rectitude effective des `MOVL` (vérifiée en repassant par le MGD),
appartenance des `MOVC` au cercle et passage par le point intermédiaire,
conditions aux limites des lois horaires, sérialisation JSON et validité
syntaxique du script exporté.

Pour la cadence et le lissage : respect **exact** de `v_max` par les quatre
lois horaires, saturation simultanée de `v_max` et `a_max` par le trapèze,
bascule en profil triangulaire quand la course est trop courte, ordre des
durées entre lois, unités (deg/s pour un `MOVJ`, mm/s pour un `MOVL`),
effet monotone du lissage sur le coin coupé **et** sur la durée, continuité
de la consigne dans le raccord, non-absorption des `ATTENTE`, et décalage
correct des évènements outil. Pour l'API de script : chaînage, coordonnées
omises, surcharge locale des réglages, rejet d'une loi horaire inconnue,
cycle complet sans violation de butées, export du script robot réel.

Pour les outils et les E/S : maillage non dégénéré et rattaché au repère de
l'axe outil pour les cinq outils, emprise verticale cohérente avec la
longueur annoncée, ouverture effective des doigts de la pince, effet de `lt`
sur le MGI (la pointe atteint bien l'altitude demandée), bornes des ports,
notification uniquement sur changement d'état, chronologie et datation des
évènements `SORTIE` / `ENTREE` / `OUTIL`, neutralisation du lissage par une
attente, sérialisation de l'outil, et export conforme (`SetIOMultiplexing`,
`SetIODO`, `GetIODI`, et la bonne commande d'effecteur selon l'outil).

Pour les repères outil et objet : validité et refus d'une inclinaison,
extraction du déport, aller-retour MGD/MGI avec un outil déporté
latéralement, atteinte exacte de la consigne par la pointe, non-équivalence
avec un simple déport vertical, résolution des coordonnées locales,
insensibilité des cibles en repère base, déplacement du repère qui déplace
la trajectoire, sérialisation et export en coordonnées absolues.

Pour la description officielle : lecture des liaisons et des corps, STL
ramenés en millimètres, longueurs de bras relevées **dans la chaîne URDF**
et comparées aux nôtres, identité de la pose de référence et de la
convention articulaire sur quatre configurations, et surtout la constance
de la transformation corps ↔ repère, qui est la validation croisée des
deux modèles.

Pour les cibles matricielles : validité homogène de cible(), axe d'outil
dirigé vers le bas, aller-retour cible ↔ decomposer_cible, coordonnées
locales dans un repère de travail incliné, sémantique de la pré- et de la
post-multiplication, refus d'une orientation inatteignable et d'une matrice
non homogène, cohérence avec le MGD, et acceptation des matrices par
movj / movl / movc.

### État de la validation

Sur cette machine (Python 3.12.10, numpy 2.5.3, PyQt5 5.15.11,
pyqtgraph 0.14.0, PyOpenGL 3.1.10) :

- 179 tests unitaires au vert ;
- `main.py --verifier` : écart MGD/MGI **1,7 · 10⁻¹³ mm** sur 2000 tirages,
  jacobienne analytique contre différences finies **8,1 · 10⁻⁸** ;
- les 9 corps de l'URDF officiel se chargent (27 982 facettes), chacun rattaché automatiquement au bon repère avec un résidu inférieur à 10⁻¹³ mm ;
- l'IHM a été exercée en entier (commandes articulaires et cartésiennes,
  édition de programme, cadence, lissage, lecture, bascules d'affichage,
  exports) en rendu OpenGL **et** en repli 2D ;
- le parcours VS Code a été rejoué de bout en bout : `mon_programme.py`
  ouvre la fenêtre, charge ses 32 instructions et lance l'animation ;
- lissage mesuré sur ce cycle : **35,1 s → 28,8 s**, vitesse maximale
  inchangée (220 mm/s), aucun pic créé dans les raccords ;
- rectitude mesurée d'un `MOVL` : **10⁻¹³ mm** d'écart à la droite, contre
  1 mm pour le même déplacement en `MOVJ` — c'est toute la différence entre
  les deux modes.

`pyflakes` ne signale rien.
