#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare les deux methodes de detection (taches : vision.detecter ; modele :
vision_modele.detecter) sur des scenes synthetiques a verite connue, puis
sur les captures reelles.

    .venv/bin/python examples/perception/evaluer_detection.py
    .venv/bin/python examples/perception/evaluer_detection.py --scenes 4 --sans-reel
    .venv/bin/python examples/perception/evaluer_detection.py --param seuil_score=0.6

Synthetique : pour chaque configuration de camera, des scenes de 8 pieces
(epaisses 17 x 19 x 68 et fines 8 x 17 x 68, a plat ou sur la tranche,
dont 3 posees CONTRE une autre). On mesure le rappel (pieces trouvees /
posees), les faux positifs, l'erreur de position du centre de la face
superieure dans le plan de la table, l'erreur d'angle, la justesse du type
et de la face, et le rappel des pieces etroites et des pieces en contact.

Empilements (--empilement) : les memes configurations, des scenes de 8
pieces dont 3 posees SUR une autre (en travers, dans le meme sens, decalee
de cote, ou en tas de trois) et 1 contre une autre. On mesure en plus le
rappel des pieces empilees et des porteuses, la justesse du type et de la
face des pieces empilees, celle du SUPPORT annonce (table, ou la bonne
porteuse) et l'ordre de prise : part des scenes ou aucune porteuse n'est
prise avant la piece qu'elle porte, et ou la premiere piece a prendre est
vraiment libre. Puis la meme chose pour l'etage geometrique de
perception.py (ajustement des masques du reseau), le reseau remplace par
des masques parfaits tires de la verite (OracleMasques).

Reel : pas de verite au millimetre, seulement le nombre de pieces trouvees
(compare au nombre de pieces posees, compte a l'oeil) et l'ecart moyen des
dimensions mesurees aux dimensions du modele.

Affinage de la scene (--raffinement, raffinement.py) : avant / apres, sur
  - des scenes synthetiques de 8 pieces (en contact ; avec 3 piles ; avec
    3 pieces INCLINEES et une pile), pour chaque camera : geometrie seule,
    masques parfaits (OracleMasques), masques ABIMES (ReseauDegrade :
    masques perdus, fusionnes, faux masques sur le fond) ; rappel, faux,
    position, lacet, inclinaison, type et face, support, ordre de prise ;
  - les images de VALIDATION du jeu reel annote (apprentissage/dataset,
    celles que le reseau n'a pas apprises) : la scene estimee est rendue
    dans l'image couleur et comparee aux masques annotes (rappel,
    precision, IoU, classe), sans reseau, avec YOLO, avec YOLO abime ;
  - les captures de captures/ (centres releves a l'oeil).
--ablation : l'affinage sans chacun des termes de son cout.

    .venv/bin/python examples/perception/evaluer_detection.py --raffinement --scenes 8
    .venv/bin/python examples/perception/evaluer_detection.py --raffinement --sans-synthese
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthese as sy  # noqa: E402
import vision  # noqa: E402
import vision_modele  # noqa: E402

CONFIGS = {
    # comme scene4 / scene7 : camera a 0.5 m, inclinee de 35 deg
    "dessus": dict(cam=sy.Camera(), zone=(-110, 110, -80, 80),
                   table=(-0.4, 0.4, -0.3, 0.3), couleur=False, motif=False),
    # comme scene9 : camera reculee, pieces plus petites dans l'image
    "reculee": dict(cam=sy.Camera(distance=0.65, inclinaison=40),
                    zone=(-130, 130, -90, 90), table=(-0.45, 0.45, -0.35, 0.35),
                    couleur=False, motif=False, reflet=True),
    # comme env2 : vue RASANTE, nappe a fleurs, image couleur
    "rasante": dict(cam=sy.Camera(K=sy.K_848, taille=(848, 480), distance=1.0,
                                  inclinaison=65),
                    zone=(-170, 170, -90, 90), table=(-0.7, 0.7, -1.0, 0.6),
                    couleur=True, motif=True, lissage=3),
}

#: pieces posees sur les captures reelles, comptees sur les images
ATTENDU = {"scene4": 4, "scene7": 8, "scene9": 8, "essai_25": 8, "essai_50": 8,
           "essai_90": 8, "essai_coupe": 8, "env2": 8, "env2_couleur": 8,
           "env2_reglee": 8}

#: centre de la face superieure de chaque piece, en pixels, releve A L'OEIL
#: sur l'image infrarouge ou couleur (a 10 px pres environ)
ANNOTATIONS = {
    "scene4": [(345, 245), (283, 307), (357, 300), (415, 300)],
    "scene7": [(313, 245), (382, 226), (250, 300), (314, 316), (379, 298),
               (437, 281), (400, 353), (300, 370)],
    "scene9": [(403, 160), (338, 186), (283, 222), (340, 250), (405, 230),
               (458, 213), (333, 296), (425, 285)],
    "essai_50": [(365, 248), (440, 230), (297, 295), (357, 322), (435, 320),
                 (495, 300), (450, 365), (348, 360)],
    "essai_90": [(362, 248), (437, 230), (292, 300), (370, 320), (430, 310),
                 (490, 290), (450, 360), (355, 358)],
    "env2_reglee": [(325, 322), (365, 352), (415, 305), (480, 372), (480, 283),
                    (515, 318), (535, 282), (405, 275)],
}


def apparier_image(pieces, centres, tolerance: float = 18.0):
    """(vrais positifs, faux positifs) : centre du contour reprojete a moins
    de `tolerance` pixels d'un centre annote, une piece au plus par centre."""
    if not pieces:
        return 0, 0
    det = np.array([p.contour.mean(axis=0) for p in pieces], float)
    ref = np.array(centres, float)
    d = np.linalg.norm(det[:, None] - ref[None], axis=2)
    paires = sorted((d[i, j], i, j) for i in range(len(det)) for j in range(len(ref)))
    pd, pr = set(), set()
    for dd, i, j in paires:
        if dd <= tolerance and i not in pd and j not in pr:
            pd.add(i)
            pr.add(j)
    return len(pd), len(det) - len(pd)


def scenes(config: str, nombre: int, graine: int = 0, empilees: int = 0):
    c = CONFIGS[config]
    for k in range(nombre):
        rng = np.random.default_rng(1000 * graine + k)
        if empilees:
            pcs = sy.placer(rng, 8, sy.TYPES, zone=c["zone"], touchantes=1,
                            empilees=empilees)
        else:
            pcs = sy.placer(rng, 8, sy.TYPES, zone=c["zone"], touchantes=3)
        bruit = sy.Bruit(reflet=c.get("reflet", False), lissage=c.get("lissage", 5))
        yield sy.scene(pcs, c["cam"], bruit, graine=k, couleur=c["couleur"],
                       motif=c["motif"], table=c["table"])


def evaluer_synthetique(nombre: int, params: dict, configs, methodes):
    lignes = []
    for config in configs:
        bilans = {m: sy.Bilan() for m in methodes}
        temps = {m: 0.0 for m in methodes}
        for sc, verite in scenes(config, nombre):
            for m in methodes:
                t = time.time()
                if m == "taches":
                    res = vision.detecter(sc, sy.TYPES)
                else:
                    res = vision_modele.detecter(sc, sy.TYPES, **params)
                temps[m] += time.time() - t
                bilans[m].ajouter(sy.comparer(res.pieces, verite))
        for m in methodes:
            r = bilans[m].resume()
            lignes.append((config, m, r, temps[m] / nombre))
    print("\nSCENES SYNTHETIQUES ({} par configuration, 8 pieces dont 3 en "
          "contact)".format(nombre))
    print("{:8s} {:7s} {:>6s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s} {:>6s} {:>6s} "
          "{:>6s} {:>6s} {:>6s} {:>5s}".format(
              "config", "methode", "rappel", "faux", "fines", "contact",
              "pos_med", "pos_p90", "ang_md", "ang_90", "z_med", "type", "face", "s"))
    for config, m, r, t in lignes:
        print("{:8s} {:7s} {:6.2f} {:5d} {:6.2f} {:7.2f} {:7.1f} {:7.1f} {:6.1f} "
              "{:6.1f} {:6.1f} {:6.2f} {:6.2f} {:5.1f}".format(
                  config, m, r["rappel"], r["faux"], r["rappel_fines"],
                  r["rappel_contact"], r["pos_med"], r["pos_p90"], r["ang_med"],
                  r["ang_p90"], r["z_med"], r["type"], r["face"], t))
    return lignes


def evaluer_empilement(nombre: int, params: dict, configs, detecteur=None,
                       empilees: int = 3):
    """Scenes avec empilements (voir l'en-tete), methode par modele."""
    detecteur = detecteur or vision_modele.detecter
    lignes = []
    for config in configs:
        bilan = sy.Bilan()
        t0 = time.time()
        for sc, verite in scenes(config, nombre, graine=7, empilees=empilees):
            res = detecteur(sc, sy.TYPES, **params)
            bilan.ajouter(sy.comparer(res.pieces, verite))
        lignes.append((config, bilan.resume(), (time.time() - t0) / nombre, bilan))
    print("\nSCENES AVEC EMPILEMENTS ({} par configuration, 8 pieces dont {} "
          "posees sur une autre)".format(nombre, empilees))
    print("{:8s} {:>6s} {:>4s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} "
          "{:>6s} {:>6s} {:>6s} {:>5s}".format(
              "config", "rappel", "faux", "empil", "port", "tf", "tf_emp", "supp",
              "s_emp", "pos", "p_emp", "p_port", "ordre", "1re", "s"))
    for config, r, t, _ in lignes:
        print("{:8s} {:6.2f} {:4d} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} "
              "{:6.1f} {:6.1f} {:6.1f} {:6.2f} {:6.2f} {:5.1f}".format(
                  config, r["rappel"], r["faux"], r["rappel_empilees"],
                  r["rappel_porteuses"], min(r["type"], r["face"]),
                  r["type_face_empilees"], r["support"], r["support_empilees"],
                  r["pos_med"], r["pos_med_empilees"], r["pos_med_porteuses"],
                  r["ordre_ok"], r["premiere_libre"], t))
    print("  (empil, port : rappel des pieces empilees, des porteuses ; tf : type "
          "et face justes ;\n   supp : support juste ; pos : erreur de position "
          "mediane, mm ; ordre : scenes sans\n   porteuse prise avant sa piece ; "
          "1re : scenes dont la premiere prise est libre)")
    return lignes


class OracleMasques:
    """Un faux reseau de segmentation qui rend les masques EXACTS des pieces
    (partie visible de chacune, d'apres la verite de la scene synthetique) :
    on mesure ainsi l'etage geometrique de perception.py (ajustement par
    masque, piles) seul, comme avec un reseau parfait."""

    def __init__(self, verite, types, aire_min: int = 30):
        self.names = {k: n for k, n in enumerate(sorted(types))}
        idx = {n: k for k, n in self.names.items()}
        masques, cls = [], []
        for i, p in enumerate(verite.pieces):
            m = verite.objet == i
            if m.sum() >= aire_min:
                masques.append(m)
                cls.append(idx[p.type])
        self.masques = np.array(masques, bool).reshape(-1, *verite.objet.shape)
        self.cls = np.array(cls, float)

    def predict(self, image, **kw):
        import types as _t

        class T:
            def __init__(self, a):
                self.a = a

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        n = len(self.masques)
        return [_t.SimpleNamespace(
            masks=_t.SimpleNamespace(data=T(self.masques.astype(np.float32))) if n else None,
            boxes=_t.SimpleNamespace(cls=T(self.cls), conf=T(np.full(n, 0.9))))]


def evaluer_masques(nombre: int, configs, empilees: int = 3):
    """Perception (reseau + geometrie) sur les scenes empilees, le reseau
    remplace par OracleMasques."""
    import perception
    print("\nETAGE GEOMETRIQUE DE perception.py, MASQUES PARFAITS ({} scenes, {} "
          "pieces empilees sur 8)".format(nombre, empilees))
    print("{:8s} {:>6s} {:>4s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} "
          "{:>6s}".format("config", "rappel", "faux", "empil", "port", "tf", "tf_emp",
                          "supp", "s_emp", "pos", "ordre", "1re"))
    for config in configs:
        bilan = sy.Bilan()
        for sc, verite in scenes(config, nombre, graine=7, empilees=empilees):
            gris = np.full(sc.profondeur.shape + (3,), 128, np.uint8)
            capture = {"profondeur": sc.profondeur.astype(np.float32), "K": sc.K,
                       "couleur": gris}
            p = perception.Perception(types=sy.TYPES, modele=False, raffinement=False)
            p.reseau, p.imgsz = OracleMasques(verite, sy.TYPES), 864
            bilan.ajouter(sy.comparer(p.detecter(capture), verite))
        r = bilan.resume()
        print("{:8s} {:6.2f} {:4d} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.1f} "
              "{:6.2f} {:6.2f}".format(config, r["rappel"], r["faux"], r["rappel_empilees"],
                                       r["rappel_porteuses"], min(r["type"], r["face"]),
                                       r["type_face_empilees"], r["support"],
                                       r["support_empilees"], r["pos_med"], r["ordre_ok"],
                                       r["premiere_libre"]))


class ReseauDegrade:
    """
    Un reseau de segmentation ABIME, pour mesurer ce que la geometrie
    garantit quand le reseau se trompe (fond inconnu, eclairage) : autour
    d'un vrai reseau (YOLO) ou d'OracleMasques, chaque masque est perdu
    avec la probabilite `manque`, deux masques voisins sont FUSIONNES avec
    la probabilite `fusion` (deux pieces qui se touchent vues comme une),
    et `faux` masques sont ajoutes sur le FOND (taches de la taille d'une
    piece, au hasard dans le bas de l'image : nappe, motif, ombre).
    """

    def __init__(self, reseau, manque=0.3, fusion=0.3, faux=2, graine=0):
        self.reseau = reseau
        self.names = reseau.names
        self.manque, self.fusion, self.faux = manque, fusion, faux
        self.rng = np.random.default_rng(graine)

    def predict(self, image, **kw):
        import types as _t
        import cv2
        r = self.reseau.predict(image, **kw)[0]
        H, W = image.shape[:2]
        if r.masks is None:
            m = np.zeros((0, H, W), bool)
            cls = np.zeros(0)
            conf = np.zeros(0)
        else:
            m = r.masks.data.cpu().numpy() > 0.5
            cls = r.boxes.cls.cpu().numpy().astype(float)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            if m.shape[1:] != (H, W):
                m = np.array([cv2.resize(x.astype(np.uint8), (W, H)) > 0 for x in m]).reshape(-1, H, W)
        garde = self.rng.random(len(m)) >= self.manque
        m, cls, conf = m[garde], cls[garde], conf[garde]
        # fusions de masques voisins
        m = list(m)
        cls, conf = list(cls), list(conf)
        i = 0
        while i < len(m):
            if self.rng.random() < self.fusion:
                gros = cv2.dilate(m[i].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                for j in range(i + 1, len(m)):
                    if (gros & m[j]).any():
                        m[i] = m[i] | m[j]
                        del m[j], cls[j], conf[j]
                        break
            i += 1
        for _ in range(self.faux):
            f = np.zeros((H, W), np.uint8)
            c = (int(self.rng.uniform(0.1, 0.9) * W), int(self.rng.uniform(0.5, 0.95) * H))
            cv2.ellipse(f, c, (int(self.rng.uniform(20, 50)), int(self.rng.uniform(6, 15))),
                        float(self.rng.uniform(0, 180)), 0, 360, 1, -1)
            m.append(f > 0)
            cls.append(float(self.rng.integers(len(self.names))))
            conf.append(0.6)

        class T:
            def __init__(self, a):
                self.a = a

            def cpu(self):
                return self

            def numpy(self):
                return self.a

        n = len(m)
        data = np.array(m, np.float32).reshape(n, H, W)
        return [_t.SimpleNamespace(
            masks=_t.SimpleNamespace(data=T(data)) if n else None,
            boxes=_t.SimpleNamespace(cls=T(np.array(cls, float)), conf=T(np.array(conf, float))))]


#: scenes de l'evaluation de l'affinage : (nom, arguments de synthese.placer)
SCENES_RAFFINEMENT = {
    "contact": dict(touchantes=3),
    "piles": dict(touchantes=1, empilees=3),
    "inclinees": dict(touchantes=1, empilees=1, inclinees=3),
}

#: methodes comparees : (nom, masques, affinage)
METHODES_RAFFINEMENT = (("geometrie", None, False), ("geometrie+raff", None, True),
                        ("masques", "oracle", False), ("masques+raff", "oracle", True),
                        ("degrades", "degrade", False), ("degrades+raff", "degrade", True))


def _perception_synthetique(sc, verite, masques, affinage, reglages=None):
    """Perception sur une scene synthetique : geometrie seule, ou un reseau
    remplace par les masques exacts (oracle) ou abimes (degrade)."""
    import perception
    capture = {"profondeur": sc.profondeur.astype(np.float32), "K": sc.K}
    if sc.infrarouge is not None:
        capture["infrarouge"] = sc.infrarouge
    if masques is not None:
        capture["couleur"] = np.full(sc.profondeur.shape + (3,), 128, np.uint8)
    p = perception.Perception(types=sy.TYPES, modele=False, raffinement=affinage,
                              reglages=reglages)
    if masques == "oracle":
        p.reseau, p.imgsz = OracleMasques(verite, sy.TYPES), 864
    elif masques == "degrade":
        p.reseau, p.imgsz = ReseauDegrade(OracleMasques(verite, sy.TYPES), graine=7), 864
    t = time.perf_counter()
    pieces = p.detecter(capture)
    return pieces, time.perf_counter() - t, p.durees.get("raffinement", 0.0)


def evaluer_raffinement(nombre: int, configs, types_scenes=None, methodes=None,
                        reglages=None, titre=""):
    """Avant / apres l'affinage (raffinement.py), sur des scenes synthetiques
    de 8 pieces : en contact, avec piles, avec pieces inclinees ; geometrie
    seule, masques parfaits, masques abimes (ReseauDegrade)."""
    methodes = [m for m in METHODES_RAFFINEMENT if methodes is None or m[0] in methodes]
    types_scenes = types_scenes or list(SCENES_RAFFINEMENT)
    print("\nAFFINAGE DE LA SCENE {}({} scenes de 8 pieces par cas)".format(titre, nombre))
    print("{:8s} {:9s} {:15s} {:>5s} {:>4s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} "
          "{:>5s} {:>5s} {:>5s} {:>5s} {:>5s}".format(
              "config", "scenes", "methode", "rapp", "faux", "pos", "lacet", "incl", "tf",
              "supp", "r_inc", "i_inc", "s_inc", "r_emp", "ordre", "t", "t_raf"))
    lignes = []
    for config in configs:
        c = CONFIGS[config]
        for ts in types_scenes:
            bilans = {m[0]: sy.Bilan() for m in methodes}
            temps = {m[0]: [0.0, 0.0] for m in methodes}
            for k in range(nombre):
                rng = np.random.default_rng(900 + k)
                pcs = sy.placer(rng, 8, sy.TYPES, zone=c["zone"], **SCENES_RAFFINEMENT[ts])
                bruit = sy.Bruit(reflet=c.get("reflet", False), lissage=c.get("lissage", 5))
                sc, verite = sy.scene(pcs, c["cam"], bruit, graine=k, couleur=c["couleur"],
                                      motif=c["motif"], table=c["table"])
                for nom, masques, aff in methodes:
                    pieces, dt, dt_r = _perception_synthetique(sc, verite, masques, aff, reglages)
                    bilans[nom].ajouter(sy.comparer(pieces, verite))
                    temps[nom][0] += dt
                    temps[nom][1] += dt_r
            for nom, _, _ in methodes:
                r = bilans[nom].resume()
                lignes.append((config, ts, nom, r, temps[nom][0] / nombre))
                print("{:8s} {:9s} {:15s} {:5.2f} {:4d} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} "
                      "{:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.1f} {:5.1f}".format(
                          config, ts, nom, r["rappel"], r["faux"], r["pos_med"], r["lacet_med"],
                          r["incl_med"], min(r["type"], r["face"]), r["support"],
                          r["rappel_inclinees"], r["incl_med_inclinees"], r["support_inclinees"],
                          r["rappel_empilees"], r["ordre_ok"], temps[nom][0] / nombre,
                          temps[nom][1] / nombre), flush=True)
    print("  (rapp : rappel ; pos : erreur de position mediane, mm ; lacet, incl : erreurs "
          "medianes, deg ;\n   tf : type et face justes ; supp : support juste ; r_inc, i_inc, "
          "s_inc : rappel, erreur\n   d'inclinaison (deg) et support juste des pieces inclinees ; "
          "r_emp : rappel des pieces\n   empilees ; ordre : scenes sans porteuse prise avant sa "
          "piece ; t, t_raf : s par scene,\n   total et affinage seul)")
    return lignes


#: images du jeu de donnees reel annote (apprentissage/dataset) : celles de
#: VALIDATION (exporter.en_validation, 20 %), que le reseau n'a pas apprises
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apprentissage", "dataset")


def images_validation(dataset=DATASET):
    import json
    import zlib
    noms = []
    dossier = os.path.join(dataset, "annotations")
    if not os.path.isdir(dossier):
        return noms
    for f in sorted(os.listdir(dossier)):
        n = f[:-5]
        with open(os.path.join(dossier, f)) as fi:
            a = json.load(fi)
        if a.get("statut") == "valide" and (zlib.crc32(n.encode()) % 1000) / 1000.0 < 0.2:
            noms.append(n)
    return noms


def comparer_masques(pieces, T_ct, vue, annotation, types):
    """Projette la scene estimee dans l'image couleur (rendu z-buffer :
    partie VISIBLE de chaque piece) et l'apparie aux masques annotes (IoU
    >= 0.5, glouton) : (annotees, predites, appariees, IoU des paires,
    classes justes)."""
    import raffinement as rf
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "apprentissage"))
    import commun
    T_tc = np.linalg.inv(T_ct)
    bs = [b for b in (rf.boite_de_piece(p, T_tc, types) for p in pieces) if b is not None]
    lab = rf.masque_visible(bs, T_ct, vue)
    W, H = annotation["taille"]
    vrais = [commun.polygones_vers_masque(i["polygones"], (W, H)) for i in annotation["instances"]]
    cls = [i["classe"] for i in annotation["instances"]]
    preds = [lab == k for k in range(len(bs))]
    paires = []
    for i, m in enumerate(preds):
        if not m.any():
            continue
        for j, v in enumerate(vrais):
            inter = np.count_nonzero(m & v)
            if inter:
                paires.append((inter / np.count_nonzero(m | v), i, j))
    pi, pj, ious, bon = set(), set(), [], 0
    for iou, i, j in sorted(paires, reverse=True):
        if iou < 0.5:
            break
        if i in pi or j in pj:
            continue
        pi.add(i)
        pj.add(j)
        ious.append(iou)
        bon += int(bs[i].type == cls[j])
    return len(vrais), sum(1 for m in preds if m.any()), len(pi), ious, bon


#: methodes sur les images reelles : (nom, reseau, affinage)
METHODES_REEL = (("geometrie", "aucun", False), ("geometrie+raff", "aucun", True),
                 ("yolo", "yolo", False), ("yolo+raff", "yolo", True),
                 ("yolo_degrade", "degrade", False), ("yolo_degrade+raff", "degrade", True))


def evaluer_reel_annote(nombre=None, methodes=None, reglages=None):
    """Les images de validation du jeu reel annote : rappel, precision, IoU
    des masques (scene estimee projetee dans l'image couleur), classe."""
    import json
    import perception
    noms = images_validation()
    if nombre:
        noms = noms[:nombre]
    if not noms:
        print("\n(pas de jeu reel annote)")
        return
    methodes = [m for m in METHODES_REEL if methodes is None or m[0] in methodes]
    if not os.path.exists(perception.MODELE):
        methodes = [m for m in methodes if m[1] == "aucun"]
    res = {m[0]: [0, 0, 0, [], 0, 0.0] for m in methodes}
    print("\nJEU REEL ANNOTE ({} images de validation)".format(len(noms)))
    for n in noms:
        cap = perception.charger_capture(os.path.join(DATASET, "brut", n + ".npz"))
        with open(os.path.join(DATASET, "annotations", n + ".json")) as f:
            ann = json.load(f)
        vue = perception.vue_couleur(cap)
        T_ct = perception.analyser_table(np.asarray(cap["profondeur"], float),
                                         np.asarray(cap["K"], float)).T_ct
        for nom, reseau, aff in methodes:
            p = perception.Perception(modele=None if reseau != "aucun" else False,
                                      raffinement=aff, reglages=reglages)
            if reseau == "degrade":
                p.reseau = ReseauDegrade(p.reseau, graine=int(n))
            t = time.perf_counter()
            pieces = p.detecter(cap)
            dt = time.perf_counter() - t
            v, npred, tp, ious, bon = comparer_masques(pieces, T_ct, vue, ann, p.types)
            r = res[nom]
            r[0] += v
            r[1] += npred
            r[2] += tp
            r[3] += ious
            r[4] += bon
            r[5] += dt
    print("{:20s} {:>7s} {:>9s} {:>6s} {:>7s} {:>7s}".format(
        "methode", "rappel", "precision", "IoU", "classe", "s/img"))
    for nom, (v, npred, tp, ious, bon, dt) in res.items():
        print("{:20s} {:7.2f} {:9.2f} {:6.2f} {:7.2f} {:7.1f}".format(
            nom, tp / max(v, 1), tp / max(npred, 1), float(np.mean(ious)) if ious else np.nan,
            bon / max(tp, 1), dt / len(noms)), flush=True)
    return res


def evaluer_captures_raffinement(reglages=None):
    """Les captures de captures/ (centres releves a l'oeil) : perception sans
    reseau, avant / apres l'affinage."""
    import perception
    dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    print("\nCAPTURES (centres releves a l'oeil) : geometrie, avant / apres l'affinage")
    tot = {False: [0, 0, 0], True: [0, 0, 0]}
    for nom, centres in ANNOTATIONS.items():
        chemin = os.path.join(dossier, nom + ".npz")
        if not os.path.exists(chemin):
            continue
        types = {"": (16, 30, 68)} if nom == "scene4" else sy.TYPES
        cases = []
        for aff in (False, True):
            p = perception.Perception(types=types, modele=False, raffinement=aff,
                                      reglages=reglages)
            t = time.perf_counter()
            pieces = p.detecter(chemin)
            dt = time.perf_counter() - t
            # centres dans l'image de PROFONDEUR (celle des releves)
            pts = []
            for q in pieces:
                c = q.T_camera[:3, 3]
                K = perception.charger_capture(chemin)["K"]
                pts.append(np.array([[K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]]]))
            faux_pieces = [type("P", (), {"contour": x})() for x in pts]
            vp, fp = apparier_image(faux_pieces, centres)
            tot[aff][0] += vp
            tot[aff][1] += fp
            tot[aff][2] += len(centres)
            cases.append("{}: {} justes, {} fausses ({:.1f} s)".format(
                "apres" if aff else "avant", vp, fp, dt))
        print("  {:12s} {} posees   {}".format(nom, len(centres), "   ".join(cases)))
    for aff in (False, True):
        print("  total {} : {} justes sur {}, {} fausses".format(
            "apres" if aff else "avant", tot[aff][0], tot[aff][2], tot[aff][1]))


#: ablations : (nom, reglages modifies)
ABLATIONS = (("complet", {}),
             ("sans lissage", {"flou": False, "flou_auto": False}),
             ("sans physique", {"k_penetration": 0.0, "k_air": 0.0, "k_stabilite": 0.0}),
             ("sans a priori incl.", {"k_inclinaison": 0.0}),
             ("sans variantes", {"variantes": False}),
             ("sans repropositions", {"repropositions": 0}),
             ("sans masques", {"masques": 0.0, "classe": 0.0}))


def evaluer_ablation(nombre, configs, types_scenes=("inclinees", "piles")):
    """L'affinage sans chacun de ses termes (masques parfaits ou geometrie)."""
    import raffinement as rf
    for nom, kw in ABLATIONS:
        reg = rf.Reglages(**kw)
        meth = ["geometrie+raff"] if nom != "sans masques" else ["masques+raff"]
        if nom == "complet":
            meth = ["geometrie+raff", "masques+raff"]
        evaluer_raffinement(nombre, configs, list(types_scenes), meth, reg,
                            titre="- {} ".format(nom.upper()))


def evaluer_reel(params: dict, methodes):
    dossier = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    print("\nCAPTURES REELLES (pieces posees ; trouvees ; justes et fausses d'apres "
          "les centres releves a l'oeil ; ecart moyen des dimensions, mm)")
    for nom, attendu in ATTENDU.items():
        chemin = os.path.join(dossier, nom + ".npz")
        if not os.path.exists(chemin):
            continue
        sc = vision.charger(chemin)
        dims = (16, 30, 68) if nom == "scene4" else sy.TYPES
        cases = []
        for m in methodes:
            if m == "taches":
                res = vision.detecter(sc, dims)
            else:
                res = vision_modele.detecter(sc, dims, **params)
            ecart = np.mean([p.ecart for p in res.pieces]) if res.pieces else np.nan
            if nom in ANNOTATIONS:
                vp, fp = apparier_image(res.pieces, ANNOTATIONS[nom])
                cases.append("{}: {:2d} trouvees, {} justes, {} fausses, ecart {:4.1f}"
                             .format(m, len(res.pieces), vp, fp, ecart))
            else:
                cases.append("{}: {:2d} trouvees{:16s}ecart {:4.1f}"
                             .format(m, len(res.pieces), "", ecart))
        print("  {:12s} {} posees   {}".format(nom, attendu, "   ".join(cases)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scenes", type=int, default=10,
                    help="scenes synthetiques par configuration (defaut 10)")
    ap.add_argument("--config", action="append", choices=sorted(CONFIGS),
                    help="configuration(s) a evaluer (defaut : toutes)")
    ap.add_argument("--methode", action="append", choices=("taches", "modele"),
                    help="methode(s) a evaluer (defaut : les deux)")
    ap.add_argument("--empilement", action="store_true",
                    help="evaluer aussi des scenes avec des pieces empilees")
    ap.add_argument("--sans-reel", action="store_true")
    ap.add_argument("--sans-synthese", action="store_true")
    ap.add_argument("--param", action="append", default=[], metavar="NOM=VALEUR",
                    help="parametre de vision_modele.detecter, ex. seuil_score=0.6")
    ap.add_argument("--raffinement", action="store_true",
                    help="avant / apres l'affinage de la scene (raffinement.py) : synthese, "
                    "jeu reel annote, captures")
    ap.add_argument("--images", type=int, default=0,
                    help="avec --raffinement : nombre d'images reelles annotees (0 : toutes)")
    ap.add_argument("--ablation", action="store_true",
                    help="avec --raffinement : l'affinage sans chacun des termes du cout")
    args = ap.parse_args()
    if args.raffinement:
        configs = args.config or list(CONFIGS)
        if not args.sans_synthese:
            evaluer_raffinement(args.scenes, configs)
            if args.ablation:
                evaluer_ablation(args.scenes, configs)
        if not args.sans_reel:
            evaluer_reel_annote(args.images or None)
            evaluer_captures_raffinement()
        return
    params = {}
    for p in args.param:
        k, v = p.split("=")
        params[k] = float(v)
    methodes = args.methode or ["taches", "modele"]
    if not args.sans_synthese:
        evaluer_synthetique(args.scenes, params, args.config or list(CONFIGS), methodes)
    if args.empilement and "modele" in methodes:
        evaluer_empilement(args.scenes, params, args.config or list(CONFIGS))
        evaluer_masques(args.scenes, args.config or list(CONFIGS))
    if not args.sans_reel:
        evaluer_reel(params, methodes)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
