"""Verification de la pile d'apprentissage de la perception, sans camera.

    .venv/bin/python examples/perception/verifier_installation.py
    .venv/bin/python examples/perception/verifier_installation.py --telecharger
    .venv/bin/python examples/perception/verifier_installation.py --sortie /tmp/essai

Affiche les versions (numpy, OpenCV, PyTorch, ultralytics), le peripherique
de calcul (mps, cuda ou cpu), la presence des poids dans
examples/perception/modeles/, puis fait une inference YOLO11 et SAM 2.1 sur
une image de test : la capture captures/env2_reglee.npz si elle existe,
sinon une image fournie avec ultralytics.

Ne leve jamais d'exception : chaque element manquant est signale avec ce
qu'il faut faire. Code de retour 0 si tout va bien, 1 sinon.

--telecharger : telecharge les poids absents (environ 270 Mo en tout).
--sortie DOSSIER : enregistre les images annotees dans DOSSIER.
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

ICI = Path(__file__).resolve().parent
MODELES = ICI / "modeles"
CAPTURE = ICI / "captures" / "env2_reglee.npz"

POIDS_YOLO = ("yolo11n-seg.pt", "yolo11s-seg.pt")
POIDS_SAM = ("sam2.1_t.pt", "sam2.1_b.pt")

# Points-invites pour SAM sur la capture env2_reglee (x, y en pixels) :
# deux blocs de bois qui se touchent, a gauche sur la nappe.
INVITES_CAPTURE = [[330, 335], [365, 368]]

INSTALLER = ".venv/bin/pip install -r examples/perception/requirements.txt"

problemes = []


def signaler(message):
    problemes.append(message)
    print("  !! " + message)


def version(nom_module):
    try:
        module = importlib.import_module(nom_module)
    except Exception as erreur:  # ImportError, ou bibliotheque native cassee
        signaler("{} introuvable ou casse ({}). A faire : {}".format(
            nom_module, erreur, INSTALLER))
        return None
    print("  {:<12} {}".format(nom_module, getattr(module, "__version__", "?")))
    return module


def verifier_versions():
    print("Versions :")
    print("  {:<12} {}".format("python", sys.version.split()[0]))
    np = version("numpy")
    cv2 = version("cv2")
    torch = version("torch")
    ultralytics = version("ultralytics")
    if cv2 is not None:
        manque = [f for f in ("createLineSegmentDetector", "HoughLinesP")
                  if not hasattr(cv2, f)]
        if manque:
            signaler("OpenCV sans {} : plusieurs distributions d'OpenCV "
                     "concurrentes ? A faire : .venv/bin/pip list | grep -i "
                     "opencv, n'en garder qu'une (opencv-python)".format(
                         ", ".join(manque)))
    return np, cv2, torch, ultralytics


def choisir_peripherique(torch):
    print("Peripherique de calcul :")
    if torch is None:
        print("  (PyTorch absent)")
        return None
    if torch.cuda.is_available():
        peripherique = "cuda"
        print("  cuda :", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        peripherique = "mps"
        print("  mps (GPU Apple)")
    else:
        peripherique = "cpu"
        print("  cpu (aucun GPU utilisable : ca marche, mais plus lentement)")
    try:
        a = torch.randn(512, 512, device=peripherique)
        (a @ a).sum().item()
        print("  petit calcul sur {} : ok".format(peripherique))
    except Exception as erreur:
        signaler("calcul impossible sur {} ({}) ; on repasse sur cpu".format(
            peripherique, erreur))
        peripherique = "cpu"
    return peripherique


def verifier_poids(telecharger):
    print("Poids (dans {}) :".format(MODELES))
    presents = {}
    for nom in POIDS_YOLO + POIDS_SAM:
        chemin = MODELES / nom
        if chemin.exists():
            print("  {:<16} {:6.1f} Mo".format(nom, chemin.stat().st_size / 1e6))
            presents[nom] = True
        elif telecharger:
            print("  {:<16} absent : sera telecharge".format(nom))
            presents[nom] = True
        else:
            print("  {:<16} absent".format(nom))
            presents[nom] = False
    if not all(presents.values()):
        signaler("poids absents. A faire : relancer avec --telecharger")
    return presents


def image_de_test(np, cv2):
    if CAPTURE.exists():
        try:
            donnees = np.load(CAPTURE)
            if "couleur" in donnees.files:
                return donnees["couleur"], CAPTURE.name, INVITES_CAPTURE
        except Exception as erreur:
            print("  capture illisible ({}) ; image d'ultralytics".format(erreur))
    from ultralytics.utils import ASSETS
    image = cv2.imread(str(ASSETS / "bus.jpg"))   # BGR, comme la camera
    if image is None:
        return None, None, None
    h, l = image.shape[:2]
    return image, "bus.jpg (ultralytics)", [[l // 2, h // 2]]


def chronometrer(torch, peripherique, fonction, repetitions=3):
    """Mediane en ms, apres un premier appel de chauffe (non compte)."""
    resultat = fonction()
    durees = []
    for _ in range(repetitions):
        debut = time.perf_counter()
        resultat = fonction()
        if peripherique == "mps":
            torch.mps.synchronize()
        elif peripherique == "cuda":
            torch.cuda.synchronize()
        durees.append(time.perf_counter() - debut)
    durees.sort()
    return resultat, 1000 * durees[len(durees) // 2]


def essayer_modeles(np, cv2, torch, peripherique, presents, sortie):
    print("Inference :")
    image, nom_image, invites = image_de_test(np, cv2)
    if image is None:
        signaler("aucune image de test")
        return
    print("  image : {} ({}x{})".format(nom_image, image.shape[1], image.shape[0]))
    from ultralytics import SAM, YOLO
    MODELES.mkdir(exist_ok=True)

    for nom in POIDS_YOLO:
        if not presents[nom]:
            continue
        try:
            modele = YOLO(str(MODELES / nom))
            r, ms = chronometrer(torch, peripherique, lambda: modele.predict(
                image, device=peripherique, verbose=False))
            print("  {:<16} {:7.1f} ms  {} objet(s) (classes COCO : il "
                  "faudra un modele entraine sur nos pieces)".format(
                      nom, ms, len(r[0].boxes)))
            if sortie:
                cv2.imwrite(str(sortie / ("yolo_" + nom[:-3] + ".png")), r[0].plot())
        except Exception as erreur:
            signaler("{} : echec ({})".format(nom, erreur))

    for nom in POIDS_SAM:
        if not presents[nom]:
            continue
        try:
            modele = SAM(str(MODELES / nom))
            r, ms = chronometrer(torch, peripherique, lambda: modele(
                image, points=invites, labels=[1] * len(invites),
                device=peripherique, verbose=False))
            n = 0 if r[0].masks is None else len(r[0].masks)
            print("  {:<16} {:7.1f} ms  {} masque(s) pour {} point(s)-invite(s)"
                  .format(nom, ms, n, len(invites)))
            if sortie and r[0].masks is not None:
                cv2.imwrite(str(sortie / ("sam_" + nom[:-3] + ".png")),
                            superposer(np, cv2, image, r[0].masks.data.cpu().numpy()))
        except Exception as erreur:
            signaler("{} : echec ({})".format(nom, erreur))


def superposer(np, cv2, image, masques):
    """Colorie chaque masque sur une copie de l'image (BGR)."""
    resultat = image.copy()
    couleurs = [(60, 180, 255), (255, 120, 60), (90, 220, 90), (200, 80, 220)]
    for i, masque in enumerate(masques):
        masque = cv2.resize(masque.astype(np.uint8), image.shape[1::-1],
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        c = np.array(couleurs[i % len(couleurs)], dtype=np.float32)
        resultat[masque] = (0.45 * resultat[masque] + 0.55 * c).astype(np.uint8)
    return resultat


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--telecharger", action="store_true",
                        help="telecharger les poids absents")
    parser.add_argument("--sortie", type=Path,
                        help="dossier ou enregistrer les images annotees")
    args = parser.parse_args()

    np, cv2, torch, ultralytics = verifier_versions()
    peripherique = choisir_peripherique(torch)
    presents = verifier_poids(args.telecharger)
    if None in (np, cv2, torch, ultralytics):
        print("Inference : sautee (modules manquants)")
    else:
        if args.sortie:
            args.sortie.mkdir(parents=True, exist_ok=True)
        try:
            essayer_modeles(np, cv2, torch, peripherique, presents, args.sortie)
        except Exception as erreur:
            signaler("inference impossible ({})".format(erreur))

    print()
    if problemes:
        print("{} probleme(s) :".format(len(problemes)))
        for p in problemes:
            print("  - " + p)
        return 1
    print("Tout est en place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
