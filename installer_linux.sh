#!/usr/bin/bash
# Installation pour un compte etudiant : AUCUN droit administrateur requis.
git pull
# Sans le paquet python3-venv (absent par defaut sur Ubuntu, et qu'un
# etudiant ne peut pas installer), "python3 -m venv" echoue : on cree alors
# l'environnement sans pip, puis on y installe pip a la main.
if ! python3 -m venv .venv; then
    echo "python3-venv absent : installation de pip dans .venv sans sudo"
    python3 -m venv --without-pip --clear .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py -o .venv/get-pip.py
    .venv/bin/python .venv/get-pip.py
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py --verifier
