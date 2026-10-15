#!/usr/bin/env bash
#
# Installation du simulateur Dobot Magician sous Linux.
#
#   ./installer_linux.sh
#
# 1. paquets systeme : Python, venv, bibliotheques X11/OpenGL dont Qt a besoin
# 2. environnement virtuel .venv avec les dependances de requirements.txt
# 3. acces au port serie USB (groupe dialout) pour piloter le robot reel
#
# Distributions reconnues : Debian/Ubuntu (apt), Fedora (dnf), Arch (pacman).
# Le script peut etre relance sans risque.

set -euo pipefail

cd "$(dirname "$0")"

info()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
alerte() { printf '\033[1;33m/!\\ %s\033[0m\n' "$*"; }

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# --- 1. Paquets systeme ----------------------------------------------------

info "Paquets systeme"
if command -v apt-get >/dev/null; then
    $SUDO apt-get update
    $SUDO apt-get install -y \
        python3 python3-venv python3-pip \
        libgl1 libegl1 libgl1-mesa-dri mesa-utils libglib2.0-0 libdbus-1-3 libfontconfig1 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 \
        libxcb-shape0 libxcb-cursor0 libxcb-xkb1
elif command -v dnf >/dev/null; then
    $SUDO dnf install -y \
        python3 python3-pip \
        mesa-libGL mesa-libEGL mesa-dri-drivers glx-utils glib2 dbus-libs fontconfig \
        libxkbcommon-x11 xcb-util-wm xcb-util-image xcb-util-keysyms \
        xcb-util-renderutil xcb-util-cursor
elif command -v pacman >/dev/null; then
    $SUDO pacman -S --needed --noconfirm \
        python python-pip \
        mesa mesa-utils glib2 dbus fontconfig \
        libxkbcommon-x11 xcb-util-wm xcb-util-image xcb-util-keysyms \
        xcb-util-renderutil xcb-util-cursor
else
    alerte "Gestionnaire de paquets non reconnu : installez Python 3.8+ avec"
    alerte "venv, et les bibliotheques xcb/OpenGL de Qt, puis relancez."
fi

# --- 2. Environnement virtuel ----------------------------------------------

info "Environnement virtuel .venv"
python3 - <<'EOF'
import sys
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 ou plus recent requis (trouve %d.%d)" % sys.version_info[:2])
EOF
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

info "Verification de l'installation"
.venv/bin/python main.py --verifier || alerte "La verification a signale un probleme (voir ci-dessus)."

# --- 3. Port serie (robot reel) ---------------------------------------------

info "Acces au port serie"
if getent group dialout >/dev/null; then
    GROUPE=dialout
elif getent group uucp >/dev/null; then   # Arch
    GROUPE=uucp
else
    GROUPE=""
fi
if [ -n "$GROUPE" ] && ! id -nG "$USER" | grep -qw "$GROUPE"; then
    $SUDO usermod -aG "$GROUPE" "$USER"
    alerte "$USER ajoute au groupe $GROUPE : fermez puis rouvrez la session"
    alerte "pour que l'acces a /dev/ttyUSB* soit effectif."
fi

# Sous Ubuntu, brltty (afficheurs braille) s'empare des adaptateurs CH340 du
# Dobot : le port /dev/ttyUSB0 apparait puis disparait aussitot.
if command -v brltty >/dev/null; then
    alerte "brltty est installe : il peut capturer le port USB du Dobot."
    alerte "Si /dev/ttyUSB0 n'apparait pas : sudo apt remove brltty"
fi

info "Termine. Pour lancer :"
echo "    .venv/bin/python main.py"
echo "    .venv/bin/python test_prise.py"
