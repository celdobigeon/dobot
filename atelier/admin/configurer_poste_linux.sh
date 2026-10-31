#!/usr/bin/env bash
#
# Configuration d'un poste Ubuntu de TP pour le simulateur Dobot Magician.
#
# RESERVE A L'ADMINISTRATEUR, une seule fois par poste :
#
#   sudo ./admin/configurer_poste_linux.sh
#
# Les etudiants n'ont ensuite besoin d'aucun droit particulier : ils
# lancent ./installer_linux.sh depuis leur compte.
#
# 1. paquets systeme : venv, bibliotheques X11/OpenGL dont Qt a besoin
# 2. port serie du robot accessible a TOUS les comptes (regle udev), sans
#    avoir a ajouter chaque etudiant au groupe dialout
# 3. ModemManager et brltty empeches de capturer le port du robot
# 4. camera RealSense accessible a tous les comptes (regle udev)
#
# Peut etre relance sans risque.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "A lancer en administrateur : sudo $0" >&2
    exit 1
fi

info() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# --- 1. Paquets systeme ----------------------------------------------------

info "Paquets systeme"
apt-get update
apt-get install -y \
    python3 python3-venv python3-pip git \
    libgl1 libegl1 libgl1-mesa-dri mesa-utils libglib2.0-0 libdbus-1-3 \
    libfontconfig1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 \
    libxcb-xfixes0 libxcb-shape0 libxcb-cursor0 libxcb-xkb1

# --- 2 et 3. Port serie ----------------------------------------------------
#
# Le Magician se presente par un pont USB-serie CH340 (1a86:7523), parfois
# CP210x (10c4:ea60).
#  - MODE 0666 : tout compte peut ouvrir le port, y compris un compte
#    etudiant cree apres coup ; pas de groupe dialout a gerer.
#  - ID_MM_DEVICE_IGNORE : ModemManager sonde sinon chaque nouveau port
#    serie par des commandes AT pendant plusieurs secondes.

info "Regle udev pour le port du robot"
REGLE=/etc/udev/rules.d/99-dobot-magician.rules
cat > "$REGLE" <<'REGLES'
# Dobot Magician : pont USB-serie (installe par admin/configurer_poste_linux.sh)
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
REGLES
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty || true
echo "installee : $REGLE"

# Camera RealSense (Intel, 8086) : sans cette regle, seul root peut
# l'ouvrir. Trois chemins d'acces selon ce que lit la bibliotheque : le
# peripherique USB lui-meme, les flux video du noyau (/dev/video*), et la
# centrale inertielle des modeles "i" (hidraw).
info "Regle udev pour la camera RealSense"
CAMERA=/etc/udev/rules.d/99-realsense.rules
cat > "$CAMERA" <<'REGLES'
# Camera Intel RealSense (installe par admin/configurer_poste_linux.sh)
SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", ATTRS{product}=="*RealSense*", MODE="0666"
KERNEL=="video*", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="8086", MODE="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="8086", MODE="0666"
REGLES
udevadm control --reload-rules
udevadm trigger || true
echo "installee : $CAMERA (rebrancher la camera si elle l'etait)"

# brltty (afficheurs braille) s'empare des CH340 : le port /dev/ttyUSB0
# apparait puis disparait aussitot. Sans afficheur braille, il ne sert a rien.
if dpkg -s brltty >/dev/null 2>&1; then
    info "Retrait de brltty (capture le port du Dobot)"
    apt-get remove -y brltty
fi

info "Poste configure. Debrancher puis rebrancher le robot s'il l'etait."
