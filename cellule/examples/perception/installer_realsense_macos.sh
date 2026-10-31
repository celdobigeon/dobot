#!/usr/bin/env bash
#
# Module Python de la RealSense (pyrealsense2 2.59.1) pour le .venv du
# projet, sous macOS (Apple Silicon).
#
#   ./examples/perception/installer_realsense_macos.sh
#
# Pourquoi ce script : Homebrew fournit la bibliotheque RealSense et ses
# outils (rs-enumerate-devices, realsense-viewer...), mais PAS le module
# Python ; et le module "pyrealsense2-macosx" de PyPI (2.54) plante sous
# macOS 15. On compile donc le module depuis les sources de la MEME
# version que Homebrew, pour le Python du .venv, puis on l'installe dans le
# .venv avec sa bibliotheque a cote de lui.
#
# A refaire si le .venv est recree. Duree : 10 a 15 minutes la premiere
# fois. Aucun sudo (sauf pour UTILISER la camera ensuite : sous macOS, les
# scripts qui l'ouvrent se lancent avec sudo).

set -euo pipefail

VERSION=2.59.1
PROJET="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$PROJET/.venv/bin/python"
TRAVAIL="$PROJET/.build-realsense"        # ignore par git

info() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

[ -x "$PY" ] || { echo "pas de .venv : lancez d'abord l'installation du projet" >&2; exit 1; }
command -v brew >/dev/null || { echo "Homebrew est requis : https://brew.sh" >&2; exit 1; }

info "Bibliotheque RealSense et outils de compilation (Homebrew)"
brew install librealsense cmake pkgconf libusb

info "Sources de librealsense $VERSION"
mkdir -p "$TRAVAIL"
if [ ! -d "$TRAVAIL/src" ]; then
    git clone -q --depth 1 --branch "v$VERSION" \
        https://github.com/realsenseai/librealsense.git "$TRAVAIL/src"
fi

info "Configuration (Python : $PY)"
PYROOT="$("$PY" -c 'import sys; print(sys.base_prefix)')"
cmake -S "$TRAVAIL/src" -B "$TRAVAIL/build" \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_CCACHE=OFF -DBUILD_WITH_OPENMP=OFF \
    -DCMAKE_CXX_STANDARD=17 -DCHECK_FOR_UPDATES=OFF -DENABLE_AI_ASSISTANT=OFF \
    -DENABLE_STATS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE="$PY" \
    -DPython_EXECUTABLE="$PY" -DPython_ROOT_DIR="$PYROOT" > "$TRAVAIL/configuration.log"

info "Compilation (10 a 15 minutes la premiere fois)"
cmake --build "$TRAVAIL/build" --target pyrealsense2 -j "$(sysctl -n hw.ncpu)" \
    > "$TRAVAIL/compilation.log" || { tail -20 "$TRAVAIL/compilation.log"; exit 1; }

info "Installation dans le .venv"
B="$TRAVAIL/build/Release"
TAG="$("$PY" -c 'import sys; print("cpython-%d%d" % sys.version_info[:2])')"
SO="$B/pyrealsense2.$VERSION.$TAG-darwin.so"
[ -f "$SO" ] || { echo "module introuvable : $SO" >&2; exit 1; }
SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
"$PY" -m pip uninstall -q -y pyrealsense2-macosx pyrealsense2 2>/dev/null || true
D="$SP/pyrealsense2"
rm -rf "$D"
mkdir -p "$D"
COURTE="${VERSION%.*}"                    # 2.59
cp "$SO" "$D/pyrealsense2.$TAG-darwin.so"
cp "$B/librealsense2.$VERSION.dylib" "$D/librealsense2.$COURTE.dylib"
# le module cherchait sa bibliotheque dans le dossier de compilation : on
# la lui fait trouver A COTE de lui, puis on re-signe (obligatoire sur
# Apple Silicon apres toute modification d'un binaire)
install_name_tool -delete_rpath "$B" "$D/pyrealsense2.$TAG-darwin.so" 2>/dev/null || true
install_name_tool -add_rpath @loader_path "$D/pyrealsense2.$TAG-darwin.so"
install_name_tool -id "@rpath/librealsense2.$COURTE.dylib" "$D/librealsense2.$COURTE.dylib"
codesign --force -s - "$D/librealsense2.$COURTE.dylib" "$D/pyrealsense2.$TAG-darwin.so"
cat > "$D/__init__.py" <<EOF
# Module RealSense $VERSION, compile depuis les sources par
# examples/perception/installer_realsense_macos.sh (Homebrew ne le fournit pas)
from .pyrealsense2 import *  # noqa: F401,F403
__version__ = "$VERSION"
EOF

info "Verification"
"$PY" -c "import pyrealsense2 as rs; print('pyrealsense2', rs.__version__, 'charge')"
echo
echo "Pour utiliser la camera (macOS) : sudo .venv/bin/python examples/perception/capture_realsense.py"
