#!/usr/bin/env bash
# =============================================================================
# setup_ds_env.sh - Environnement Data Science GPU (projet Leyenda / CESI)
# Cible : VM Ubuntu 22.04 / 24.04 avec GPU NVIDIA (T4) et driver déjà installé.
# A lancer avec un utilisateur normal (pas root) qui a sudo :
#
#     chmod +x setup_ds_env.sh && ./setup_ds_env.sh
#
# Optionnel : GIT_NAME="Prénom Nom" GIT_EMAIL="x@y.fr" ./setup_ds_env.sh
#             TF_VERSION=2.xx.x ./setup_ds_env.sh   (pour figer la version)
#             REPO_URL=https://.../projet.git ./setup_ds_env.sh   (clone le dépôt du groupe)
#   Sans GIT_NAME / GIT_EMAIL, le script les demande (si git n'est pas déjà configuré).
#
# Le script s'arrête avec un message clair si le GPU n'est pas utilisable.
# Journal complet : ~/setup_ds_env.log (à envoyer en cas de problème).
# =============================================================================
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/leyenda}"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_FILE="$HOME/setup_ds_env.log"
# Version validée sur la VM du projet (T4 vGPU) : la 2.21.0 ne voyait pas le GPU.
TF_VERSION="${TF_VERSION:-2.20.0}"
REPO_URL="${REPO_URL:-}"

exec > >(tee -a "$LOG_FILE") 2>&1
step() { echo -e "\n\033[1;34m==> $*\033[0m"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m $*"; }
die()  { echo -e "\033[1;31m[ERREUR]\033[0m $*" >&2; exit 1; }
trap 'die "Échec ligne $LINENO. Voir $LOG_FILE"' ERR

echo "=== $(date '+%F %T') ==="

# --- 1. Pré-requis (on s'arrête tout de suite si quelque chose manque) --------
step "Vérifications"
[ "$(id -u)" -ne 0 ] || die "Ne lance pas le script en root."
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Ubuntu requis (détecté : ${ID:-inconnu})."
sudo -v || die "Droits sudo requis."

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi ne fonctionne pas : le driver GPU n'est pas utilisable sur cette VM. Contacte l'administrateur."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ok "GPU détecté par le driver"

# --- 2. Paquets système -------------------------------------------------------
step "Paquets système"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential git curl wget ca-certificates gpg apt-transport-https \
  python3 python3-venv python3-pip python3-dev
curl -sS -o /dev/null -I --max-time 10 https://pypi.org || die "PyPI inaccessible depuis cette VM (proxy / pare-feu ?)."
ok "Paquets installés, PyPI accessible"

# --- 3. Git -------------------------------------------------------------------
step "Git"
# Identité : variables d'environnement, sinon question posée (si git n'est pas déjà configuré)
if [ -z "${GIT_NAME:-}" ] && [ -z "$(git config --global user.name || true)" ] && [ -t 0 ]; then
  read -r -p "Ton nom pour git (ex. Prénom Nom) : " GIT_NAME
fi
if [ -z "${GIT_EMAIL:-}" ] && [ -z "$(git config --global user.email || true)" ] && [ -t 0 ]; then
  read -r -p "Ton email pour git : " GIT_EMAIL
fi
[ -n "${GIT_NAME:-}" ]  && git config --global user.name  "$GIT_NAME"
[ -n "${GIT_EMAIL:-}" ] && git config --global user.email "$GIT_EMAIL"
git config --global init.defaultBranch main
ok "Git prêt ($(git config --global user.name || echo 'identité non configurée'))"

# Clone du dépôt du groupe (fait AVANT la création du projet, pour avoir un dossier vide)
if [ -n "$REPO_URL" ] && [ ! -d "$PROJECT_DIR/.git" ]; then
  if [ -d "$PROJECT_DIR" ] && [ -n "$(ls -A "$PROJECT_DIR" 2>/dev/null)" ]; then
    echo "[WARN] $PROJECT_DIR n'est pas vide : clone ignoré."
  else
    echo "Clonage de $REPO_URL (identifiants demandés si le dépôt est privé)"
    git clone "$REPO_URL" "$PROJECT_DIR" \
      || die "Clone impossible (URL, identifiants ou accès au dépôt). Corrige puis relance : le script est relançable."
    ok "Dépôt cloné dans $PROJECT_DIR"
  fi
fi

# --- 4. VS Code (ignoré s'il est déjà là ou si le dépôt Microsoft est bloqué) -
step "VS Code"
if command -v code >/dev/null 2>&1; then
  ok "VS Code déjà installé"
elif curl -sS -o /dev/null -I --max-time 10 https://packages.microsoft.com; then
  sudo install -d -m 0755 /etc/apt/keyrings
  wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | \
    sudo tee /etc/apt/keyrings/packages.microsoft.gpg >/dev/null
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | \
    sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y code
  ok "VS Code installé"
else
  echo "[WARN] packages.microsoft.com inaccessible : VS Code non installé (le reste continue)."
fi
if command -v code >/dev/null 2>&1; then
  for ext in ms-python.python ms-toolsai.jupyter; do
    code --install-extension "$ext" --force >/dev/null 2>&1 || echo "[WARN] extension $ext à installer depuis VS Code"
  done
fi

# --- 5. Projet + venv + librairies -------------------------------------------
step "Projet et environnement virtuel"
mkdir -p "$PROJECT_DIR"/{notebooks,data,models,src}
[ -f "$PROJECT_DIR/.gitignore" ] || printf '.venv/\n__pycache__/\n.ipynb_checkpoints/\ndata/\n*.h5\n*.keras\n*.npy\n' > "$PROJECT_DIR/.gitignore"
# Un dépôt cloné peut avoir son propre .gitignore : on s'assure que venv et réglages VS Code restent hors git
for pat in ".venv/" ".vscode/"; do
  grep -qxF "$pat" "$PROJECT_DIR/.gitignore" || printf '\n%s\n' "$pat" >> "$PROJECT_DIR/.gitignore"
done
[ -d "$PROJECT_DIR/.git" ] || git -C "$PROJECT_DIR" init -q
# Python standard du groupe : 3.10 (celui d'Ubuntu 22.04, donc de la VM). TensorFlow stable n'a pas de roues pour 3.14.
PYTHON_REQUIRED="${PYTHON_REQUIRED:-3.10}"
PYTHON="${PYTHON:-python3}"
if [ ! -d "$VENV_DIR" ]; then
  PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [ "$PY_VER" = "$PYTHON_REQUIRED" ] \
    || die "$PYTHON est en Python $PY_VER, le groupe utilise $PYTHON_REQUIRED. Utilise Ubuntu 22.04 (python3 = 3.10) ou lance : PYTHON=python3.10 bash setup_ds_env.sh"
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
VENV_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[ "$VENV_VER" = "$PYTHON_REQUIRED" ] \
  || die "Le venv existant ($VENV_DIR) utilise Python $VENV_VER au lieu de $PYTHON_REQUIRED. Supprime-le (rm -rf $VENV_DIR) puis relance."
python -m pip install --upgrade pip setuptools wheel

step "Librairies et TensorFlow GPU"
LOCK_FILE="${LOCK_FILE:-$PROJECT_DIR/requirements.lock.txt}"
if [ -f "$LOCK_FILE" ]; then
  # Le dépôt contient déjà les versions du groupe : on les installe à l'identique
  echo "requirements.lock.txt trouvé : installation des versions exactes du groupe."
  pip install -r "$LOCK_FILE"
  LOCK_USED=1
else
  pip install numpy pandas scipy scikit-learn matplotlib seaborn imageio pillow \
    opencv-python-headless tqdm jupyterlab ipykernel
  pip install "tensorflow[and-cuda]${TF_VERSION:+==$TF_VERSION}"
  LOCK_USED=0
fi
# TF 2.21 ne retrouve pas tout seul les libs CUDA installées dans le venv (nvidia/*/lib) :
# on les ajoute au chemin de recherche, pour ce script, pour le venv (activate) et pour le kernel Jupyter.
NVIDIA_LIBS="$(python -c 'import glob, site; print(":".join(sorted(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))))')"
if [ -n "$NVIDIA_LIBS" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if ! grep -q "LEYENDA_NVIDIA_LIBS" "$VENV_DIR/bin/activate"; then
    cat >> "$VENV_DIR/bin/activate" <<'ACTIVATE_PATCH'

# LEYENDA_NVIDIA_LIBS : libs CUDA pip du venv visibles par TensorFlow
for _d in "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
unset _d
ACTIVATE_PATCH
  fi
fi
python -m ipykernel install --user --name leyenda --display-name "Python (Leyenda)"
if [ -n "$NVIDIA_LIBS" ]; then
  python - "$LD_LIBRARY_PATH" <<'KERNEL_PATCH'
import json, os, sys
from jupyter_client.kernelspec import KernelSpecManager
path = os.path.join(KernelSpecManager().find_kernel_specs()["leyenda"], "kernel.json")
spec = json.load(open(path))
spec.setdefault("env", {})["LD_LIBRARY_PATH"] = sys.argv[1]
json.dump(spec, open(path, "w"), indent=1)
KERNEL_PATCH
fi

# VS Code : sélectionne le venv automatiquement à l'ouverture du dossier
mkdir -p "$PROJECT_DIR/.vscode"
[ -f "$PROJECT_DIR/.vscode/settings.json" ] || cat > "$PROJECT_DIR/.vscode/settings.json" <<SETTINGS
{
  "python.defaultInterpreterPath": "$VENV_DIR/bin/python",
  "python.terminal.activateEnvironment": true
}
SETTINGS
if [ "$LOCK_USED" -eq 0 ]; then
  pip freeze > "$LOCK_FILE"
  ok "Versions figées dans requirements.lock.txt : à commiter pour que le groupe ait les mêmes."
fi
ok "Installation terminée"

# --- 6. Vérification GPU (le script échoue si TensorFlow n'utilise pas le GPU) -
step "Vérification TensorFlow + GPU"
python - <<'EOF' || {
import sys, time
import tensorflow as tf
print("TensorFlow :", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU vus par TensorFlow :", gpus)
if not gpus:
    sys.exit(1)
with tf.device("/GPU:0"):
    a = tf.random.normal((4000, 4000))
    t = time.time()
    c = tf.reduce_sum(a @ a).numpy()
    print(f"Calcul sur GPU OK ({time.time()-t:.2f}s)")
EOF
  echo
  echo -e "\033[1;31m[ECHEC GPU]\033[0m TensorFlow n'utilise pas le GPU alors que le driver répond."
  echo "Causes possibles : version de TensorFlow incompatible avec cette VM (la 2.21 l'était),"
  echo "ou profil vGPU / licence NVIDIA qui bloque CUDA."
  echo "Test pour départager : python -c \"import ctypes; cu=ctypes.CDLL('libcuda.so.1'); print(cu.cuInit(0))\""
  echo "(0 = CUDA fonctionne sur la VM, le problème vient alors de TensorFlow)."
  echo "Sinon, envoie ceci à l'administrateur de la VM :"
  nvidia-smi -q | grep -i -E "product (name|brand)|driver version|cuda version|vgpu|licen" || true
  echo "Journal complet : $LOG_FILE"
  exit 1
}

echo
ok "TOUT EST BON : environnement prêt avec GPU."

# Ouvre le projet dans VS Code (seulement si on a un bureau graphique)
if command -v code >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  nohup code "$PROJECT_DIR" >/dev/null 2>&1 &
  echo "VS Code s'ouvre sur $PROJECT_DIR."
else
  echo "Pour ouvrir le projet : cd $PROJECT_DIR && code ."
fi
echo "Dans VS Code : si besoin, kernel 'Python (Leyenda)' pour les notebooks (en haut à droite)."
echo "Pour un terminal : cd $PROJECT_DIR && source .venv/bin/activate"
