# environment.sh - Configure build environment

# Environment
export PROJECT_ROOT=$PWD
export PROJECT_NAME=$(basename $PROJECT_ROOT)

# Python versions
export VERSION=${VERSION:-0.1.0}
export PY_VERSION=$(echo $VERSION | sed 's/-/\.dev0+/')

# Jupyter
export JUPYTER_CONFIG_DIR=${PROJECT_ROOT}/.build/jupyter
export JUPYTER_DATA_DIR=${JUPYTER_CONFIG_DIR}
export JUPYTER_PLATFORM_DIRS=1

# Set mtimes to timestamp of latest commit if project has git repo
if [[ -d .git ]]; then
  export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
else
  unset SOURCE_DATE_EPOCH
fi

# Export variables to temporary .env
tmp_project_env=$(mktemp)

project_variables=(
  PROJECT_ROOT
  PROJECT_NAME
  VERSION
  PY_VERSION
  JUPYTER_CONFIG_DIR
  JUPYTER_DATA_DIR
  JUPYTER_PLATFORM_DIRS
  SOURCE_DATE_EPOCH
)

for v in "${project_variables[@]}"; do
  if [ -n "${BASH_VERSION:-}" ]; then
    # For bash
    echo "$v=${!v}" >> $tmp_project_env
  elif [ -n "${ZSH_VERSION:-}" ]; then
    # For zsh
    echo "$v=${(P)v}" >> $tmp_project_env
  fi
done

# Only update .env if they're different.
#   Note: Prevents parallel make processes from stepping on each other.

if [[ ! -f .env ]] || ! cmp -s .env $tmp_project_env; then
  echo "Updating .env"
  mv $tmp_project_env .env
fi
