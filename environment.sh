# environment.sh - Configure build environment

# Environment
export PROJECT_ROOT=$PWD
export PROJECT_NAME=stickshift

export VERSION=${VERSION:-0.1-dev}
export PY_VERSION=$(echo $VERSION | sed 's/-/\.dev0+/')

# Set mtimes to timestamp of latest commit if project has git repo
if [[ -d .git ]]; then
  export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
else
  unset SOURCE_DATE_EPOCH
fi

# Export variables to temporary project.env
tmp_project_env=$(mktemp)
project_variables=(
  PROJECT_ROOT
  PROJECT_NAME
  VERSION
  PY_VERSION
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

# Only update project.env if they're different.
#   Note: Prevents parallel make processes from stepping on each other.

if [[ ! -f project.env ]] || ! cmp -s project.env $tmp_project_env; then
  echo "Updating project.env"
  mv $tmp_project_env project.env
fi