################################################################################
# Makefile
################################################################################

################################################################################
# Settings
################################################################################

# Verify environment.sh
ifeq ($(strip $(PROJECT_ROOT)),)
$(error Environment not configured. Run `source environment.sh`)
endif

#-------------------------------------------------------------------------------
# Shell
#-------------------------------------------------------------------------------

# Bash
SHELL := /bin/bash
.SHELLFLAGS := -e -u -o pipefail -c

# Colors - Supports colorized messages
COLOR_H1=\033[38;5;12m
COLOR_OK=\033[38;5;02m
COLOR_COMMENT=\033[38;5;08m
COLOR_RESET=\033[0m

# EXCLUDE_SRC - Source patterns to ignore

EXCLUDE_SRC := __pycache__ \
			   .egg-info \
			   .ipynb_checkpoints
EXCLUDE_SRC := $(subst $(eval ) ,|,$(EXCLUDE_SRC))

#-------------------------------------------------------------------------------
# Commands
#-------------------------------------------------------------------------------

RM := rm -rf

#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

BUILD_DIR := .build

#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

VENV_ROOT := .venv
VENV := $(VENV_ROOT)/bin/activate

#-------------------------------------------------------------------------------
# Requirements
#-------------------------------------------------------------------------------

REQUIREMENTS := requirements.txt

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

DEPENDENCIES := $(BUILD_DIR)/deps.ts

#-------------------------------------------------------------------------------
# Packages
#-------------------------------------------------------------------------------

PACKAGES_DIR := $(BUILD_DIR)/packages
PACKAGES :=

# Package: stickshift

STICKSHIFT_PACKAGE_SRC := $(shell find src -type f | egrep -v '$(EXCLUDE_SRC)')
STICKSHIFT_PACKAGE_REQUIRES = $(STICKSHIFT_PACKAGE_SRC)
STICKSHIFT_PACKAGE := $(PACKAGES_DIR)/stickshift-$(PY_VERSION)-py3-none-any.whl

PACKAGES := $(PACKAGES) $(STICKSHIFT_PACKAGE)

#-------------------------------------------------------------------------------
# Site
#-------------------------------------------------------------------------------

SITE_SRC_DIR := site
SITE_SRC := $(SITE_SRC_DIR)/_config.yml \
            $(shell find $(SITE_SRC_DIR) -type f -name '*.md') \
			$(shell find $(SITE_SRC_DIR)/assets -type f)

#-------------------------------------------------------------------------------
# Posts
#-------------------------------------------------------------------------------

POSTS_SRC_DIR := posts
POSTS_BUILD_DIR := $(SITE_SRC_DIR)/_posts

# Map $(POSTS_SRC_DIR)/**/*.ipynb to $(POSTS_BUILD_DIR)/**/*.md
POSTS_SRC := $(shell find $(POSTS_SRC_DIR) -type f -name '*.ipynb' | egrep -v '$(EXCLUDE_SRC)')
POSTS := $(subst $(POSTS_SRC_DIR),$(POSTS_BUILD_DIR),$(POSTS_SRC))
POSTS := $(patsubst %.ipynb, %.md, $(POSTS))

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

PYTEST_OPTS ?=

#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

RUFF_CHECK_OPTS ?= --preview
RUFF_FORMAT_OPTS ?= --preview

#-------------------------------------------------------------------------------
# Phonies
#-------------------------------------------------------------------------------

PHONIES :=

################################################################################
# Targets
################################################################################

all: deps

#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

$(VENV):
	uv venv --seed

venv: $(VENV)
PHONIES := $(PHONIES) venv


#-------------------------------------------------------------------------------
# Requirements
#-------------------------------------------------------------------------------

$(REQUIREMENTS): pyproject.toml | $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# $@$(COLOR_RESET)"
	@echo

	source $(VENV) && uv pip compile -o $@ pyproject.toml

	@echo -e "$(COLOR_COMMENT)# Add Project$(COLOR_RESET)"
	echo "-e file://." >> $@
	@echo

requirements: $(REQUIREMENTS)
PHONIES := $(PHONIES) requirements


#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------

$(DEPENDENCIES): $(REQUIREMENTS) | $(BUILD_DIR)
	source $(VENV) && uv pip sync $(REQUIREMENTS)
	@echo
	@echo -e "$(COLOR_COMMENT)# Activate venv: $(COLOR_OK)source $(VENV)$(COLOR_RESET)"
	@echo -e "$(COLOR_COMMENT)# Deactivate venv: $(COLOR_OK)deactivate$(COLOR_RESET)"
	@echo
	touch $@

deps: $(DEPENDENCIES)
PHONIES := $(PHONIES) deps


#-------------------------------------------------------------------------------
# Packages
#-------------------------------------------------------------------------------

$(PACKAGES_DIR):
	mkdir -p $@

# Package: stickshift

$(STICKSHIFT_PACKAGE): $(STICKSHIFT_PACKAGE_REQUIRES) | $(PACKAGES_DIR) $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Package: $$(basename $@)$(COLOR_RESET)"
	@echo

	@echo -e "$(COLOR_COMMENT)# Build Package$(COLOR_RESET)"
	source $(VENV) && python -m build --outdir $(PROJECT_ROOT)/$$(dirname $@)
	@echo

packages: $(PACKAGES)

PHONIES := $(PHONIES) packages


#-------------------------------------------------------------------------------
# Posts
#-------------------------------------------------------------------------------

$(POSTS_BUILD_DIR)/%.md: $(POSTS_SRC_DIR)/%.ipynb | $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Post: $$(basename $@)$(COLOR_RESET)"
	@echo

	@echo -e "$(COLOR_COMMENT)# Content$(COLOR_RESET)"
	source $(VENV) && python -m stickshift.build.post --notebook "$<" --markdown "$@"
	@echo

	@echo -e "$(COLOR_COMMENT)# Resources$(COLOR_RESET)"
	find $$(dirname $<) -type f -name '*.svg' -exec cp {} $$(dirname $@) \;

posts: $(POSTS)

PHONIES := $(PHONIES) posts


#-------------------------------------------------------------------------------
# Site
#-------------------------------------------------------------------------------

site: $(SITE_SRC) $(POSTS)
	@echo
	@echo -e "$(COLOR_H1)# Site$(COLOR_RESET)"
	@echo

	source $(VENV) && \
	  cd $(SITE_SRC_DIR) && \
	  bundle install && \
	  bundle exec jekyll clean && \
	  bundle exec jekyll build --verbose

PHONIES := $(PHONIES) site

#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

tests: $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Tests$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) tests

coverage: $(DEPENDENCIES)
	@echo
	@echo -e "$(COLOR_H1)# Coverage$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) --cov=xformers --cov-report=html:$(BUILD_DIR)/coverage tests

PHONIES := $(PHONIES) tests coverage


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

lint-fmt: deps
	source $(VENV) && \
	  ruff format $(RUFF_FORMAT_OPTS) && \
	  ruff check --fix $(RUFF_CHECK_OPTS) && \
	  make lint-style

lint-style: deps
	source $(VENV) && \
	  ruff check $(RUFF_CHECK_OPTS) && \
	  ruff format --check $(RUFF_FORMAT_OPTS)

PHONIES := $(PHONIES) lint-fmt lint-style


#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	$(RM) $(VENV_ROOT)

clean-requirements:
	$(RM) $(REQUIREMENTS)

clean-site:
	cd $(SITE_SRC_DIR) && bundle exec jekyll clean
	$(RM) $(POSTS_BUILD_DIR)

clean: clean-cache clean-venv clean-requirements clean-site
PHONIES := $(PHONIES) clean-cache clean-venv clean-requirements clean-site clean


.PHONY: $(PHONIES)
