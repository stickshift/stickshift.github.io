################################################################################
# Makefile
#
#   * General
#   * Output Dirs
#   * Environment
#   * Articles
#   * Theme
#   * Site
#   * Tests
#   * Linters
#   * Phonies
#
################################################################################

# Verify environment.sh
ifneq ($(PROJECT_NAME),stickshift.github.io)
$(error Environment not configured. Run `source environment.sh`)
endif


################################################################################
# Settings
################################################################################


#-------------------------------------------------------------------------------
# General
#-------------------------------------------------------------------------------

# Bash
export SHELL := /bin/bash
.SHELLFLAGS := -e -u -o pipefail -c

# Colors - Supports colorized messages
COLOR_H1=\033[38;5;12m
COLOR_OK=\033[38;5;02m
COLOR_COMMENT=\033[38;5;08m
COLOR_RESET=\033[0m

# EXCLUDE_SRC - Source patterns to ignore

EXCLUDE_SRC := __pycache__ \
			   .egg-info \
			   .ipynb_checkpoints \
			   .venv
EXCLUDE_SRC := $(subst $(eval ) ,|,$(EXCLUDE_SRC))

# Commands
RM := rm -rf


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

OUTPUT_DIRS :=

BUILD_DIR := $(PROJECT_ROOT)/.build
OUTPUT_DIRS := $(OUTPUT_DIRS) $(BUILD_DIR)


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

VENV_ROOT := .venv
VENV_SRC := pyproject.toml uv.lock
VENV := $(VENV_ROOT)/bin/activate


#-------------------------------------------------------------------------------
# Site
#-------------------------------------------------------------------------------

SITE_SRC_DIR := $(PROJECT_ROOT)/site
SITE_BUILD_DIR := $(BUILD_DIR)/site
SITE_PUBLISH_DIR := $(SITE_SRC_DIR)/dist

ARTICLES_DIR := $(PROJECT_ROOT)/articles
ARTICLE_IDS := $(foreach dir,$(shell find $(ARTICLES_DIR) -mindepth 1 -maxdepth 1 -type d),$(notdir $(dir)))
ARTICLE_VENV_ROOTS := $(foreach id,$(ARTICLE_IDS),$(ARTICLES_DIR)/$(id)/.venv)
ARTICLE_VENVS := $(foreach id,$(ARTICLE_IDS),$(ARTICLES_DIR)/$(id)/.venv/bin/activate)
ARTICLE_KERNEL_SPEC_ROOTS := $(foreach id,$(ARTICLE_IDS),$(JUPYTER_DATA_DIR)/kernels/$(id))
ARTICLE_KERNEL_SPECS := $(foreach id,$(ARTICLE_IDS),$(JUPYTER_DATA_DIR)/kernels/$(id)/kernel.json)
ARTICLE_BUNDLE_ROOTS := $(foreach id,$(ARTICLE_IDS),$(SITE_BUILD_DIR)/articles/$(id))
ARTICLE_BUNDLES := $(foreach id,$(ARTICLE_IDS),$(SITE_BUILD_DIR)/articles/$(id)/index.html)
ARTICLE_ROOTS := $(ARTICLE_VENV_ROOTS) $(ARTICLE_KERNEL_SPEC_ROOTS) $(ARTICLE_BUNDLE_ROOTS)
ARTICLES := $(ARTICLE_VENVS) $(ARTICLE_KERNEL_SPECS) $(ARTICLE_BUNDLES)

THEME := stickshift
THEME_PYGMENTS_STYLE := tango
THEME_SRC_DIR := $(SITE_SRC_DIR)/themes/$(THEME)

SITE_IMAGE_SRC := $(shell find $(THEME_SRC_DIR)/assets/images -type f | egrep -v '$(EXCLUDE_SRC)')
SITE_IMAGE_BUNDLE := $(patsubst $(THEME_SRC_DIR)/assets/%,$(SITE_BUILD_DIR)/%,$(SITE_IMAGE_SRC))

SITE_CSS_SRC := $(shell find $(THEME_SRC_DIR)/assets/styles -type f | egrep -v '$(EXCLUDE_SRC)')

SITE_CSS_BUNDLE :=
SITE_CSS_BUNDLE := $(SITE_CSS_BUNDLE) $(SITE_BUILD_DIR)/styles/$(THEME).css
SITE_CSS_BUNDLE := $(SITE_CSS_BUNDLE) $(SITE_BUILD_DIR)/styles/pygments.css

SITE_BUNDLE := $(SITE_BUILD_DIR)/index.html

SITE := $(ARTICLES) $(SITE_IMAGE_BUNDLE) $(SITE_CSS_BUNDLE) $(SITE_BUNDLE)

SITE_PUBLISHED_BUNDLE := $(SITE_PUBLISH_DIR)/index.html


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

PYTEST_OPTS ?= -n auto


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

all: site
	@echo
	@echo -e "$(COLOR_H1)# $(PROJECT_NAME)$(COLOR_RESET)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Activate VENV$(COLOR_RESET)"
	@echo -e "source $(VENV)"
	@echo
	@echo -e "$(COLOR_COMMENT)# Deactivate VENV$(COLOR_RESET)"
	@echo -e "deactivate"
	@echo


#-------------------------------------------------------------------------------
# Output Dirs
#-------------------------------------------------------------------------------

$(BUILD_DIR):
	mkdir -p $@


#-------------------------------------------------------------------------------
# Environment
#-------------------------------------------------------------------------------

$(VENV): $(VENV_SRC)
	uv sync
	
	source $@ && python -m ipykernel install --user --name "workspace" --display-name "Workspace"
	
	touch $@

venv: $(VENV)
PHONIES := $(PHONIES) venv


#-------------------------------------------------------------------------------
# Site
#-------------------------------------------------------------------------------

# Venvs
$(ARTICLES_DIR)/%/.venv/bin/activate: $(ARTICLES_DIR)/%/pyproject.toml
	@echo
	@echo -e "$(COLOR_H1)# Configure venv $*$(COLOR_RESET)"
	@echo

	uv sync --directory "$(ARTICLES_DIR)/$*"
	
	touch $@

# Kernel Specs
$(JUPYTER_DATA_DIR)/kernels/%/kernel.json: $(ARTICLES_DIR)/%/.venv/bin/activate
	@echo
	@echo -e "$(COLOR_H1)# Configure kernel $*$(COLOR_RESET)"
	@echo

	mkdir -p "$(dir $@)"
	source "$<" && python -m ipykernel install --user --name "$*" --display-name "$*"
	
	touch $@

# Articles
$(SITE_BUILD_DIR)/articles/%/index.html: $(ARTICLES_DIR)/%/index.md $(JUPYTER_DATA_DIR)/kernels/%/kernel.json | $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Article $*$(COLOR_RESET)"
	@echo

	mkdir -p $$(dirname $@)	
	
	@echo
	@echo -e "$(COLOR_COMMENT)# Sync resources$(COLOR_RESET)"	
	mkdir -p $$(dirname $@)/resources
	rsync -a $(ARTICLES_DIR)/$*/resources/ $$(dirname $@)/resources/

	@echo
	@echo -e "$(COLOR_COMMENT)# Build article$(COLOR_RESET)"
	source $(VENV) && python -m stickshift.build_article --theme $(THEME_SRC_DIR) $< $@

# Theme Images
$(SITE_BUILD_DIR)/images/%: $(THEME_SRC_DIR)/assets/images/%
	@echo
	@echo -e "$(COLOR_COMMENT)# $*$(COLOR_RESET)"
	mkdir -p $(dir $@)
	cp $< $@

# Theme Styles
$(SITE_BUILD_DIR)/styles/$(THEME).css: $(SITE_CSS_SRC)
	@echo
	@echo -e "$(COLOR_H1)# Compile CSS$(COLOR_RESET)"
	@echo

	mkdir -p $(dir $@)
	sass $(THEME_SRC_DIR)/assets/styles/$(THEME).scss $@

# Pygments
$(SITE_BUILD_DIR)/styles/pygments.css: | $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Pygments$(COLOR_RESET)"
	@echo

	mkdir -p $(dir $@)
	source $(VENV) && pygmentize -S $(THEME_PYGMENTS_STYLE) -f html > $@

# Site Index
$(SITE_BUILD_DIR)/index.html: $(ARTICLES) | $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Site Index$(COLOR_RESET)"
	@echo

	source $(VENV) && python -m stickshift.build_index --theme $(THEME_SRC_DIR) $@

# Published Site
$(SITE_PUBLISHED_BUNDLE): $(SITE)
	@echo
	@echo -e "$(COLOR_H1)# Publish Site$(COLOR_RESET)"
	@echo
	
	$(RM) $(SITE_PUBLISH_DIR)
	mkdir -p $(SITE_PUBLISH_DIR)
	cp -R $(SITE_BUILD_DIR)/* $(SITE_PUBLISH_DIR)

	touch $@

articles: $(ARTICLES)

site: $(SITE)

deploy: $(SITE)
	source $(VENV) && python -m http.server -d $(SITE_BUILD_DIR)

publish: $(SITE_PUBLISHED_BUNDLE)

PHONIES := $(PHONIES) articles site deploy publish


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

tests: $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Tests$(COLOR_RESET)"
	@echo

	source $(VENV) && pytest $(PYTEST_OPTS) tests

coverage: $(VENV)
	@echo
	@echo -e "$(COLOR_H1)# Coverage$(COLOR_RESET)"
	@echo
	mkdir -p $$(dirname $(BUILD_DIR)/coverage)
	source $(VENV) && pytest $(PYTEST_OPTS) --cov=xformers --cov-report=html:$(BUILD_DIR)/coverage tests

PHONIES := $(PHONIES) tests coverage


#-------------------------------------------------------------------------------
# Linters
#-------------------------------------------------------------------------------

lint: venv
	uvx ruff check $(RUFF_CHECK_OPTS)
	uvx ruff format --check $(RUFF_FORMAT_OPTS)

lint-fix: venv
	uvx ruff format $(RUFF_FORMAT_OPTS)
	uvx ruff check --fix $(RUFF_CHECK_OPTS)
	make lint

PHONIES := $(PHONIES) lint-fix lint


#-------------------------------------------------------------------------------
# Clean
#-------------------------------------------------------------------------------

clean-cache:
	find . -type d -name "__pycache__" -exec rm -rf {} +

clean-venv:
	$(RM) "$(VENV_ROOT)"

clean-articles:
	$(RM) $(ARTICLE_ROOTS)

clean-site:
	$(RM) "$(SITE_BUILD_DIR)"

clean-build: clean-articles clean-site
	$(RM) "$(BUILD_DIR)"

clean: clean-cache clean-venv clean-articles clean-site clean-build 
PHONIES := $(PHONIES) clean-cache clean-venv clean-articles clean-site clean-build clean

.PHONY: $(PHONIES)
