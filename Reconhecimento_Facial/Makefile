#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Reconhecimento_Facial
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Run the end-to-end test with the real model (downloads weights and sample photos)
.PHONY: test-integracao
test-integracao:
	python -m pytest tests -m integracao


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Tira 3 fotos pela webcam e cadastra a pessoa: make capturar NOME=gustavo
.PHONY: capturar
capturar:
	$(PYTHON_INTERPRETER) -m module_reconhecimento_facial.main capturar $(NOME)


## Cadastra as pessoas autorizadas (fotos em data/autorizados)
.PHONY: cadastrar
cadastrar:
	$(PYTHON_INTERPRETER) -m module_reconhecimento_facial.main cadastrar


## Verifica as fotos de data/teste e salva o resultado em reports/resultados
.PHONY: reconhecer
reconhecer:
	$(PYTHON_INTERPRETER) -m module_reconhecimento_facial.main reconhecer


## Controle de acesso ao vivo pela webcam (q para sair)
.PHONY: webcam
webcam:
	$(PYTHON_INTERPRETER) -m module_reconhecimento_facial.main webcam


## Mede falsos aceites/rejeicoes em uma base rotulada: make avaliar PASTA=caminho/lfw
.PHONY: avaliar
avaliar:
	$(PYTHON_INTERPRETER) -m module_reconhecimento_facial.main avaliar $(PASTA)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
