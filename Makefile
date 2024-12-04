.ONESHELL:

SHELL := /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh; conda activate
CONDA_DEACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh; conda deactivate
ENV_NAME := zero_shot_tree
all: init

init:
	$(CONDA_ACTIVATE); \
	source ./build_env.sh ${ENV_NAME}; \

clean:
	$(CONDA_DEACTIVATE); \
	conda remove -n ${ENV_NAME} --all -y; \
