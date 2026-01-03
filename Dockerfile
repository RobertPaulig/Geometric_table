FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /workspace
COPY --chown=micromamba:micromamba . /workspace

RUN micromamba install -y -n base -c conda-forge python=3.11 rdkit=2022.09.5 pillow && \
    micromamba clean --all --yes

RUN micromamba run -n base pip install -e ".[dev]"

ENV PATH=/opt/conda/bin:$PATH

ENTRYPOINT ["micromamba", "run", "-n", "base"]
