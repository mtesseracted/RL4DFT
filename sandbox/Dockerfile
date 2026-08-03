# Sandbox image for running Claude Code against dft_xc_search with
# --dangerously-skip-permissions.
#
# The safety property does NOT come from this image: it comes from how the
# container is run (see sandbox/xcdev.sh). Nothing on the host is ever mounted
# writable, and the Docker socket is never exposed. Inside the container the
# user has passwordless sudo on purpose -- the whole point is that Claude can
# do anything it likes in here.

FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        sudo \
        less \
        jq \
        ripgrep \
        procps \
        vim-tiny \
        openssh-client \
        build-essential \
        gfortran \
        pkg-config \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Claude Code. Installed globally as root so that it is NOT shadowed by the
# named volume mounted over /home/dev at runtime; that also means a new Claude
# release needs `xcdev.sh build` rather than an in-place self-update.
RUN npm install -g @anthropic-ai/claude-code \
    && npm cache clean --force \
    && claude --version

# Match the host UID/GID so that anything exported later has sane ownership.
ARG UID=1000
ARG GID=1000
RUN if ! getent group "$GID" >/dev/null; then groupadd -g "$GID" dev; fi \
    && useradd -m -u "$UID" -g "$GID" -s /bin/bash dev \
    && echo 'dev ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/dev \
    && chmod 0440 /etc/sudoers.d/dev

# Project dependencies live in a venv owned by the dev user, so Claude can
# `pip install` more of them without sudo.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/home/dev/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
RUN python -m venv "$VIRTUAL_ENV" \
    && chown -R "$UID:$GID" "$VIRTUAL_ENV"

# Kept in sync with pyproject.toml by hand; the entrypoint runs
# `pip install -e` on the checkout, which picks up anything added since.
RUN pip install --no-cache-dir \
        numpy \
        scipy \
        optuna \
        matplotlib \
        pytest

# A login shell sources /etc/profile, which resets PATH and would drop the venv
# -- so `xcdev.sh shell` (bash -l) has to be told about it separately from the
# ENV above.
RUN printf '%s\n' \
        'export VIRTUAL_ENV=/opt/venv' \
        'case ":$PATH:" in *":/opt/venv/bin:"*) ;; *) PATH="/opt/venv/bin:$PATH" ;; esac' \
        'export PATH' \
        > /etc/profile.d/10-xcdev-venv.sh

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

USER dev
WORKDIR /work

ENV XCDEV_CHECKOUT=/work/dft_xc_search \
    XCDEV_HOST_REPO=/host/repo \
    DISABLE_AUTOUPDATER=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash", "-l"]
