#!/usr/bin/env bash
# Container entrypoint: prepare a writable checkout that lives only inside
# Docker, then hand off to the requested command.
set -euo pipefail

CHECKOUT="${XCDEV_CHECKOUT:-/work/dft_xc_search}"
HOST_REPO="${XCDEV_HOST_REPO:-/host/repo}"
SEED_DIR="${XCDEV_SEED_DIR:-/seed}"

log() { printf '[xcdev] %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. Checkout
#
# In the default (read-only) mode the host repo is bind-mounted at /host/repo
# read-only and we clone from it into /work, which is a Docker named volume.
# Everything Claude does then lands in the volume, never on the host drive.
#
# In --rw mode xcdev.sh bind-mounts the host repo directly at $CHECKOUT, so it
# is already a git repo and we leave it alone.
# ---------------------------------------------------------------------------
if [ -e "$CHECKOUT/.git" ]; then
    log "using existing checkout at $CHECKOUT"
elif [ -e "$HOST_REPO/.git" ]; then
    log "cloning $HOST_REPO -> $CHECKOUT (host repo stays read-only)"
    # stdout is kept clean throughout: `xcdev.sh export` streams a binary git
    # bundle through it.
    git clone "$HOST_REPO" "$CHECKOUT" >&2
    git -C "$CHECKOUT" remote rename origin host 2>/dev/null || true
else
    log "no repo found at $HOST_REPO and no checkout at $CHECKOUT; starting empty"
    mkdir -p "$CHECKOUT"
fi

if [ -e "$CHECKOUT/.git" ]; then
    # The clone source is a read-only mount, so fetching is safe; pushing to it
    # is not possible, which is exactly what we want.
    git -C "$CHECKOUT" config --global --add safe.directory "$CHECKOUT" 2>/dev/null || true
    git config --global user.name  "${GIT_AUTHOR_NAME:-Claude Sandbox}" 2>/dev/null || true
    git config --global user.email "${GIT_AUTHOR_EMAIL:-claude@sandbox.invalid}" 2>/dev/null || true
    if git -C "$CHECKOUT" remote get-url host >/dev/null 2>&1; then
        git -C "$CHECKOUT" fetch --quiet host || log "warning: could not fetch from host remote"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Claude credentials
#
# xcdev.sh may mount the host credentials file read-only at /seed. We copy it
# into the container's own home (a named volume) so the host copy is never
# written to. Skipped entirely if ANTHROPIC_API_KEY is set or a login already
# exists in the volume.
# ---------------------------------------------------------------------------
if [ ! -f "$HOME/.claude/.credentials.json" ] && [ -f "$SEED_DIR/.credentials.json" ]; then
    log "seeding Claude credentials from host (copy; host file is read-only)"
    mkdir -p "$HOME/.claude"
    cp "$SEED_DIR/.credentials.json" "$HOME/.claude/.credentials.json"
    chmod 600 "$HOME/.claude/.credentials.json"
fi

# A minimal config so Claude does not run the interactive onboarding or the
# bypass-permissions confirmation on every fresh container.
if [ ! -f "$HOME/.claude.json" ]; then
    printf '%s\n' \
        '{"hasCompletedOnboarding": true, "bypassPermissionsModeAccepted": true}' \
        > "$HOME/.claude.json"
fi

# ---------------------------------------------------------------------------
# 3. Editable install of the project
# ---------------------------------------------------------------------------
STAMP="${VIRTUAL_ENV:-/tmp}/.xcsearch-installed"
if [ -f "$CHECKOUT/pyproject.toml" ] && [ ! -e "$STAMP" ]; then
    log "pip install -e $CHECKOUT"
    if pip install --quiet -e "$CHECKOUT" >&2; then
        touch "$STAMP"
    else
        log "warning: editable install failed; continuing anyway"
    fi
fi

cd "$CHECKOUT"
exec "$@"
