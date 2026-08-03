#!/usr/bin/env bash
# Host-side driver for the dft_xc_search Claude sandbox.
#
#   ./docker/xcdev.sh build          build (or rebuild) the image
#   ./docker/xcdev.sh claude [args]  start Claude with all permissions on
#   ./docker/xcdev.sh shell          plain bash shell in the sandbox
#   ./docker/xcdev.sh run <cmd...>   one-off command in the sandbox
#   ./docker/xcdev.sh sync-wip       copy uncommitted host changes into sandbox
#   ./docker/xcdev.sh log            what the sandbox has committed so far
#   ./docker/xcdev.sh export [file]  git bundle of the sandbox work -> host
#   ./docker/xcdev.sh reset          throw away the sandbox work volume
#   ./docker/xcdev.sh reset-all      also throw away the sandbox Claude login
#
# Safety model
# ------------
# The host filesystem is mounted read-only and nothing else is shared:
#
#   /host/repo   <- this repository, READ-ONLY
#   /seed        <- your Claude credentials file, READ-ONLY (opt out below)
#   /work        <- a Docker named volume; this is the only writable storage
#
# There is no Docker socket, no host network namespace, and no writable bind
# mount, so a process inside cannot modify anything on your drive. Getting work
# back out is always initiated from the host, via `export`.
#
# Environment knobs:
#   XCDEV_RW=1        bind-mount the repo WRITABLE instead (drops the guarantee)
#   XCDEV_NO_AUTH=1   do not share host Claude credentials; log in inside instead
#   XCDEV_MEMORY=8g   container memory cap (default: unset)
#   XCDEV_IMAGE=...   image tag (default: dft-xc-search-sandbox)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the *main* repository root, even when this script is invoked from a
# git worktree (a worktree's .git is a file pointing outside the mount).
if command -v git >/dev/null 2>&1 \
   && GIT_COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; then
    REPO_ROOT="$(dirname -- "$GIT_COMMON_DIR")"
else
    REPO_ROOT="$(dirname -- "$SCRIPT_DIR")"
fi

IMAGE="${XCDEV_IMAGE:-dft-xc-search-sandbox}"
PROJECT="$(basename -- "$REPO_ROOT")"
WORK_VOLUME="${XCDEV_WORK_VOLUME:-${PROJECT}-sandbox-work}"
HOME_VOLUME="${XCDEV_HOME_VOLUME:-${PROJECT}-sandbox-home}"
CHECKOUT="/work/${PROJECT}"

die() { printf 'xcdev: %s\n' "$*" >&2; exit 1; }

require_docker() {
    command -v docker >/dev/null 2>&1 \
        || die "docker is not installed. On Ubuntu: sudo apt install docker.io && sudo usermod -aG docker \$USER (then log out and back in)"
    docker info >/dev/null 2>&1 \
        || die "cannot talk to the Docker daemon. Try: sudo systemctl start docker (and check you are in the 'docker' group)"
}

# ---------------------------------------------------------------------------
# docker run argument assembly
# ---------------------------------------------------------------------------
build_run_args() {
    RUN_ARGS=(
        --rm
        --init
        --hostname xcdev
        --pids-limit 4096
        -e "XCDEV_CHECKOUT=${CHECKOUT}"
        -e "XCDEV_HOST_REPO=/host/repo"
        -v "${HOME_VOLUME}:/home/dev"
    )

    # -i always, so piped stdin (see sync-wip) reaches the container; -t only
    # for a real terminal, so that `export` can stream a binary bundle out.
    RUN_ARGS+=(-i)
    if [ -t 0 ] && [ -t 1 ]; then
        RUN_ARGS+=(-t)
    fi

    if [ "${XCDEV_RW:-0}" = "1" ]; then
        printf 'xcdev: WARNING: XCDEV_RW=1 -- the repo is mounted WRITABLE.\n' >&2
        RUN_ARGS+=(-v "${REPO_ROOT}:${CHECKOUT}")
    else
        RUN_ARGS+=(-v "${REPO_ROOT}:/host/repo:ro" -v "${WORK_VOLUME}:/work")
    fi

    # Credentials: prefer an explicit API key, else share the host login as a
    # read-only file that the entrypoint copies into the container's own home.
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        RUN_ARGS+=(-e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
    elif [ "${XCDEV_NO_AUTH:-0}" != "1" ] && [ -f "$HOME/.claude/.credentials.json" ]; then
        RUN_ARGS+=(-v "$HOME/.claude/.credentials.json:/seed/.credentials.json:ro")
    fi

    for var in GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL; do
        [ -n "${!var:-}" ] && RUN_ARGS+=(-e "${var}=${!var}")
    done

    [ -n "${XCDEV_MEMORY:-}" ] && RUN_ARGS+=(--memory "${XCDEV_MEMORY}")

    return 0
}

docker_run() {
    build_run_args
    exec docker run "${RUN_ARGS[@]}" "$IMAGE" "$@"
}

# Non-interactive variant used by export/log, which need to capture stdout.
docker_run_quiet() {
    build_run_args
    docker run "${RUN_ARGS[@]}" "$IMAGE" "$@"
}

# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
cmd_build() {
    docker build \
        --build-arg "UID=$(id -u)" \
        --build-arg "GID=$(id -g)" \
        -t "$IMAGE" \
        "$@" \
        "$SCRIPT_DIR"
}

cmd_claude() {
    # --dangerously-skip-permissions is the point of the sandbox: inside the
    # container there is nothing of yours left to damage.
    docker_run claude --dangerously-skip-permissions "$@"
}

cmd_shell() { docker_run bash -l; }

cmd_run() {
    [ "$#" -gt 0 ] || die "run: need a command"
    docker_run "$@"
}

cmd_sync_wip() {
    # The sandbox clones committed history only. This carries your uncommitted
    # changes to tracked files across as well. Run it before `claude`.
    [ "${XCDEV_RW:-0}" = "1" ] && die "sync-wip is unnecessary in XCDEV_RW=1 mode"

    if git -C "$REPO_ROOT" diff --quiet HEAD; then
        printf 'xcdev: no uncommitted changes to sync\n' >&2
        return 0
    fi

    git -C "$REPO_ROOT" diff HEAD \
        | docker_run_quiet git apply --whitespace=nowarn - \
        && printf 'xcdev: applied host work-in-progress into the sandbox checkout\n' >&2
}

cmd_log() {
    docker_run_quiet git -c color.ui=always log --oneline --graph --decorate -n 40 --all
}

cmd_export() {
    local out="${1:-${REPO_ROOT}/sandbox-export.bundle}"
    [ "${XCDEV_RW:-0}" = "1" ] && die "export is pointless in XCDEV_RW=1 mode; the work is already on your drive"

    # Written by the *host* user via a shell redirect -- the container never
    # gets a writable handle on your filesystem.
    docker_run_quiet bash -lc \
        'git bundle create /tmp/sandbox.bundle --all >/dev/null 2>&1 && cat /tmp/sandbox.bundle' \
        > "$out"

    printf 'xcdev: wrote %s\n' "$out" >&2
    printf 'xcdev: inspect it with:  git bundle list-heads %q\n' "$out" >&2
    printf 'xcdev: pull a branch in: git fetch %q <branch>:<local-branch>\n' "$out" >&2
}

cmd_reset() {
    printf 'This deletes volume %s and every uncommitted change in the sandbox.\n' "$WORK_VOLUME" >&2
    read -r -p 'Type yes to continue: ' reply
    [ "$reply" = "yes" ] || die "aborted"
    docker volume rm "$WORK_VOLUME" >/dev/null && printf 'xcdev: removed %s\n' "$WORK_VOLUME" >&2
}

cmd_reset_all() {
    printf 'This deletes volumes %s and %s (work AND the sandbox Claude login).\n' "$WORK_VOLUME" "$HOME_VOLUME" >&2
    read -r -p 'Type yes to continue: ' reply
    [ "$reply" = "yes" ] || die "aborted"
    docker volume rm "$WORK_VOLUME" "$HOME_VOLUME" >/dev/null \
        && printf 'xcdev: removed both volumes\n' >&2
}

# Print the header comment block (everything after the shebang, up to the first
# non-comment line) as the usage text.
cmd_help() {
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

case "${1:-help}" in
    help|-h|--help) cmd_help; exit 0 ;;
esac
require_docker

case "${1:-help}" in
    build)     shift; cmd_build "$@" ;;
    claude|c)  shift; cmd_claude "$@" ;;
    shell|sh)  shift; cmd_shell ;;
    run)       shift; cmd_run "$@" ;;
    sync-wip)  shift; cmd_sync_wip ;;
    log)       shift; cmd_log ;;
    export)    shift; cmd_export "$@" ;;
    reset)     shift; cmd_reset ;;
    reset-all) shift; cmd_reset_all ;;
    *)         die "unknown subcommand '$1' (try: help)" ;;
esac
