#!/usr/bin/env bash
# Host-side driver for the dft_xc_search Claude sandbox.
#
#   ./docker/xcdev.sh build          build (or rebuild) the image
#   ./docker/xcdev.sh claude [args]  start Claude with all permissions on
#   ./docker/xcdev.sh shell          plain bash shell in the sandbox
#   ./docker/xcdev.sh run <cmd...>   one-off command in the sandbox
#   ./docker/xcdev.sh sync-wip       copy uncommitted host changes into sandbox
#   ./docker/xcdev.sh log            what the sandbox has committed so far
#   ./docker/xcdev.sh ls [path]      list files inside the sandbox
#   ./docker/xcdev.sh export [file]  git bundle of committed work -> host
#   ./docker/xcdev.sh fetch <path..> tar of result files -> host
#   ./docker/xcdev.sh reset          throw away the sandbox work volume
#   ./docker/xcdev.sh reset-all      also throw away the sandbox Claude login
#
# Safety model
# ------------
# The host filesystem is mounted read-only and nothing else is shared:
#
#   /host/repo   <- this repository, READ-ONLY
#   /seed        <- your Claude credentials file, READ-ONLY (opt out below)
#   /work        <- a container-managed named volume; the only writable storage
#
# There is no container socket, no host network namespace, and no writable bind
# mount, so a process inside cannot modify anything on your drive. Getting work
# back out is always initiated from the host, via `export` (commits) or
# `fetch` (result files): the host shell does the writing, the container only
# writes to its own stdout.
#
# Environment knobs:
#   XCDEV_RW=1        bind-mount the repo WRITABLE instead (drops the guarantee)
#   XCDEV_NO_AUTH=1   do not share host Claude credentials; log in inside instead
#   XCDEV_MEMORY=8g   container memory cap (default: unset)
#   XCDEV_IMAGE=...   image tag (default: dft-xc-search-sandbox)
#   XCDEV_ENGINE=...  container engine to use (default: podman, else docker)

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

# Container engine: docker or podman, whichever is present (override with
# XCDEV_ENGINE). Rootless podman needs --userns=keep-id, otherwise the host
# user maps to container root and the mode-0600 credentials file mounted at
# /seed is unreadable by the unprivileged `dev` user inside.
ENGINE="${XCDEV_ENGINE:-}"
if [ -z "$ENGINE" ]; then
    for candidate in podman docker; do
        if command -v "$candidate" >/dev/null 2>&1; then ENGINE="$candidate"; break; fi
    done
fi

require_engine() {
    [ -n "$ENGINE" ] \
        || die "neither podman nor docker found. On Ubuntu: sudo apt install podman"
    command -v "$ENGINE" >/dev/null 2>&1 \
        || die "XCDEV_ENGINE=$ENGINE is not installed"
    "$ENGINE" info >/dev/null 2>&1 \
        || die "cannot talk to the $ENGINE runtime ('$ENGINE info' failed)"
}

engine_is_rootless_podman() {
    [ "$(basename -- "$ENGINE")" = "podman" ] \
        && [ "$("$ENGINE" info --format '{{.Host.Security.Rootless}}' 2>/dev/null)" = "true" ]
}

# ---------------------------------------------------------------------------
# `<engine> run` argument assembly
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

    # Under rootless podman the host user would otherwise map to container root
    # and everything else to a subuid, leaving the mode-0600 credentials file
    # unreadable to `dev`. keep-id maps host UID -> the same UID inside.
    if engine_is_rootless_podman; then
        RUN_ARGS+=(--userns=keep-id)
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

engine_run() {
    build_run_args
    exec "$ENGINE" run "${RUN_ARGS[@]}" "$IMAGE" "$@"
}

# Non-interactive variant used by export/log, which need to capture stdout.
engine_run_quiet() {
    build_run_args
    "$ENGINE" run "${RUN_ARGS[@]}" "$IMAGE" "$@"
}

# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
cmd_build() {
    "$ENGINE" build \
        --build-arg "UID=$(id -u)" \
        --build-arg "GID=$(id -g)" \
        -t "$IMAGE" \
        "$@" \
        "$SCRIPT_DIR"
}

cmd_claude() {
    # --dangerously-skip-permissions is the point of the sandbox: inside the
    # container there is nothing of yours left to damage.
    engine_run claude --dangerously-skip-permissions "$@"
}

cmd_shell() { engine_run bash -l; }

cmd_run() {
    [ "$#" -gt 0 ] || die "run: need a command"
    engine_run "$@"
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
        | engine_run_quiet git apply --whitespace=nowarn - \
        && printf 'xcdev: applied host work-in-progress into the sandbox checkout\n' >&2
}

cmd_log() {
    engine_run_quiet git -c color.ui=always log --oneline --graph --decorate -n 40 --all
}

cmd_export() {
    local out="${1:-${REPO_ROOT}/sandbox-export.bundle}"
    [ "${XCDEV_RW:-0}" = "1" ] && die "export is pointless in XCDEV_RW=1 mode; the work is already on your drive"

    # Written by the *host* user via a shell redirect -- the container never
    # gets a writable handle on your filesystem.
    engine_run_quiet bash -lc \
        'git bundle create /tmp/sandbox.bundle --all >/dev/null 2>&1 && cat /tmp/sandbox.bundle' \
        > "$out"

    printf 'xcdev: wrote %s\n' "$out" >&2
    printf 'xcdev: inspect it with:  git bundle list-heads %q\n' "$out" >&2
    printf 'xcdev: pull a branch in: git fetch %q <branch>:<local-branch>\n' "$out" >&2
}

cmd_ls() {
    # Browse the sandbox before deciding what to pull out.
    local target="${1:-.}"
    engine_run_quiet bash -lc 'cd -- "$1" && ls -la --si' _ "$target"
}

cmd_fetch() {
    # Pull result files (plots, .npz, optuna dbs -- anything gitignored, so
    # anything `export` will not carry) out of the sandbox as a tar.
    #
    # Note the direction of the write: the *host* shell creates and fills
    # $out via the redirect below. The container only writes to its own
    # stdout, so this needs no writable mount and does not weaken anything.
    [ "${XCDEV_RW:-0}" = "1" ] && die "fetch is pointless in XCDEV_RW=1 mode; the files are already on your drive"

    local out="${REPO_ROOT}/sandbox-data.tar"
    local paths=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -o) shift; [ "$#" -gt 0 ] || die "fetch: -o needs a filename"; out="$1"; shift ;;
            *)  paths+=("$1"); shift ;;
        esac
    done
    [ "${#paths[@]}" -gt 0 ] \
        || die "fetch: name at least one path inside the sandbox checkout (try 'xcdev.sh ls' first), e.g. 'fetch data figures -o results.tar'"

    engine_run_quiet tar -cf - -C "$CHECKOUT" -- "${paths[@]}" > "$out"

    printf 'xcdev: wrote %s\n' "$out" >&2
    printf 'xcdev: inspect it with: tar -tvf %q\n' "$out" >&2
    printf 'xcdev: unpack it with:  tar -xf %q -C <somewhere>\n' "$out" >&2
}

cmd_reset() {
    printf 'This deletes volume %s and every uncommitted change in the sandbox.\n' "$WORK_VOLUME" >&2
    read -r -p 'Type yes to continue: ' reply
    [ "$reply" = "yes" ] || die "aborted"
    "$ENGINE" volume rm "$WORK_VOLUME" >/dev/null && printf 'xcdev: removed %s\n' "$WORK_VOLUME" >&2
}

cmd_reset_all() {
    printf 'This deletes volumes %s and %s (work AND the sandbox Claude login).\n' "$WORK_VOLUME" "$HOME_VOLUME" >&2
    read -r -p 'Type yes to continue: ' reply
    [ "$reply" = "yes" ] || die "aborted"
    "$ENGINE" volume rm "$WORK_VOLUME" "$HOME_VOLUME" >/dev/null \
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
require_engine

case "${1:-help}" in
    build)     shift; cmd_build "$@" ;;
    claude|c)  shift; cmd_claude "$@" ;;
    shell|sh)  shift; cmd_shell ;;
    run)       shift; cmd_run "$@" ;;
    sync-wip)  shift; cmd_sync_wip ;;
    log)       shift; cmd_log ;;
    ls)        shift; cmd_ls "$@" ;;
    fetch)     shift; cmd_fetch "$@" ;;
    export)    shift; cmd_export "$@" ;;
    reset)     shift; cmd_reset ;;
    reset-all) shift; cmd_reset_all ;;
    *)         die "unknown subcommand '$1' (try: help)" ;;
esac
