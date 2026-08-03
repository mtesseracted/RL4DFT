# Claude sandbox

A container for running Claude Code against this repo with
`--dangerously-skip-permissions` on, arranged so that a runaway agent cannot
touch anything on your drive.

## What is actually isolated

| Path in container | Backed by | Access |
| --- | --- | --- |
| `/host/repo` | this repository on your drive | **read-only** |
| `/seed/.credentials.json` | `~/.claude/.credentials.json` | **read-only** |
| `/work` | Docker named volume `<project>-sandbox-work` | read-write |
| `/home/dev` | Docker named volume `<project>-sandbox-home` | read-write |

Nothing else from the host is shared. In particular there is **no Docker
socket** (which would be equivalent to root on the host), no `--privileged`, no
`--network host`, and no writable bind mount. Inside the container the `dev`
user has passwordless `sudo` on purpose — Claude is meant to be able to
`apt install`, rewrite files and delete things freely, because the blast radius
is a Docker volume you can throw away with `xcdev.sh reset`.

On startup the entrypoint clones `/host/repo` into `/work/<project>`. That
clone is where Claude works. The host remote is kept as `host`, so you can
`git fetch host && git merge host/main` inside the sandbox to pick up new
commits — and you can never push back over it, because the mount is read-only.

What this does **not** protect against: the container has normal outbound
network access (Claude needs the API), so anything readable inside — your
repo and the Claude credentials you chose to share — could in principle leave.
See "Tightening further" below.

## Setup

Docker is not currently installed on this machine. On Ubuntu:

```sh
sudo apt install docker.io
sudo usermod -aG docker "$USER"   # then log out and back in
```

Then build the image (a few minutes, mostly apt and npm):

```sh
./docker/xcdev.sh build
```

## Use

```sh
./docker/xcdev.sh sync-wip     # optional: carry uncommitted changes across
./docker/xcdev.sh claude       # Claude, all permissions on
```

`claude` passes any extra arguments straight through, so
`./docker/xcdev.sh claude -p "run the stage 1 tests"` works.

Other subcommands:

```sh
./docker/xcdev.sh shell             # bash in the sandbox
./docker/xcdev.sh run pytest -q     # one-off command
./docker/xcdev.sh log               # commits made inside the sandbox
./docker/xcdev.sh export            # git bundle of the sandbox work -> host
./docker/xcdev.sh reset             # discard the sandbox work volume
./docker/xcdev.sh reset-all         # discard the work volume AND the login
```

The work volume persists between runs, so you can stop and restart the sandbox
without losing anything.

## Getting work back out

Because the host mount is read-only, the sandbox cannot write results to your
drive; extraction is always initiated by you, from the host:

```sh
./docker/xcdev.sh export                        # writes sandbox-export.bundle
git bundle list-heads sandbox-export.bundle     # see what is in it
git fetch sandbox-export.bundle stage1:sandbox-stage1
git log sandbox-stage1                          # review before merging
```

That review step is the point of the design: nothing lands in your working
tree until you have looked at it.

Two caveats on what crosses into the sandbox in the first place:

- The clone carries **committed history only**. Run `xcdev.sh sync-wip` first
  to apply your uncommitted changes to tracked files, or just commit them.
- Untracked and gitignored paths do not cross — including `data/` and `.venv/`.
  `data/` is a cache the code rebuilds, and the container has its own venv at
  `/opt/venv`, so neither matters.

## Notes

- Python deps are pinned informally in the Dockerfile and mirror
  `pyproject.toml`; the entrypoint additionally runs `pip install -e` on the
  checkout, so dependencies you add to `pyproject.toml` are picked up on the
  next container start without a rebuild.
- `MPLBACKEND=Agg` is set, so matplotlib writes files instead of trying to open
  a window.
- Claude's auto-updater is disabled inside the image (it is installed globally
  as root and would be shadowed by the `/home/dev` volume anyway). To move to a
  newer Claude Code, rerun `./docker/xcdev.sh build`.
- If you would rather not share your host login, run with `XCDEV_NO_AUTH=1` and
  do `/login` inside the sandbox once — the token then lives in the sandbox
  home volume. Setting `ANTHROPIC_API_KEY` in your host environment takes
  precedence over both.
- `XCDEV_RW=1` bind-mounts the repo writable instead. That is the ordinary
  dev-container arrangement and it **discards** the guarantee this setup exists
  to provide; the script warns when you use it.
- `XCDEV_MEMORY=8g` caps container memory if a search run starts thrashing.

## Tightening further

If you want to close the exfiltration path as well, the usual approach is an
egress allowlist: start the container with `--cap-add=NET_ADMIN`, and have an
init script install `iptables` rules that drop everything except DNS and the
Anthropic API endpoints. That is a meaningful amount of extra machinery and it
breaks `pip install` and `apt`, so it is deliberately not wired up here.
