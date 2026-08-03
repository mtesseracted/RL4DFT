# Claude sandbox

A container for running Claude Code against this repo with
`--dangerously-skip-permissions` on, arranged so that a runaway agent cannot
touch anything on your drive.

## What is actually isolated

| Path in container | Backed by | Access |
| --- | --- | --- |
| `/host/repo` | this repository on your drive | **read-only** |
| `/seed/.credentials.json` | `~/.claude/.credentials.json` | **read-only** |
| `/work` | named volume `<project>-sandbox-work` | read-write |
| `/home/dev` | named volume `<project>-sandbox-home` | read-write |

Nothing else from the host is shared. In particular there is **no container
socket** (which would be equivalent to root on the host), no `--privileged`, no
`--network host`, and no writable bind mount. Inside the container the `dev`
user has passwordless `sudo` on purpose — Claude is meant to be able to
`apt install`, rewrite files and delete things freely, because the blast radius
is a volume you can throw away with `xcdev.sh reset`.

On startup the entrypoint clones `/host/repo` into `/work/<project>`. That
clone is where Claude works. The host remote is kept as `host`, so you can
`git fetch host && git merge host/main` inside the sandbox to pick up new
commits — and you can never push back over it, because the mount is read-only.

What this does **not** protect against: the container has normal outbound
network access (Claude needs the API), so anything readable inside — your
repo and the Claude credentials you chose to share — could in principle leave.
See "Tightening further" below.

## Setup

`xcdev.sh` drives either **podman** (preferred, and what this is tested on) or
**docker** — it picks whichever is installed, and `XCDEV_ENGINE=docker` forces
the choice. Rootless podman is the better fit here: the container runtime
itself is unprivileged, so a container escape lands you as your own user rather
than as root.

Build the image (a few minutes, mostly apt and npm):

```sh
./docker/xcdev.sh build
```

Under rootless podman the script adds `--userns=keep-id`, which maps your host
UID to the same UID inside. Without it your host user maps to container root
and everything else to a subuid, and the mode-0600 credentials file mounted at
`/seed` would be unreadable to the unprivileged `dev` user in the container.

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
./docker/xcdev.sh ls [path]         # list files inside the sandbox
./docker/xcdev.sh export            # git bundle of committed work -> host
./docker/xcdev.sh fetch data -o r.tar   # result files -> host
./docker/xcdev.sh reset             # discard the sandbox work volume
./docker/xcdev.sh reset-all         # discard the work volume AND the login
```

The work volume persists between runs, so you can stop and restart the sandbox
without losing anything.

## Getting work and data back out

The sandbox cannot write to your drive, so extraction is always something *you*
run from the host. The trick is the direction of the write: the host command
redirects the container's **stdout** into a file on your drive. The container
only ever writes to a pipe it was handed — it never holds a writable handle on
your filesystem — so this costs nothing in isolation.

**Committed code** — as a git bundle:

```sh
./docker/xcdev.sh export                        # writes sandbox-export.bundle
git bundle list-heads sandbox-export.bundle     # see what is in it
git fetch sandbox-export.bundle stage1:sandbox-stage1
git log sandbox-stage1                          # review before merging
```

**Result files** — plots, `.npz` arrays, `optuna*.db`, anything gitignored and
so *not* in the bundle. Look first, then pull what you want as a tar:

```sh
./docker/xcdev.sh ls                      # what is in the sandbox checkout
./docker/xcdev.sh ls data/stage1          # drill in
./docker/xcdev.sh fetch data figures -o results.tar
tar -tvf results.tar                      # inspect before unpacking
tar -xf results.tar -C /tmp/from-sandbox
```

Paths are relative to the sandbox checkout. `fetch` writes
`sandbox-data.tar` next to the repo unless you pass `-o`.

**Everything at once** — if you would rather grab the whole volume and sift
through it later, the engine can do that directly:

```sh
podman volume export dft_xc_search-sandbox-work > sandbox-work.tar
```

In all three cases you review before anything lands in your working tree. That
review step is the point of the design.

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
- `XCDEV_ENGINE=docker` forces the engine when both podman and docker are
  installed; the default preference is podman.

## Tightening further

If you want to close the exfiltration path as well, the usual approach is an
egress allowlist: start the container with `--cap-add=NET_ADMIN`, and have an
init script install `iptables` rules that drop everything except DNS and the
Anthropic API endpoints. That is a meaningful amount of extra machinery and it
breaks `pip install` and `apt`, so it is deliberately not wired up here.
