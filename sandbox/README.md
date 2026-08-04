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
./sandbox/xcdev.sh build
```

Under rootless podman the script adds `--userns=keep-id`, which maps your host
UID to the same UID inside. Without it your host user maps to container root
and everything else to a subuid, and the mode-0600 credentials file mounted at
`/seed` would be unreadable to the unprivileged `dev` user in the container.

## Use

```sh
./sandbox/xcdev.sh sync-wip     # optional: carry uncommitted changes across
./sandbox/xcdev.sh claude       # Claude, all permissions on
```

`claude` passes any extra arguments straight through, so
`./sandbox/xcdev.sh claude -p "run the stage 1 tests"` works.

Other subcommands:

```sh
./sandbox/xcdev.sh shell             # bash in the sandbox
./sandbox/xcdev.sh run pytest -q     # one-off command
./sandbox/xcdev.sh log               # commits made inside the sandbox
./sandbox/xcdev.sh ls [path]         # list files inside the sandbox
./sandbox/xcdev.sh export            # git bundle of committed work -> host
./sandbox/xcdev.sh fetch data -o r.tar   # result files -> host
./sandbox/xcdev.sh reset             # discard the sandbox work volume
./sandbox/xcdev.sh reset-all         # discard the work volume AND the login
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
./sandbox/xcdev.sh export                        # writes sandbox-export.bundle
git bundle list-heads sandbox-export.bundle     # see what is in it
git fetch sandbox-export.bundle stage1:sandbox-stage1
git log sandbox-stage1                          # review before merging
```

**Result files** — plots, `.npz` arrays, `optuna*.db`, anything gitignored and
so *not* in the bundle. Look first, then pull what you want as a tar:

```sh
./sandbox/xcdev.sh ls                      # what is in the sandbox checkout
./sandbox/xcdev.sh ls data/stage1          # drill in
./sandbox/xcdev.sh fetch data figures -o results.tar
tar -tvf results.tar                      # inspect before unpacking
tar -xf results.tar -C /tmp/from-sandbox
```

Paths are relative to the sandbox checkout. `fetch` writes
`sandbox-data.tar` next to the repo unless you pass `-o`.

**Straight off the volume** — the volumes are ordinary directories under
`~/.local/share/containers/storage/volumes/`, and `xcdev.sh` keeps them
readable from your own shell, so you can just `ls` and `cp`:

```sh
ls ~/.local/share/containers/storage/volumes/dft_xc_search-sandbox-work/_data/dft_xc_search
```

This needs the `unlock` step. A volume that podman auto-creates during `run`
gets its top-level directory owned by the first subuid (100000) with mode
0700, so your own shell cannot even traverse into it — even though `_data`
underneath is already yours and world-readable. `unlock` fixes that by
pre-creating volumes (which avoids it entirely) and, for any already in the
broken state, running `podman unshare chown 0:0` on that one directory —
inside the namespace uid 0 *is* your host uid. Podman does not undo it.

`build`, `reset`, `reset-all` and every container start do this automatically,
so it should already be true. To check or repair by hand:

```sh
./sandbox/xcdev.sh unlock
```

Nothing is weakened by this: files in the volume are already owned by you
thanks to `--userns=keep-id`, and the container still cannot write anywhere on
your drive outside its own volumes.

**Everything at once** — if you would rather grab the whole volume as one
archive, the engine can do that directly:

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
  newer Claude Code, rerun `./sandbox/xcdev.sh build`.
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
- The image is deliberately not minimal: man pages (the slim base excludes
  them at the dpkg level, so that is undone and the base packages reinstalled),
  networking tools (`ping`, `ip`, `ss`, `dig`, `nc`, `traceroute`, `netstat`),
  `vim`, `nano`, `tmux`, `htop`, `tree`, `file`, `rsync`, `strace`, `lsof`,
  `pstree`, archive tools and bash completion.
- **`ping` needs one host-side change.** Rootless containers get no
  `CAP_NET_RAW`, so `xcdev.sh` passes
  `--sysctl net.ipv4.ping_group_range=<gid> <gid>` to allow unprivileged ICMP
  sockets inside the container. That is necessary but not sufficient: the
  rootless network helper on the *host* also needs to open an ICMP socket, and
  Debian/Ubuntu ship `net.ipv4.ping_group_range = 1 0` (an empty range). The
  host `ping` works regardless because its binary has `cap_net_raw=ep`, but the
  helper has no such capability, so echo replies never come back. To enable it:

  ```sh
  sudo sysctl -w net.ipv4.ping_group_range="0 2147483647"
  # persist it:
  echo 'net.ipv4.ping_group_range = 0 2147483647' \
      | sudo tee /etc/sysctl.d/99-rootless-ping.conf
  ```

  Everything else — DNS, HTTPS, `traceroute` (UDP) — works without it.

## Tightening further

If you want to close the exfiltration path as well, the usual approach is an
egress allowlist: start the container with `--cap-add=NET_ADMIN`, and have an
init script install `iptables` rules that drop everything except DNS and the
Anthropic API endpoints. That is a meaningful amount of extra machinery and it
breaks `pip install` and `apt`, so it is deliberately not wired up here.
