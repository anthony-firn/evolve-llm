# Evolve Local: Pop!_OS MVP

Connect the live Evolve LLM browser agent to this computer using Python's standard
library, Zenity and Polkit. There is no SSH server, public tunnel, cloud execution,
package download at startup, or background login autostart.

Install once from this checkout with `python3 companion/install.py`. Open **Evolve
Local** from the application launcher. It starts the user service and opens the
live site with a private pairing link. Allow the browser's local-device connection
prompt. Load a tool-capable coding model, then use **Local computer → Ask Pi**.
**The main chat** works on the connected computer, with its steps shown under each
reply and Stop cancelling the active operation. Follow-up messages include recent
conversation context. Disconnect to return the chat to the workspace in the browser.

**Run command** works without a model. Select **Run direct command as root** only
for operations that require administrator rights.

Full-machine mode exposes the host with your user permissions. Every file read,
write, edit, and command requires a desktop approval showing the actual operation.
Root commands additionally use Polkit authentication. Read/write tools use ordinary
user permissions; use an approved root shell command for privileged files. Command
working directory is `/`; specify paths or `cd /path/to/project && ...` explicitly.
Commands are noninteractive, output is returned on completion, and execution is
limited to 60 seconds (root authentication has additional time). Long builds,
streaming output, automatic dependency installation and unattended operation are
outside this first version.

Full root access is inherently capable of compromising or permanently modifying
this system. Local approval is the authorization boundary, not a guarantee that an
approved command is safe. Root commands can alter these controls. Never approve a
command you do not understand. Output and approved file reads go to the browser
model. The connector never sends them to an EvolveTech server, but other enabled
site features have their own network behavior.

The service binds only `127.0.0.1:8791`, requires the exact production website
Origin and loopback Host, and authenticates all operations with a random 256-bit
session credential. Credentials are kept in tab memory and a mode-0600 pairing
file in the user's runtime directory, never localStorage or request logs. Pairing
fragments are removed from the address bar immediately. Reopening the launcher
reuses the current service session; restarting the service rotates its credential.
No credentials or local file contents belong in either source repository.

**Stop** cancels the active operation. **Disconnect** clears this tab's credential.
To revoke all tabs, run `~/.local/bin/evolve-local stop` or
`systemctl --user restart evolve-local.service`. Reopen the launcher to pair again.
The service is installed but not enabled at login.

Optional restricted mode: stop the service and run
`python3 companion/local.py --root /path/to/dedicated/project`. Commands then use
bubblewrap with only that folder writable, no host home and no network. Do not
put secrets, hardlinks to sensitive files, or untrusted mount points in that folder.

## Project mode: build and test for the browser's Linux VM

The VM in the browser keeps the clone, `git` and GitHub, but has no compiler or Node. Pair this
computer for one repository and its builds and tests run here instead of waiting on CI:

```text
python3 companion/local.py --project OWNER/NAME [--setup] [--origin http://localhost:8080]
```

The connector keeps its own clone of the repository under
`~/.local/share/evolve-local/projects/OWNER/NAME` (never your checkout). In a clone of that
repository in the chat or the Linux terminal, `cargo`, `npm`, `npx` and `node` run here, and
`native COMMAND` runs any other command here (`native python3 tests/browser_test.py`). Each run
sends the clone's base commit and file hashes; only the files that differ are sent and written,
so builds stay incremental. Output streams back to the terminal.

Commands run in bubblewrap: no network, no home folder, your toolchains (rustup, cargo and its
crate registry, Node, a Python and Chrome installed under your home or `/opt`) read-only at their
own paths, `cargo` offline, and only the copy writable except its `.git`, so nothing run there can
plant a hook or config for the `git` commands the connector runs (with hooks and fsmonitor off).
Memory is capped at 6 GB with lower CPU priority (systemd-run), and a command stops after 30
minutes or 64 MB of output. One desktop approval covers an hour of commands: there is no
reviewing what `cargo test` runs, and the sandbox is the boundary. Results reach your own checkout
only through pull requests you review.

The first clone and `--setup` install dependencies with the network (`npm ci --ignore-scripts`,
and `cargo fetch` into your own crate registry, as your own build would). A new dependency needs
`--setup` again. `--origin` accepts another site, such as a local build on port 8080. This is
Linux only (bubblewrap); elsewhere the VM's `gh run watch` shows CI.

Validate with `python3 -m unittest discover -s companion -p test_local.py`,
`python3 companion/browser_test.py` (connector must be stopped), and
`python3 companion/main_chat_test.py` (connector stopped),
`python3 tests/native_build_browser_test.py` (connector stopped), and
`node --test tests/pi-agent.test.mjs`. The browser fixture uses real transport and
Pi with a scripted model, a temporary folder, and test-only approvals; it never
authenticates as root. Actual desktop/Polkit approval is tested manually.

The long-term plan is signed Windows/macOS/Linux installers, project selection,
automatic pairing, streamed terminals, change review, stronger optional isolation,
and an SSH backend for remote machines. This MVP deliberately keeps the existing
browser inference and deployment pipeline.
