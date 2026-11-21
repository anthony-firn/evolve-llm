# Evolve Local: Pop!_OS MVP

Connect the live Evolve LLM browser agent to this computer using Python's standard
library, Zenity and Polkit. There is no SSH server, public tunnel, cloud execution,
package download at startup, or background login autostart.

Install once from this checkout with `python3 companion/install.py`. Open **Evolve
Local** from the application launcher. It starts the user service and opens the
live site with a private pairing link. Allow the browser's local-device connection
prompt. Load a tool-capable coding model, then use **Local computer → Ask Pi**.
**Main chat** uses the connected computer in Auto and Build modes, with tool
activity shown in each reply and Stop cancelling the active operation. Follow-up
messages include recent conversation context. Chat mode has no computer tools.
Disconnect to return Auto/Build to the browser workspace.

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

Validate with `python3 -m unittest discover -s companion -p test_local.py`,
`python3 companion/browser_test.py` (connector must be stopped), and
`python3 companion/main_chat_test.py` (connector stopped), and
`node --test tests/pi-agent.test.mjs`. The browser fixture uses real transport and
Pi with a scripted model, a temporary folder, and test-only approvals; it never
authenticates as root. Actual desktop/Polkit approval is tested manually.

The long-term plan is signed Windows/macOS/Linux installers, project selection,
automatic pairing, streamed terminals, change review, stronger optional isolation,
and an SSH backend for remote machines. This MVP deliberately keeps the existing
browser inference and deployment pipeline.
