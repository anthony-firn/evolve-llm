#!/usr/bin/env python3
"""Install this checkout's opt-in Pop!_OS connector and desktop launcher for the current user."""
from pathlib import Path
import os
import shlex
import subprocess

root = Path(__file__).resolve().parent
home = Path.home()
unit_dir = home / '.config/systemd/user'
bin_dir = home / '.local/bin'
apps = home / '.local/share/applications'
for directory in (unit_dir, bin_dir, apps):
    directory.mkdir(parents=True, exist_ok=True)
# Fixed command uses the Python stdlib only, independent of an activated environment.
(unit_dir / 'evolve-local.service').write_text(f'''[Unit]
Description=Evolve Local (full machine, local approval, root via Polkit)
[Service]
Type=simple
ExecStart=/usr/bin/python3 "{root / 'local.py'}" --root / --full-machine --no-open
Restart=no
KillMode=control-group
TimeoutStopSec=5
''')
launcher = bin_dir / 'evolve-local'
launcher.write_text('''#!/bin/sh
set -eu
if [ "${1:-}" = stop ]; then
  systemctl --user stop evolve-local.service
  exit
fi
systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS 2>/dev/null || true
systemctl --user start evolve-local.service
pair_file="${XDG_RUNTIME_DIR}/evolve-local-$(id -u).url"
i=0
while [ ! -s "$pair_file" ] && [ "$i" -lt 30 ]; do
  sleep 0.1
  i=$((i+1))
done
[ -s "$pair_file" ] || { echo 'Connector did not start. Run: journalctl --user -u evolve-local.service'; exit 1; }
xdg-open "$(cat "$pair_file")" >/dev/null 2>&1
''')
launcher.chmod(0o700)
(apps / 'evolve-local.desktop').write_text(f'''[Desktop Entry]
Type=Application
Name=Evolve Local
Comment=Connect Evolve LLM to this computer with local approval
Exec={launcher}
Icon=utilities-terminal
Terminal=false
Categories=Development;
''')
subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
print('Installed Evolve Local desktop launcher. No login autostart. Stop: ~/.local/bin/evolve-local stop')
