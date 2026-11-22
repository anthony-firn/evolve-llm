#!/usr/bin/env python3
"""Pop!_OS local connector. No dependencies beyond Python, bubblewrap and Zenity."""
import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ORIGIN = 'https://www.evolvetech.dev'
PORT = 8791
LIMIT = 65536

# Runs after Polkit authentication. EOF on stdin cancels even a root command; the
# unprivileged connector never needs permission to signal a root process.
ROOT_RUNNER = r"""
import os, select, signal, subprocess, sys, time
p = subprocess.Popen(['/bin/bash', '--noprofile', '--norc', '-c', sys.argv[1]], stdin=subprocess.DEVNULL, start_new_session=True)
end = time.monotonic() + 60
try:
    while p.poll() is None:
        readable, _, _ = select.select([0], [], [], 0.2)
        if time.monotonic() >= end or (readable and not os.read(0, 1)):
            os.killpg(p.pid, signal.SIGKILL)
            break
    code = p.wait()
finally:
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass
sys.exit(code if code >= 0 else 128 - code)
"""


def sandbox(root, command):
    # No host home, /etc, network, session bus, credentials or inherited environment.
    return ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
            '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
            '--dir', '/home', '--dir', '/home/agent', '--bind', str(root), '/workspace',
            '--chdir', '/workspace', '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
            '--setenv', 'HOME', '/home/agent', '--setenv', 'LANG', 'C.UTF-8',
            '/bin/bash', '--noprofile', '--norc', '-c', command]


# ---- Project mode: build and test one GitHub repository for the browser's VM ----
#
# The VM keeps the clone, git and GitHub. Here a separate copy of the repository (never the
# person's own checkout) gets the VM's files, and commands run on it in bubblewrap: no network, no
# home folder, the person's toolchains read-only, and only the copy writable except its .git, so
# nothing run there can plant a hook or config for the git commands this connector runs.

PROJECTS = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share') / 'evolve-local' / 'projects'
RUN_LIMIT = 30 * 60          # seconds per command
OUTPUT_LIMIT = 64 << 20      # bytes of output before a command is stopped
MEMORY = '6G'
GRANT = 3600                 # one approval covers this many seconds of commands
GIT = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false']


def blob_sha(data):
    import hashlib
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def find_tools(home=None):
    """The person's toolchains outside /usr, bound read-only at the same paths in the sandbox."""
    import shutil
    home = Path(home or Path.home())
    tools = {'binds': [], 'path': []}
    rustup = Path(os.environ.get('RUSTUP_HOME') or home / '.rustup')
    cargo = home / '.cargo'
    if rustup.is_dir():
        tools['binds'].append(str(rustup)); tools['rustup'] = str(rustup)
    if (cargo / 'bin').is_dir():
        tools['cargo_bin'] = str(cargo / 'bin')
    if (cargo / 'registry').is_dir():
        tools['registry'] = str(cargo / 'registry')
    if (cargo / 'git').is_dir():
        tools['cargo_git'] = str(cargo / 'git')  # dependencies taken from git repositories
    node = shutil.which('node')
    nodes = sorted((home / '.nvm' / 'versions' / 'node').glob('v*/bin/node')) if not node else []
    node = Path(node).resolve() if node else (nodes[-1] if nodes else None)
    if node and not str(node).startswith('/usr/'):
        tools['binds'].append(str(node.parent.parent)); tools['path'].append(str(node.parent))
    python = shutil.which('python3')
    if python and str(Path(python).resolve()).startswith(str(home) + '/'):
        prefix = Path(python).resolve().parent.parent
        tools['binds'].append(str(prefix)); tools['path'].append(str(prefix / 'bin'))
    if Path('/opt/google/chrome').is_dir():
        tools['binds'].append('/opt/google/chrome')
    return tools


def project_sandbox(copy, tools, cwd, command, network=False):
    a = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
         '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin',
         '--symlink', 'usr/lib', '/lib', '--symlink', 'usr/lib64', '/lib64',
         '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/home/agent',
         '--bind', str(copy), '/workspace', '--ro-bind', str(Path(copy) / '.git'), '/workspace/.git']
    for etc in ('/etc/fonts', '/etc/ld.so.cache', '/etc/alternatives'):
        a += ['--ro-bind-try', etc, etc]
    if network:
        a += ['--share-net', '--ro-bind-try', '/etc/resolv.conf', '/etc/resolv.conf', '--ro-bind-try', '/etc/ssl', '/etc/ssl',
              '--ro-bind-try', '/etc/ca-certificates', '/etc/ca-certificates', '--ro-bind-try', '/etc/hosts', '/etc/hosts']
    for b in tools.get('binds', []):
        a += ['--ro-bind', b, b]
    path = list(tools.get('path', []))
    env = {'HOME': '/home/agent', 'LANG': 'C.UTF-8', 'CI': '1', 'npm_config_cache': '/tmp/npm', 'npm_config_update_notifier': 'false'}
    if 'cargo_bin' in tools:
        # The agent's own CARGO_HOME: crates read-only from the person's registry, nothing fetched.
        a += ['--dir', '/home/agent/cargo', '--ro-bind', tools['cargo_bin'], '/home/agent/cargo/bin']
        if 'registry' in tools:
            a += ['--ro-bind', tools['registry'], '/home/agent/cargo/registry']
        if 'cargo_git' in tools:
            a += ['--ro-bind', tools['cargo_git'], '/home/agent/cargo/git']
        path.insert(0, '/home/agent/cargo/bin')
        env.update(CARGO_HOME='/home/agent/cargo', CARGO_NET_OFFLINE='true')
    if 'rustup' in tools:
        env['RUSTUP_HOME'] = tools['rustup']
    env['PATH'] = ':'.join(path + ['/usr/bin', '/bin'])
    a += ['--chdir', '/workspace/' + cwd if cwd else '/workspace', '--clearenv']
    for k, v in env.items():
        a += ['--setenv', k, v]
    return a + ['/bin/bash', '--noprofile', '--norc', '-c', command]


def safe_parts(path):
    if not isinstance(path, str) or not path or path.startswith('/') or '\\' in path or '\x00' in path:
        raise ValueError(f'Invalid project path: {path!r}')
    parts = path.split('/')
    if any(p in ('', '.', '..') for p in parts) or parts[0] == '.git':
        raise ValueError(f'Invalid project path: {path!r}')
    return parts


class Project:
    """One repository's copy: sync files from the VM, run commands on it in the sandbox."""

    def __init__(self, repo, approve, url=None, tools=None, base=None, limit=RUN_LIMIT, memory=MEMORY):
        import re
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo or '') or '..' in repo:
            raise ValueError('Give the repository as OWNER/NAME.')
        self.repo, self.approve, self.limit, self.memory = repo, approve, limit, memory
        self.url = url or f'https://github.com/{repo}.git'
        self.copy = Path(base or PROJECTS) / repo
        self.tools = find_tools() if tools is None else tools
        self.granted_until = 0
        self.unknown = set()
        self.runs = {}
        self.process = None
        self.lock = threading.Lock()

    def git(self, *args, check=True):
        p = subprocess.run(GIT + ['-C', str(self.copy), *args], capture_output=True, text=True, timeout=900,
                           env={'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()), 'GIT_TERMINAL_PROMPT': '0'})
        if check and p.returncode:
            raise ValueError(f'git {args[0]} failed: {p.stderr.strip()[:500]}')
        return p.stdout

    def prepare(self, setup=False):
        """Clone the copy if missing; with setup, install dependencies (network allowed, once)."""
        if not (self.copy / '.git').is_dir():
            self.copy.parent.mkdir(parents=True, exist_ok=True)
            p = subprocess.run(GIT + ['clone', '--quiet', self.url, str(self.copy)], capture_output=True, text=True, timeout=1800,
                               env={'PATH': '/usr/bin:/bin', 'HOME': str(Path.home()), 'GIT_TERMINAL_PROMPT': '0'})
            if p.returncode:
                raise ValueError(f'Could not clone {self.repo}: {p.stderr.strip()[:500]}')
            setup = True
        if setup and (self.copy / 'package-lock.json').exists():
            subprocess.run(project_sandbox(self.copy, self.tools, '', 'npm ci --ignore-scripts --no-audit --no-fund', network=True), check=True, timeout=1800)
        if setup and 'cargo_bin' in self.tools:
            # Every Rust project in the repository (a Cargo.lock beside its Cargo.toml): crates and
            # git dependencies go into the person's own cargo home, as their own build would.
            for lock in self.git('ls-files', '-z', '--', 'Cargo.lock', '*/Cargo.lock').split('\0'):
                if lock and (self.copy / lock).with_name('Cargo.toml').exists():
                    subprocess.run([str(Path(self.tools['cargo_bin']) / 'cargo'), 'fetch', '--locked', '--manifest-path', str((self.copy / lock).with_name('Cargo.toml'))], check=True, timeout=1800)

    def opendir(self, parts, create=False):
        """A directory of the copy, opened without following any symlink."""
        fd = os.open(self.copy, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd
        except BaseException:
            os.close(fd)
            raise

    def file_sha(self, path):
        parts = safe_parts(path)
        try:
            d = self.opendir(parts[:-1])
        except OSError:
            return None
        try:
            import stat
            st = os.stat(parts[-1], dir_fd=d, follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                return blob_sha(os.readlink(parts[-1], dir_fd=d).encode())
            if not stat.S_ISREG(st.st_mode):
                return None
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=d)
            with os.fdopen(fd, 'rb') as f:
                return blob_sha(f.read())
        except OSError:
            return None
        finally:
            os.close(d)

    def write(self, path, data):
        parts = safe_parts(path)
        d = self.opendir(parts[:-1], create=True)
        temp = '.evolve-' + secrets.token_hex(8)
        try:
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=d)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            os.rename(temp, parts[-1], src_dir_fd=d, dst_dir_fd=d)
        finally:
            try:
                os.unlink(temp, dir_fd=d)
            except FileNotFoundError:
                pass
            os.close(d)

    def delete(self, path):
        parts = safe_parts(path)
        try:
            d = self.opendir(parts[:-1])
        except OSError:
            return
        try:
            os.unlink(parts[-1], dir_fd=d)
        except (FileNotFoundError, IsADirectoryError):
            pass
        finally:
            os.close(d)

    def sync(self, data):
        """Make the copy hold the VM's files: {repo, commit, files: {path: sha}, contents: {path: base64}}.
        Answers {need: [paths]} until every file matches, then {ok: true}."""
        import base64, re
        if data.get('repo') != self.repo:
            raise ValueError(f'This computer builds {self.repo}, not {data.get("repo")}.')
        commit, files, contents = data.get('commit'), data.get('files'), data.get('contents') or {}
        if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit) or not isinstance(files, dict) or len(files) > 100000:
            raise ValueError('Invalid sync request.')
        for path, sha in files.items():
            safe_parts(path)
            if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
                raise ValueError('Invalid sync request.')
        with self.lock:
            # The base commit only saves sending files: one that cannot be fetched (not pushed
            # anywhere this copy can reach) leaves the checkout as it is, and the files still match.
            if self.git('rev-parse', 'HEAD').strip() != commit and commit not in self.unknown:
                known = lambda: self.git('cat-file', '-t', commit, check=False).strip() == 'commit'
                if not known():
                    self.git('fetch', '--quiet', 'origin', commit, check=False)
                if known():
                    self.git('checkout', '--quiet', '--force', '--detach', commit)
                    self.git('clean', '-fdq')
                else:
                    self.unknown.add(commit)
            for path, b64 in contents.items():
                raw = base64.b64decode(b64)
                if files.get(path) != blob_sha(raw):
                    raise ValueError(f'{path}: contents do not match their hash.')
                self.write(path, raw)
            present = [p for p in self.git('ls-files', '-z', '--cached', '--others', '--exclude-standard').split('\0') if p]
            for path in present:
                if path not in files:
                    self.delete(path)
            need = [path for path, sha in files.items() if self.file_sha(path) != sha]
            return {'need': need[:2000]} if need else {'ok': True}

    def start(self, data):
        command, cwd = data.get('command'), data.get('cwd') or ''
        if not isinstance(command, str) or not command.strip() or len(command) > 16384:
            raise ValueError('A command of at most 16384 characters is required.')
        if cwd:
            os.close(self.opendir(safe_parts(cwd)))
        if time.time() > self.granted_until:
            if not self.approve(f'Allow the assistant to build and test {self.repo} on this computer for the next hour?\n\n'
                                f'Commands run in a sandbox: no network, no access to your home folder or keys, and only the '
                                f"assistant's own copy of the project ({self.copy}) can change. Each command is limited to "
                                f'{self.memory}B of memory and {self.limit // 60} minutes.\n\nFirst command:\n{command}'):
                raise ValueError('Building on this computer was declined.')
            self.granted_until = time.time() + GRANT
        if any(r['done'] is False for r in self.runs.values()):
            raise ValueError('Another command is running on this computer.')
        import shutil, tempfile
        argv = project_sandbox(self.copy, self.tools, cwd, command)
        if shutil.which('systemd-run'):
            argv = ['systemd-run', '--user', '--scope', '--quiet', '--collect', '-p', f'MemoryMax={self.memory}', '-p', 'CPUWeight=20', '--'] + argv
        out = tempfile.TemporaryFile()
        p = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                             env={'PATH': '/usr/bin:/bin', 'XDG_RUNTIME_DIR': os.environ.get('XDG_RUNTIME_DIR', ''),
                                  'DBUS_SESSION_BUS_ADDRESS': os.environ.get('DBUS_SESSION_BUS_ADDRESS', '')})
        run_id = secrets.token_hex(8)
        run = {'out': out, 'done': False, 'code': None, 'note': ''}
        self.close()
        self.runs[run_id] = run
        self.process = p

        def wait():
            end = time.monotonic() + self.limit
            while p.poll() is None:
                if time.monotonic() > end or out.tell() > OUTPUT_LIMIT:
                    run['note'] = f'\n[stopped: over {self.limit // 60} minutes]\n' if time.monotonic() > end else '\n[stopped: too much output]\n'
                    self.kill()
                time.sleep(0.2)
            run['code'] = p.returncode if p.returncode >= 0 else 128 - p.returncode
            run['done'] = True
        threading.Thread(target=wait, daemon=True).start()
        return {'id': run_id}

    def poll(self, data):
        run = self.runs.get(data.get('id'))
        if not run:
            raise ValueError('No such command.')
        offset = int(data.get('offset') or 0)
        done = run['done']
        f = run['out']
        end = f.seek(0, 2)
        f.seek(offset)
        chunk = f.read(min(end - offset, 256 << 10))
        f.seek(end)
        new = offset + len(chunk)
        text = chunk.decode('utf-8', 'replace')
        finished = done and new >= end
        if finished:
            text += run['note']
        return {'output': text, 'offset': new, 'done': finished, 'code': run['code'] if finished else None}

    def close(self):
        """Forget finished commands and their output."""
        for k, v in list(self.runs.items()):
            if v['done']:
                v['out'].close()
                del self.runs[k]

    def kill(self):
        p = self.process
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def status(self):
        return {'ok': True, 'project': self.repo, 'root': str(self.copy),
                'sandbox': f'bubblewrap; network off; your toolchains read-only; {self.memory}B memory; {self.limit // 60} minutes per command; one approval per hour'}


class Connector:
    def __init__(self, root, approve=None, full_machine=False):
        self.full_machine = full_machine
        self.root = Path(root).resolve(strict=True)
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.process = None
        self.root_process = False
        self.cancelled = threading.Event()
        self.approve = approve or self.dialog

    def dialog(self, label):
        # Use a text file on stdin: Zenity markup cannot disguise the proposed operation.
        p = subprocess.Popen(['zenity', '--text-info', '--title=Evolve Local: approve operation',
                              '--width=760', '--height=500', '--checkbox=Allow this operation',
                              '--ok-label=Allow', '--cancel-label=Deny'], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.process = p
        try:
            footer = '' if getattr(self, 'project', None) else '\n\nReview carefully: full-machine mode can access this computer. Root commands require administrator authentication.'
            p.communicate((f'Project: {self.root}\n\n{label}{footer}\n').encode(), timeout=120)
            return p.returncode == 0 and not self.cancelled.is_set()
        except subprocess.TimeoutExpired:
            self.stop()
            return False
        finally:
            self.process = None

    def stop(self):
        self.cancelled.set()
        p = self.process
        if p and p.poll() is None:
            if self.root_process:
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def parent(self, path):
        if not isinstance(path, str):
            raise ValueError('A file path is required.')
        if self.full_machine and path.startswith('/'):
            path = path[1:]
        elif path.startswith('/workspace/'):
            path = path[len('/workspace/'):]
        parts = path.split('/')
        if not parts or any(p in ('', '.', '..') for p in parts) or '\\' in path or '\x00' in path:
            raise ValueError('Use a relative project path; traversal is forbidden.')
        fd = os.dup(self.fd)
        try:
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            return fd, parts[-1]
        except BaseException:
            os.close(fd)
            raise

    def read(self, path):
        parent, name = self.parent(path)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            import stat
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError('Only regular text files are supported.')
            with os.fdopen(fd, 'rb') as f:
                data = f.read(LIMIT + 1)
            if len(data) > LIMIT:
                raise ValueError('File exceeds 64 KiB; use an approved shell command for selective reads.')
            return data.decode('utf-8')
        finally:
            os.close(parent)

    def operation(self, kind, data):
        project = getattr(self, 'project', None)
        if project:
            if kind == 'status':
                return project.status()
            if kind == 'cancel':
                project.kill()
                return {'ok': True}
            if kind in ('project-sync', 'project-run', 'project-poll'):
                return {'project-sync': project.sync, 'project-run': project.start, 'project-poll': project.poll}[kind](data)
            raise ValueError(f'This computer is paired for building {project.repo}: files are changed in the browser, and commands run with cargo, npm, node or native there.')
        if kind == 'status':
            return {'ok': True, 'root': str(self.root), 'fullMachine': self.full_machine, 'sandbox': 'Full machine; root via Polkit; local approval for every operation' if self.full_machine else 'bubblewrap; network off; approval for commands and writes'}
        if kind == 'cancel':
            self.stop()
            return {'ok': True}
        if not self.lock.acquire(blocking=False):
            raise ValueError('Another operation is running.')
        self.cancelled.clear()
        try:
            if kind == 'read':
                if self.full_machine and not self.approve('READ ' + str(data.get('path'))):
                    raise ValueError('Read denied on this computer.')
                return {'text': self.read(data.get('path'))}
            if kind in ('write', 'edit'):
                path = data.get('path')
                text = data.get('text')
                if kind == 'edit':
                    old = data.get('oldText')
                    current = self.read(path)
                    if not isinstance(old, str) or not old or current.count(old) != 1:
                        raise ValueError('oldText must match exactly once.')
                    text = current.replace(old, data['newText'], 1)
                if not isinstance(text, str) or len(text.encode()) > LIMIT:
                    raise ValueError('Write must be UTF-8 text smaller than 64 KiB.')
                if not self.approve(f'{kind.upper()} {path}\n\nProposed complete contents:\n{text}'):
                    raise ValueError('Operation denied on this computer.')
                if self.cancelled.is_set():
                    raise ValueError('Operation cancelled.')
                parent, name = self.parent(path)
                temp = '.evolve-' + secrets.token_hex(12)
                try:
                    # Atomic replacement avoids symlink and hardlink writes into other files.
                    mode = 0o644
                    try:
                        old_stat = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        import stat
                        if not stat.S_ISREG(old_stat.st_mode):
                            raise ValueError('Destination must be a regular file.')
                        mode = old_stat.st_mode & 0o777
                    except FileNotFoundError:
                        pass
                    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode, dir_fd=parent)
                    with os.fdopen(fd, 'w') as f:
                        f.write(text)
                    os.rename(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
                finally:
                    try:
                        os.unlink(temp, dir_fd=parent)
                    except FileNotFoundError:
                        pass
                    os.close(parent)
                return {'ok': True}
            if kind == 'exec':
                command = data.get('command')
                if not isinstance(command, str) or not command or len(command) > 16384:
                    raise ValueError('A command of at most 16384 characters is required.')
                as_root = data.get('asRoot') is True
                if as_root and not self.full_machine:
                    raise ValueError('Root execution requires full-machine mode.')
                label = ('RUN AS ROOT' if as_root else 'RUN') + (' on this computer' if self.full_machine else ' in /workspace')
                if not self.approve(label + ' (60 second limit):\n\n' + command):
                    raise ValueError('Command denied on this computer.')
                if self.cancelled.is_set():
                    raise ValueError('Operation cancelled.')
                # Disk-backed output avoids unbounded RAM for noisy commands; RLIMIT_FSIZE caps it.
                import tempfile
                import resource
                def limits():
                    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
                    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
                with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                    argv = sandbox(self.root, command)
                    child_env = {'PATH': '/usr/bin:/bin'}
                    if self.full_machine:
                        argv = ['/bin/bash', '--noprofile', '--norc', '-c', command]
                        child_env['HOME'] = str(Path.home())
                        if as_root:
                            argv = ['/usr/bin/pkexec', '/usr/bin/python3', '-I', '-c', ROOT_RUNNER, command]
                            for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY', 'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS'):
                                if key in os.environ:
                                    child_env[key] = os.environ[key]
                    p = subprocess.Popen(argv, stdout=out, stderr=err, cwd=self.root,
                                         stdin=subprocess.PIPE if as_root else subprocess.DEVNULL, env=child_env,
                                         start_new_session=True, preexec_fn=limits)
                    self.root_process = as_root
                    self.process = p
                    if self.cancelled.is_set():
                        self.stop()
                    try:
                        p.wait(timeout=180 if as_root else 60)
                    except subprocess.TimeoutExpired:
                        self.stop()
                        p.wait()
                        raise ValueError('Command exceeded 60 seconds and was stopped.')
                    finally:
                        self.process = None
                        self.root_process = False
                        if p.stdin and not p.stdin.closed:
                            p.stdin.close()
                    out.seek(0); err.seek(0)
                    return {'code': p.returncode, 'stdout': out.read(LIMIT).decode('utf-8', 'replace'),
                            'stderr': err.read(LIMIT).decode('utf-8', 'replace'), 'cwd': str(self.root) if self.full_machine else '/workspace'}
            raise ValueError('Unsupported operation.')
        finally:
            self.lock.release()


def handler(connector, origins=(ORIGIN,)):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log pairing credentials or project contents.

        def allowed(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}' and self.headers.get('Origin') in origins

        def respond(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            if self.allowed():
                self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin'))
                self.send_header('Vary', 'Origin')
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            if not self.allowed():
                return self.respond(403, {'error': 'Origin or Host denied.'})
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin'))
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'POST')
            self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
            self.send_header('Access-Control-Allow-Private-Network', 'true')
            self.end_headers()

        def do_POST(self):
            if not self.allowed() or not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + connector.token):
                return self.respond(403, {'error': 'Connection denied. Reopen the connector pairing link.'})
            if self.path != '/rpc' or self.headers.get('Content-Type') != 'application/json':
                return self.respond(400, {'error': 'Invalid request.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= (16 << 20 if getattr(connector, 'project', None) else 200000):
                    raise ValueError('Invalid request length.')
                self.connection.settimeout(30)
                body = json.loads(self.rfile.read(length))
                result = connector.operation(body['type'], body.get('data', {}))
                self.respond(200, result)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond(400, {'error': str(error)})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, help='Dedicated project folder exposed to the agent')
    ap.add_argument('--project', help='OWNER/NAME: build and test this GitHub repository for the browser (its own copy, sandboxed)')
    ap.add_argument('--setup', action='store_true', help='With --project: install its dependencies now (uses the network)')
    ap.add_argument('--origin', action='append', default=[], help='Also accept this site, such as http://localhost:8080 for a local build')
    ap.add_argument('--no-open', action='store_true')
    ap.add_argument('--full-machine', action='store_true', help='Expose host filesystem; root commands require Polkit authentication')
    args = ap.parse_args()
    if bool(args.root) == bool(args.project) or (args.project and args.full_machine):
        ap.error('Give either --root FOLDER (optionally --full-machine) or --project OWNER/NAME.')
    origins = tuple(args.origin) + (ORIGIN,)
    import shutil
    for executable in ('bwrap', 'zenity'):
        if not shutil.which(executable):
            ap.error(f'{executable} must be installed.')
    if args.project:
        project = Project(args.project, approve=None)
        project.prepare(setup=args.setup)
        connector = Connector(project.copy)
        project.approve = connector.dialog
        connector.project = project
        root = project.copy
    else:
        root = args.root.resolve(strict=True)
        if not args.full_machine and (root == Path.home() or root == Path('/')):
            ap.error('Choose a project folder, not your home or system root.')
        if not args.full_machine:
            subprocess.run(sandbox(root, 'true'), check=True, env={'PATH': '/usr/bin:/bin'})
        connector = Connector(root, full_machine=args.full_machine)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), handler(connector, origins))
    server.daemon_threads = True
    runtime = Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / f'evolve-local-{os.getuid()}.url'
    fd = os.open(runtime, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    url = origins[0] + '/play/evolve-llm/#local=' + connector.token
    with os.fdopen(fd, 'w') as f:
        f.write(url + '\n')
    if not args.no_open:
        import webbrowser
        webbrowser.open(url)
    print(f'Evolve Local on 127.0.0.1:{PORT}; project: {root}; pairing link: {runtime}', flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        connector.stop()
        if args.project:
            connector.project.kill()
        runtime.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
