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
            p.communicate((f'Project: {self.root}\n\n{label}\n\nReview carefully: full-machine mode can access this computer. Root commands require administrator authentication.\n').encode(), timeout=120)
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
                return
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


def handler(connector):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log pairing credentials or project contents.

        def allowed(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}' and self.headers.get('Origin') == ORIGIN

        def respond(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            if self.allowed():
                self.send_header('Access-Control-Allow-Origin', ORIGIN)
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
            self.send_header('Access-Control-Allow-Origin', ORIGIN)
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
                if not 0 < length <= 200000:
                    raise ValueError('Invalid request length.')
                self.connection.settimeout(5)
                body = json.loads(self.rfile.read(length))
                result = connector.operation(body['type'], body.get('data', {}))
                self.respond(200, result)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond(400, {'error': str(error)})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True, help='Dedicated project folder exposed to the agent')
    ap.add_argument('--no-open', action='store_true')
    ap.add_argument('--full-machine', action='store_true', help='Expose host filesystem; root commands require Polkit authentication')
    args = ap.parse_args()
    import shutil
    for executable in ('bwrap', 'zenity'):
        if not shutil.which(executable):
            ap.error(f'{executable} must be installed.')
    root = args.root.resolve(strict=True)
    if not args.full_machine and (root == Path.home() or root == Path('/')):
        ap.error('Choose a project folder, not your home or system root.')
    if not args.full_machine:
        subprocess.run(sandbox(root, 'true'), check=True, env={'PATH': '/usr/bin:/bin'})
    connector = Connector(root, full_machine=args.full_machine)
    server = ThreadingHTTPServer(('127.0.0.1', PORT), handler(connector))
    server.daemon_threads = True
    runtime = Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp')) / f'evolve-local-{os.getuid()}.url'
    fd = os.open(runtime, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    url = ORIGIN + '/play/evolve-llm/#local=' + connector.token
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
        runtime.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
