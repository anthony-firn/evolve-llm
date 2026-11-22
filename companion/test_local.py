import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

spec = importlib.util.spec_from_file_location('local', Path(__file__).with_name('local.py'))
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.c = local.Connector(self.root, approve=lambda _: True)

    def tearDown(self):
        self.c.stop()
        os.close(self.c.fd)
        self.tmp.cleanup()

    def test_files_and_denial(self):
        self.c.operation('write', {'path': '/workspace/test.txt', 'text': 'hello'})
        self.assertEqual(self.c.operation('read', {'path': 'test.txt'}), {'text': 'hello'})
        self.c.operation('edit', {'path': 'test.txt', 'oldText': 'hello', 'newText': 'world'})
        self.c.approve = lambda _: False
        with self.assertRaises(ValueError):
            self.c.operation('write', {'path': 'test.txt', 'text': 'bad'})
        self.assertEqual(self.c.read('test.txt'), 'world')
        with self.assertRaises(ValueError):
            self.c.operation('exec', {'command': 'touch forbidden'})
        self.assertFalse((self.root / 'forbidden').exists())

    def test_paths_symlinks_hardlinks(self):
        for path in ('../etc/passwd', '/etc/passwd', 'x/../../etc/passwd'):
            with self.assertRaises(ValueError):
                self.c.read(path)
        (self.root / 'escape').symlink_to('/etc/passwd')
        with self.assertRaises(OSError):
            self.c.read('escape')
        with self.assertRaises(ValueError):
            self.c.operation('write', {'path': 'escape', 'text': 'bad'})
        outside = self.root.parent / (self.root.name + '-outside')
        outside.write_text('unchanged')
        try:
            os.link(outside, self.root / 'hardlink')
            self.c.operation('write', {'path': 'hardlink', 'text': 'changed'})
            self.assertEqual(outside.read_text(), 'unchanged')
        finally:
            outside.unlink()

    def test_isolation(self):
        result = self.c.operation('exec', {'command': 'printf "ok" > made.txt; test ! -e /etc/passwd && test ! -e /home/anthony && python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect((chr(49)+chr(46)+chr(49)+chr(46)+chr(49)+chr(46)+chr(49),443))"'})
        self.assertNotEqual(result['code'], 0)
        self.assertIn('Network is unreachable', result['stderr'])
        self.assertEqual((self.root / 'made.txt').read_text(), 'ok')
        with self.assertRaises(ValueError):
            self.c.operation('exec', {'command': 'id', 'asRoot': True})

    def test_full_machine_requires_read_approval(self):
        self.c.full_machine = True
        (self.root / 'note').write_text('hello')
        self.c.approve = lambda _: False
        with self.assertRaises(ValueError):
            self.c.operation('read', {'path': 'note'})
        self.c.approve = lambda _: True
        result = self.c.operation('exec', {'command': 'id -u'})
        self.assertEqual(result['stdout'].strip(), str(os.getuid()))

    def test_privileged_supervisor_cancels_on_disconnect(self):
        import subprocess
        p = subprocess.Popen(['/usr/bin/python3', '-I', '-c', local.ROOT_RUNNER, 'sleep 30'], stdin=subprocess.PIPE)
        import time
        time.sleep(0.3)
        p.stdin.close()
        self.assertNotEqual(p.wait(timeout=3), 0)

    def test_origin_auth_host(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), local.handler(self.c))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f'http://127.0.0.1:{server.server_port}/rpc'
        def request(origin=local.ORIGIN, token=None, host=None):
            headers = {'Origin': origin, 'Authorization': 'Bearer ' + (token or self.c.token), 'Content-Type': 'application/json'}
            if host: headers['Host'] = host
            return urllib.request.urlopen(urllib.request.Request(url, data=b'{"type":"status"}', headers=headers))
        try:
            self.assertTrue(json.load(request())['ok'])
            for kwargs in ({'origin': 'https://evil.example'}, {'origin': 'null'}, {'token': 'wrong'}, {'host': 'evil.example'}):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request(**kwargs)
                self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown(); server.server_close()


def git(cwd, *args):
    import subprocess
    subprocess.run(['git', '-C', str(cwd), *args], check=True, capture_output=True,
                   env={'PATH': '/usr/bin:/bin', 'HOME': str(cwd), 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t', 'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'})


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.origin = base / 'origin'
        (self.origin / 'sub').mkdir(parents=True)
        (self.origin / 'README.md').write_text('hello\n')
        (self.origin / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
        (self.origin / 'sub' / 'keep.txt').write_text('keep\n')
        (self.origin / '.gitignore').write_text('target/\n')
        git(self.origin, 'init', '-q', '-b', 'main'); git(self.origin, 'add', '-A'); git(self.origin, 'commit', '-qm', 'first')
        self.commit = local.subprocess.run(['git', '-C', str(self.origin), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        self.asked = []
        self.p = local.Project('demo/app', approve=lambda q: self.asked.append(q) or True, url=str(self.origin), base=base / 'projects',
                               tools={'binds': [], 'path': []}, memory='1G')
        self.p.prepare()
        self.files = {'README.md': local.blob_sha(b'hello\n'), 'calc.py': local.blob_sha(b'def add(a, b):\n    return a - b\n'),
                      'sub/keep.txt': local.blob_sha(b'keep\n'), '.gitignore': local.blob_sha(b'target/\n')}

    def tearDown(self):
        self.p.kill()
        import time
        for _ in range(50):
            if all(r['done'] for r in self.p.runs.values()):
                break
            time.sleep(0.1)
        self.p.close()
        self.tmp.cleanup()

    def run_to_end(self, command, cwd=''):
        import time
        rid = self.p.start({'command': command, 'cwd': cwd})['id']
        out, offset = '', 0
        for _ in range(600):
            r = self.p.poll({'id': rid, 'offset': offset})
            out += r['output']; offset = r['offset']
            if r['done']:
                return r['code'], out
            time.sleep(0.1)
        self.fail('command did not finish')

    def test_sync_sends_only_what_differs_and_keeps_untouched_files(self):
        import base64
        keep = self.p.copy / 'sub' / 'keep.txt'
        before = keep.stat().st_mtime_ns
        files = dict(self.files, **{'README.md': local.blob_sha(b'changed\n'), 'new.txt': local.blob_sha(b'new\n')})
        del files['calc.py']
        self.assertEqual(self.p.sync({'repo': 'demo/app', 'commit': self.commit, 'files': files}), {'need': ['README.md', 'new.txt']})
        contents = {'README.md': base64.b64encode(b'changed\n').decode(), 'new.txt': base64.b64encode(b'new\n').decode()}
        self.assertEqual(self.p.sync({'repo': 'demo/app', 'commit': self.commit, 'files': files, 'contents': contents}), {'ok': True})
        self.assertEqual((self.p.copy / 'README.md').read_text(), 'changed\n')
        self.assertFalse((self.p.copy / 'calc.py').exists())
        self.assertEqual(keep.stat().st_mtime_ns, before)
        with self.assertRaises(ValueError):
            self.p.sync({'repo': 'other/app', 'commit': self.commit, 'files': files})
        with self.assertRaises(ValueError):
            self.p.sync({'repo': 'demo/app', 'commit': self.commit, 'files': files, 'contents': {'new.txt': base64.b64encode(b'tampered').decode()}})

    def test_commands_are_sandboxed_streamed_and_approved_once_an_hour(self):
        code, out = self.run_to_end('cat README.md; echo home=$HOME; touch .git/planted 2>&1 | grep -o "Read-only file system"; '
                                    'test -e /home/anthony || echo no-home; python3 -c "import socket; socket.create_connection((\'1.1.1.1\', 443), 1)" 2>&1 | tail -1; exit 3')
        self.assertEqual(code, 3)
        self.assertIn('hello', out); self.assertIn('home=/home/agent', out); self.assertIn('Read-only file system', out)
        self.assertIn('no-home', out); self.assertIn('Network is unreachable', out)
        self.assertFalse((self.p.copy / '.git' / 'planted').exists())
        code, out = self.run_to_end('pwd', cwd='sub')
        self.assertEqual((code, out.strip()), (0, '/workspace/sub'))
        self.assertEqual(len(self.asked), 1)
        self.assertIn('build and test demo/app', self.asked[0])

    def test_paths_and_symlinks_stay_inside_the_copy(self):
        outside = Path(self.tmp.name) / 'outside'
        outside.mkdir()
        for bad in ('../x', '/etc/passwd', '.git/config', 'a/../../x'):
            with self.assertRaises(ValueError):
                self.p.sync({'repo': 'demo/app', 'commit': self.commit, 'files': {bad: local.blob_sha(b'')}})
        self.run_to_end(f'ln -s {outside} evil')
        with self.assertRaises(OSError):
            self.p.write('evil/x.txt', b'escape')
        self.assertEqual(list(outside.iterdir()), [])

    def test_time_limit_and_decline(self):
        self.p.limit = 1
        code, out = self.run_to_end('sleep 30')
        self.assertNotEqual(code, 0); self.assertIn('[stopped: over', out)
        self.p.granted_until = 0
        self.p.approve = lambda q: False
        with self.assertRaises(ValueError):
            self.p.start({'command': 'true'})

    @unittest.skipUnless(local.find_tools().get('cargo_bin') and any('node' in b for b in local.find_tools()['binds']), 'needs cargo and node outside /usr')
    def test_toolchains_run_offline(self):
        self.p.tools = local.find_tools()
        code, out = self.run_to_end('cargo --version && node --version && npm --version')
        self.assertEqual(code, 0, out)
        self.assertIn('cargo ', out)

    def test_cargo_git_dependencies_are_mounted_read_only(self):
        tools = {'binds': [], 'path': [], 'cargo_bin': '/c/bin', 'registry': '/c/registry', 'cargo_git': '/c/git'}
        argv = local.project_sandbox('/copy', tools, '', 'true')
        for src, dst in (('/c/bin', '/home/agent/cargo/bin'), ('/c/registry', '/home/agent/cargo/registry'), ('/c/git', '/home/agent/cargo/git'), ('/copy/.git', '/workspace/.git')):
            i = argv.index(src)
            self.assertEqual((argv[i - 1], argv[i + 1]), ('--ro-bind', dst))
        self.assertIn('CARGO_NET_OFFLINE', argv)

    def test_extra_origin_for_local_builds(self):
        c = local.Connector(self.p.copy, approve=lambda _: True)
        c.project = self.p
        server = ThreadingHTTPServer(('127.0.0.1', 0), local.handler(c, ('http://localhost:8080', local.ORIGIN)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        def status(origin):
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/rpc', data=b'{"type":"status"}',
                                         headers={'Origin': origin, 'Authorization': 'Bearer ' + c.token, 'Content-Type': 'application/json'})
            return json.load(urllib.request.urlopen(req))
        try:
            self.assertEqual(status('http://localhost:8080')['project'], 'demo/app')
            with self.assertRaises(urllib.error.HTTPError):
                status('http://localhost:9999')
        finally:
            server.shutdown(); server.server_close(); os.close(c.fd)


if __name__ == '__main__':
    unittest.main()
