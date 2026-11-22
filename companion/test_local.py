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


if __name__ == '__main__':
    unittest.main()
