"""Structural reference and secret-exclusion checks for protocol trace exports."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("trace_summary", ROOT / "tools/browser_trace_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def packet(client, *, tls=None, http2=None):
    layers = {"tcp": {"tcp.stream": "0", "tcp.srcport": "30000" if client else "443"}}
    if tls is not None:
        layers["tls"] = {"tls.record.version": "0x0301", "tls.handshake": tls}
    if http2 is not None:
        layers["http2"] = {"http2.stream": http2}
    return {"_source": {"layers": layers}}


def headers(items):
    return [{"http2.header.name": key, "http2.header.value": value} for key, value in items]


class ReferenceTests(unittest.TestCase):
    def test_real_reference_includes_hybrid_tls_h2_and_the_synthetic_identifier_rpc(self):
        path = ROOT / "browser/tests/fixtures/firefox-esr-153.7-protocol.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["tls"]["negotiated_alpn"], ["h2"])
        google = [h for h in data["tls"]["hellos"] if h["host"] == "accounts.google.com"]
        cold = next(h for h in google if h["side"] == "client")
        shares = next(e["shares"] for e in cold["extensions"] if e["id"] == 51)
        self.assertEqual(shares, [{"group": 4588, "bytes": 1216}, {"group": 29, "bytes": 32}, {"group": 23, "bytes": 65}])
        self.assertTrue(any(e["id"] == 41 and e["selected_identity"] == [0] for h in google if h["side"] == "server" for e in h["extensions"]))
        self.assertTrue(data["http2"]["connection_preface_observed"])
        expected = [{"id": 1, "value": 65536}, {"id": 2, "value": 0}, {"id": 4, "value": 131072}, {"id": 5, "value": 16384}]
        self.assertTrue(data["http2"]["client_settings"])
        for connection in data["http2"]["client_settings"]:
            self.assertEqual(connection["settings"], expected)
        self.assertEqual(data["http2"]["initial_connection_window_increments"], [12517377])
        first = data["requests"][0]
        self.assertEqual(first["header_order"][:4], [":method", ":path", ":authority", ":scheme"])
        submitted = [r for r in data["requests"] if r["contains_synthetic_identifier"]]
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["method"], "POST")
        self.assertEqual(submitted[0]["response_status"], 200)
        self.assertEqual(submitted[0]["form_fields"], ["at", "f.req"])
        self.assertTrue(any(r["path"] == "/_/bscframe" for r in data["requests"]))
        self.assertTrue(any(r["host"] == "accounts.youtube.com" and r["path"] == "/accounts/CheckConnection" for r in data["requests"]))

    def test_export_drops_session_material_even_from_unknown_fields(self):
        secret = "NEVER_PUBLISH_THIS_SESSION"
        hello = {"tls.handshake.type": "1", "tls.handshake.length": "200", "tls.handshake.version": "0x0303",
                 "tls.handshake.random": secret, "tls.handshake.session_id": secret,
                 "tls.handshake.extensions_server_name": "accounts.google.com",
                 "Extension: pre_shared_key": {"tls.handshake.extension.type": "41", "tls.handshake.extension.len": "50",
                    "tls.handshake.extensions.psk.identity.identity": secret, "tls.handshake.extensions.psk.binders": secret},
                 "unknown": secret}
        request = {"http2.type": "1", "http2.streamid": "1", "http2.length": "100", "http2.flags": "0x04",
                   "http2.header": headers([(':method', 'POST'), (':path', '/ServiceLogin?hl=' + secret + '&' + secret + '=x'),
                        ('cookie', secret), ('authorization', secret), ('user-agent', secret), ('x-private-' + secret, secret),
                        ('accept-encoding', 'gzip, deflate, br, zstd')])}
        body = {"http2.type": "0", "http2.streamid": "1", "http2.length": "30", "http2.flags": "0x01",
                "http2.data.data": ("f.req=" + secret + "&" + secret + "=x").encode().hex()}
        response = {"http2.type": "1", "http2.streamid": "1", "http2.length": "20", "http2.flags": "0x05",
                    "http2.header": headers([(':status', '200'), ('set-cookie', secret)])}
        output = summary.summarize([packet(True, tls=hello), packet(True, http2=request), packet(True, http2=body), packet(False, http2=response)])
        encoded = json.dumps(output)
        self.assertNotIn(secret, encoded)
        self.assertNotIn(secret.encode().hex(), encoded)
        request = output["requests"][0]
        self.assertEqual(request["cookie_fields"], 1)
        self.assertEqual(request["query_keys"], ["hl"])
        self.assertEqual(request["form_fields"], ["f.req"])
        self.assertEqual(request["public_headers"], {"accept-encoding": "gzip, deflate, br, zstd"})

    def test_unknown_hosts_are_excluded(self):
        hello = {"tls.handshake.type": "1", "tls.handshake.extensions_server_name": "private.invalid"}
        self.assertEqual(summary.summarize([packet(True, tls=hello)])["tls"]["hellos"], [])


if __name__ == "__main__":
    unittest.main()
