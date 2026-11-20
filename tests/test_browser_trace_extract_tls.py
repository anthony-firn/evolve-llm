"""Verify that SOCKS removal preserves TLS bytes, including TCP retransmissions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest

try:
    import dpkt
except ImportError:
    dpkt = None

ROOT = Path(__file__).resolve().parent.parent
if dpkt:
    spec = importlib.util.spec_from_file_location("extract_tls", ROOT / "tools/browser_trace_extract_tls.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)


@unittest.skipIf(dpkt is None, "Install tools/requirements-browser-trace.txt")
class ExtractTests(unittest.TestCase):
    def capture(self, path, conflicting=False):
        host = b"accounts.google.com"
        request = b"\x05\x01\x00\x03" + bytes([len(host)]) + host + b"\x01\xbb"
        client_tls = b"\x16\x03\x01\x00\x03abc"
        server_tls = b"\x16\x03\x03\x00\x03xyz"
        client = b"\x05\x01\x00" + request + client_tls
        server = b"\x05\x00\x05\x00\x00\x01\0\0\0\0\0\0" + server_tls

        def packet(direction, seq, ack, flags, data=b""):
            src, dst = (12345, 42451) if direction == 0 else (42451, 12345)
            tcp = dpkt.tcp.TCP(sport=src, dport=dst, seq=seq, ack=ack, flags=flags, data=data)
            tcp.off = 5
            ip = dpkt.ip.IP(src=socket.inet_aton("127.0.0.1"), dst=socket.inet_aton("127.0.0.1"), p=6, data=tcp)
            ip.len = len(ip)
            return bytes(dpkt.ethernet.Ethernet(type=0x0800, src=b"\0" * 6, dst=b"\1" * 6, data=ip))

        packets = [packet(0, 1000, 0, dpkt.tcp.TH_SYN), packet(1, 9000, 1001, dpkt.tcp.TH_SYN | dpkt.tcp.TH_ACK),
                   packet(0, 1001, 9001, dpkt.tcp.TH_ACK, client[:7]),
                   packet(1, 9001, 1008, dpkt.tcp.TH_ACK, server[:12]),
                   packet(0, 1008, 9013, dpkt.tcp.TH_ACK, client[7:]),
                   packet(1, 9013, 1001 + len(client), dpkt.tcp.TH_ACK, server[12:]),
                   packet(0, 1008, 9013, dpkt.tcp.TH_ACK, (b"X" if conflicting else client[7:8]) + client[8:])]
        with path.open("wb") as output:
            writer = dpkt.pcap.Writer(output)
            for n, raw in enumerate(packets):
                writer.writepkt(raw, 1000 + n / 100)
        return [client_tls, server_tls]

    def test_payload_hashes_match_original_streams_after_socks_removal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            expected = self.capture(root / "input.pcap")
            manifest = tool.extract(root / "input.pcap", 42451, root / "output")
            self.assertEqual(len(manifest), 1)
            actual = [dict(), dict()]
            with (root / "output/tls-only.pcap").open("rb") as f:
                for _, raw in dpkt.pcap.Reader(f):
                    tcp = dpkt.ethernet.Ethernet(raw).data.data
                    if tcp.data:
                        actual[0 if tcp.dport == 443 else 1][tcp.seq] = bytes(tcp.data)
            output = [b"".join(parts[k] for k in sorted(parts)) for parts in actual]
            self.assertEqual(output, expected)
            self.assertEqual(manifest[0]["tls_sha256"], [hashlib.sha256(v).hexdigest() for v in output])

    def test_ambiguous_retransmissions_do_not_produce_a_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.capture(root / "input.pcap", conflicting=True)
            self.assertEqual(tool.extract(root / "input.pcap", 42451, root / "output"), [])

    def test_capture_is_filtered_to_the_requested_hosts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.capture(root / "input.pcap")
            self.assertEqual(tool.extract(root / "input.pcap", 42451, root / "output", ["other.invalid"]), [])


if __name__ == "__main__":
    unittest.main()
