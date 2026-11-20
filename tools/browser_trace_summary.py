"""Export a structural Firefox reference from decrypted tshark JSON.

Raw PCAPs, key logs and tshark JSON remain private. This exporter uses an allowlist:
it never copies cookie values, query values, bodies, TLS randoms, keys or tickets.
Run: python3 tools/browser_trace_summary.py decoded.json --output reference.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

HOSTS = {"accounts.google.com", "accounts.youtube.com", "www.google.com"}
PATHS = {
    "/ServiceLogin", "/v3/signin/identifier", "/favicon.ico", "/_/bscframe",
    "/generate_204", "/accounts/CheckConnection",
    "/v3/signin/_/AccountsSignInUi/data/batchexecute",
    "/v3/signin/_/AccountsSignInUi/browserinfo",
    "/v3/signin/_/AccountsSignInUi/web-reports",
    "/v3/signin/_/AccountsSignInUi/gen204/",
}
QUERY_KEYS = {
    "hl", "dsh", "flowEntry", "flowName", "_reqid", "bl", "f.sid", "rpcids",
    "rt", "source-path", "context", "pmpo", "timestamp", "v", "rtembps",
    "rttms", "tmambps",
}
HEADER_NAMES = {
    ":method", ":path", ":authority", ":scheme", "user-agent", "accept",
    "accept-language", "accept-encoding", "upgrade-insecure-requests",
    "sec-fetch-dest", "sec-fetch-mode", "sec-fetch-site", "sec-fetch-user",
    "sec-fetch-storage-access", "priority", "te", "referer", "cookie",
    "content-type", "content-length", "origin", "x-same-domain",
    "x-goog-ext-278367001-jspb", "x-goog-ext-391502476-jspb",
}
FORM_KEYS = {"f.req", "at", "bgRequest", "deviceinfo", "azt", "continue", "hl"}
PUBLIC_VALUES = {
    "accept": {"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "*/*"},
    "accept-language": {"en-US,en;q=0.9"},
    "accept-encoding": {"gzip, deflate, br, zstd"},
    "upgrade-insecure-requests": {"1"}, "priority": {"u=0, i", "u=4", "u=0"},
    "te": {"trailers"},
    "sec-fetch-dest": {"document", "iframe", "empty", "image", "script"},
    "sec-fetch-mode": {"navigate", "cors", "no-cors"},
    "sec-fetch-site": {"none", "same-origin", "same-site", "cross-site"},
    "sec-fetch-user": {"?1"},
    "content-type": {"application/x-www-form-urlencoded;charset=UTF-8", "application/x-www-form-urlencoded"},
}


def array(value):
    return value if isinstance(value, list) else [value]


def values(obj, key):
    if isinstance(obj, dict):
        for name, value in obj.items():
            if name == key:
                yield from array(value)
            yield from values(value, key)
    elif isinstance(obj, list):
        for value in obj:
            yield from values(value, key)


def first(obj, key, default=None):
    return next(values(obj, key), default)


def numbers(obj, key):
    return [int(str(v), 0) for v in values(obj, key)]


def extension_summary(handshake):
    result = []
    fields = {
        10: ("groups", "tls.handshake.extensions_supported_group"),
        11: ("point_formats", "tls.handshake.extensions_ec_point_format"),
        13: ("signature_algorithms", "tls.handshake.sig_hash_alg"),
        27: ("certificate_compression_algorithms", "tls.compress_certificate.algorithm"),
        28: ("record_size_limit", "tls.record_size_limit"),
        34: ("delegated_credential_algorithms", "tls.handshake.sig_hash_alg"),
        43: ("versions", "tls.handshake.extensions.supported_version"),
        45: ("psk_modes", "tls.extension.psk_ke_mode"),
    }
    for label, data in handshake.items():
        if not label.startswith("Extension:") or not isinstance(data, dict):
            continue
        kind = int(data["tls.handshake.extension.type"])
        item = {"id": kind, "length": int(data["tls.handshake.extension.len"])}
        if kind in fields:
            name, field = fields[kind]
            item[name] = numbers(data, field)
        elif kind == 16:
            item["alpn"] = [x for x in values(data, "tls.handshake.extensions_alpn_str") if x in {"h2", "http/1.1"}]
        elif kind == 51:
            item["shares"] = [{"group": group, "bytes": size} for group, size in zip(
                numbers(data, "tls.handshake.extensions_key_share_group"),
                numbers(data, "tls.handshake.extensions_key_share_key_exchange_length"))]
        elif kind == 41:
            item["identity_lengths"] = numbers(data, "tls.handshake.extensions.psk.identity.identity_length")
            item["selected_identity"] = numbers(data, "tls.handshake.extensions.psk.identity.selected")
        result.append(item)
    return result


def summarize(packets, browser="unspecified", source="Caller-supplied trace; browser identity and account outcome are not validated by this exporter"):
    connections = {}
    # First pass associates both sides with the actual SNI; only known public hosts survive.
    for packet in packets:
        layers = packet["_source"]["layers"]
        stream = first(layers, "tcp.stream")
        for handshake in values(layers, "tls.handshake"):
            if not isinstance(handshake, dict) or handshake.get("tls.handshake.type") != "1":
                continue
            host = first(handshake, "tls.handshake.extensions_server_name")
            if host in HOSTS:
                connections[stream] = {"host": host, "client_port": first(layers, "tcp.srcport")}
    hellos, settings, startup, requests, negotiated, frame_counts = [], [], {}, {}, set(), Counter()
    for packet in packets:
        layers = packet["_source"]["layers"]
        tcp_stream = first(layers, "tcp.stream")
        connection = connections.get(tcp_stream)
        if not connection:
            continue
        client = first(layers, "tcp.srcport") == connection["client_port"]
        for handshake in values(layers, "tls.handshake"):
            if not isinstance(handshake, dict):
                continue
            kind = handshake.get("tls.handshake.type")
            if kind in {"1", "2"}:
                hellos.append({
                    "connection": int(tcp_stream), "host": connection["host"],
                    "side": "client" if kind == "1" else "server",
                    "handshake_bytes": int(handshake["tls.handshake.length"]),
                    "legacy_version": int(handshake["tls.handshake.version"], 0),
                    "record_version": int(first(layers, "tls.record.version", "0"), 0),
                    "session_id_bytes": int(handshake.get("tls.handshake.session_id_length", 0)),
                    "cipher_suites": numbers(handshake, "tls.handshake.ciphersuite"),
                    "extensions": extension_summary(handshake),
                })
            if not client:
                negotiated.update(x for x in values(handshake, "tls.handshake.extensions_alpn_str") if x in {"h2", "http/1.1"})
        for protocol in array(layers.get("http2", [])):
            if not isinstance(protocol, dict):
                continue
            for frame in array(protocol.get("http2.stream", [])):
                if not isinstance(frame, dict) or "http2.type" not in frame:
                    continue
                kind = int(frame["http2.type"])
                stream_id = int(frame["http2.streamid"])
                frame_counts[("client" if client else "server", kind)] += 1
                if client:
                    events = startup.setdefault(tcp_stream, [])
                    if len(events) < 5:
                        events.append({"type": kind, "stream_id": stream_id, "length": int(frame["http2.length"]), "flags": int(frame["http2.flags"], 0)})
                    if kind == 4 and not int(frame["http2.flags"], 0) & 1:
                        sequence = []
                        for setting in array(frame.get("http2.settings", [])):
                            if not isinstance(setting, dict):
                                continue
                            value = next(v for k, v in setting.items() if k != "http2.settings.id")
                            sequence.append({"id": int(setting["http2.settings.id"]), "value": int(value)})
                        settings.append({"connection": int(tcp_stream), "settings": sequence})
                key = (tcp_stream, stream_id)
                headers = [(h["http2.header.name"], h.get("http2.header.value", "")) for h in array(frame.get("http2.header", [])) if isinstance(h, dict)]
                header_map = dict(headers)
                if client and header_map.get(":method") in {"GET", "POST", "HEAD", "OPTIONS"}:
                    url = urlsplit(header_map.get(":path", ""))
                    requests[key] = {
                        "connection": int(tcp_stream), "stream_id": stream_id,
                        "host": connection["host"], "method": header_map[":method"],
                        "path": url.path if url.path in PATHS else "<other>",
                        "query_keys": sorted({k for k, _ in parse_qsl(url.query) if k in QUERY_KEYS}),
                        "header_order": [k if k in HEADER_NAMES else "<other>" for k, _ in headers],
                        "public_headers": {k: v for k, v in headers if v in PUBLIC_VALUES.get(k, set())},
                        "cookie_fields": sum(k == "cookie" for k, _ in headers),
                        "response_status": None, "_body": bytearray(),
                    }
                request = requests.get(key)
                if request:
                    if client and kind == 0:
                        for data in values(frame, "http2.data.data"):
                            request["_body"].extend(bytes.fromhex(data.replace(":", "")))
                    if not client and ":status" in header_map:
                        request["response_status"] = int(header_map[":status"])
    for request in requests.values():
        body = request.pop("_body").decode("utf-8", errors="replace")
        request["form_fields"] = sorted({k for k, _ in parse_qsl(body) if k in FORM_KEYS})
        request["contains_synthetic_identifier"] = "example.invalid" in body
    initial_windows = set()
    for packet in packets:
        layers = packet["_source"]["layers"]
        stream = first(layers, "tcp.stream")
        if stream in connections and first(layers, "tcp.srcport") == connections[stream]["client_port"]:
            for protocol in array(layers.get("http2", [])):
                for frame in array(protocol.get("http2.stream", [])):
                    if isinstance(frame, dict) and frame.get("http2.type") == "8" and frame.get("http2.streamid") == "0":
                        # First connection-level update per connection, not subsequent replenishments.
                        if not connections[stream].get("window_seen"):
                            initial_windows.add(int(frame["http2.window_update.window_size_increment"]))
                            connections[stream]["window_seen"] = True
    return {
        "schema_version": 1, "browser": browser,
        "source": source,
        "notes": ["Session-specific values are excluded by an allowlist.", "When using the extractor's TLS-only PCAP, TCP/IP metadata is reconstructed; use the original PCAP for transport timing.", "Connection numbers and packet sizes are observations, not requirements for future sessions."],
        "tls": {"hellos": hellos, "negotiated_alpn": sorted(negotiated)},
        "http2": {"connection_preface_observed": any(values(packets, "http2.magic")),
                  "client_settings": settings, "initial_connection_window_increments": sorted(initial_windows),
                  "startup_frames": [{"connection": int(k), "frames": v} for k, v in startup.items()],
                  "frame_counts": [{"side": side, "type": kind, "count": count} for (side, kind), count in sorted(frame_counts.items())]},
        "requests": list(requests.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decoded", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", default="unspecified", help="Verified browser/version label; not inferred from a potentially spoofed UA")
    parser.add_argument("--source", default="Caller-supplied trace; browser identity and account outcome are not validated by this exporter")
    args = parser.parse_args()
    result = summarize(json.loads(args.decoded.read_text()), args.browser, args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exported {len(result['tls']['hellos'])} hellos and {len(result['requests'])} request structures; no session values copied.")
