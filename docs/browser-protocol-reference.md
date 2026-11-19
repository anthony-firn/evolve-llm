# Firefox ESR protocol reference

Captured on 2026-11-19 using the official Mozilla Firefox ESR 153.7.0 Linux x86_64
build, whose archive matched Mozilla's SHA512 sum. Firefox ran in a fresh profile with
`navigator.webdriver === false`, using its normal GUI and TLS implementation. Traffic
went through a byte-transparent local TCP forwarder to Tor's SOCKS5 service. The forwarder
did not terminate TLS or modify HTTP. A dedicated loopback-port capture recorded only this
test browser's proxied connections. tcpdump reported **0 packets dropped by the kernel**.

Firefox exported session keys through its supported
[`SSLKEYLOGFILE` diagnostic mechanism](https://support.mozilla.org/en-US/kb/sslkeylogfile-warning).
Wireshark/tshark 4.0.17 decoded TLS and HTTP/2 using those keys. Its SOCKS dissector did not
decode most ClientHellos correctly, so the TLS streams were also extracted into a derivative
PCAP. **TLS payload bytes are preserved; the derivative reconstructs TCP/IP headers.** Use the
original capture for TCP segmentation, retransmission or timing work.
This reference covers TCP/TLS and HTTP/2 over Tor; it does not cover QUIC/HTTP/3.

The sign-in capture used only `evolve-browser-compat@example.invalid`. The native modern
form submitted the identifier and displayed “Couldn't find this account.” No real account,
password or verification code was submitted during this capture. A separate earlier native
test reached the password prompt for an owner-authorized username over both direct and Tor
connections; that was not a completed authenticated sign-in.

## Measured target

The [machine-readable reference](../browser/tests/fixtures/firefox-esr-153.7-protocol.json)
contains 20 client/server Hello structures and 21 request structures for the Google account
hosts. It excludes randoms, session IDs, key material, tickets, cookies, query values and
request/response bodies. Numeric protocol identifiers, lengths, order and a small allowlist
of ordinary public header values remain.

| Area | Native Firefox ESR observation | Current Evolve engine |
| --- | --- | --- |
| TLS negotiation | TLS 1.3, `TLS_AES_128_GCM_SHA256`, group `0x11ec` (`X25519MLKEM768`) | rustls/ring, no hybrid ML-KEM group |
| Offered key shares | X25519MLKEM768: 1216 bytes; X25519: 32; P-256: 65 | ring's supported groups and rustls-generated shares |
| Cipher list | 16 offered suites, exact order in JSON | 9 supported suites plus SCSV; NSS-order experiment reorders the supported subset |
| ALPN | `h2`, `http/1.1`; Google selected `h2` | `h2`, `http/1.1`; HTTP/2 when the server selects it |
| Cold ClientHello | 1893-byte handshake body; 32-byte compatibility session ID | Different rustls structure and extension layout |
| TLS extensions | Ordered SNI, EMS, renegotiation, groups, point formats, ticket, ALPN, status, delegated credentials, SCT, key shares, versions, signatures, PSK modes, record limit, certificate compression, ECH extension | A smaller rustls extension set, with randomized ordering |
| Certificate compression offer | Algorithm IDs `1,2,3` | No corresponding complete NSS profile |
| Record size limit offer | `16385` | Different configuration |
| Warm reconnect | 2460-byte ClientHello body, PSK extension last; Google accepted identity 0 | Resumption behavior has not been compared on the same route |
| HTTP/2 settings, in order | `1=65536`, `2=0`, `4=131072`, `5=16384` | Same with a Firefox identity (other identities add `6`) |
| Connection flow control | Initial WINDOW_UPDATE increment `12517377`, raising 65535 to 12582912 bytes | Same |
| Client streams | First stream 3; HEADERS carry priority (non-exclusive, on 0, weight 42 documents, 32 fonts, 22 scripts/fetch/frames, 12 images), then a stream WINDOW_UPDATE of `12451840` | Same with a Firefox identity; others start at 1 without priority |
| Pseudo-header order | `:method`, `:path`, `:authority`, `:scheme` | Same with a Firefox identity |
| Request headers | Order in JSON; includes `priority: u=0, i`, `te: trailers`; multiple cookie fields as cookies accumulate | Same order and values with a Firefox identity (test `firefox_field_order_matches_reference`), cookies split into one field each over HTTP/2; `sec-fetch-storage-access` not sent |
| Content coding offer | `gzip, deflate, br, zstd` | Same (zstd decoded by `ruzstd`) |

The cold Google ClientHello extension IDs were:

```text
0,23,65281,10,11,35,16,5,34,18,51,43,13,45,28,27,65037
```

The resumed ClientHello omitted legacy ticket extension 35 and appended PSK extension 41.
Both reconnect ServerHellos selected PSK identity 0 and group 4588. Reconnects were triggered
by closing the recorder's idle TCP connections, then navigating again. No early-data extension
was observed. Extension 65037 was present; this report does not claim ECH was successfully
negotiated. Extension presence is distinct from implementing the feature.

HTTP/2 captures include the connection preface, SETTINGS/ACK, WINDOW_UPDATE, HEADERS, DATA,
PING/ACK and GOAWAY. Request-dependent frame sizes and HPACK state are observations, not
constants to copy into future connections. Session randomness and keys must be generated
fresh; captured session material is not an implementation target.

## Google request sequence

The reference includes these actual browser operations:

1. `GET /ServiceLogin`, followed by `GET /v3/signin/identifier`.
2. `GET /_/bscframe` and the cross-origin iframe
   `GET https://accounts.youtube.com/accounts/CheckConnection`.
3. Startup RPC and browser-info requests under `/v3/signin/_/AccountsSignInUi/`.
4. A later `POST /v3/signin/_/AccountsSignInUi/data/batchexecute` containing the synthetic
   identifier. Its form fields were `at` and `f.req`; it received HTTP 200 and the UI rendered
   the account-not-found result.

The JSON marks that last request with `contains_synthetic_identifier: true` without storing
its body. Earlier Evolve traces also contain generic batchexecute requests; their presence
alone does not prove identifier submission. RPC arguments and session tokens must come from
the executing page, not from replaying a reference request.

These observations also identify a separate browser-engine dependency: Evolve currently
lacks executing iframe realms and `iframe.contentWindow`. This capture establishes that
the native flow uses those frames, but it does not by itself prove which missing API causes
the stall. HTTP/2 and a closer TLS profile alone have not been shown to enable authentication.

## Reproduce the offline extraction

Keep original captures, key logs, decoded JSON and browser profiles in a private directory
outside Git. The exporter is deliberately narrow and drops unknown hosts, paths, header
names/values, query keys and form keys. Tests insert secret markers into all those locations
to check that they are not exported.

```sh
python3 -m venv /tmp/browser-trace-tools
/tmp/browser-trace-tools/bin/pip install -r tools/requirements-browser-trace.txt
/tmp/browser-trace-tools/bin/python tools/browser_trace_extract_tls.py \
  /private/firefox-tor.pcap --socks-port 42451 --output-dir /private/extracted

tshark -2 -r /private/extracted/tls-only.pcap \
  -o tls.keylog_file:/private/firefox.keys.log \
  -Y 'tls.handshake or http2' -T json --no-duplicate-keys > /private/decoded.json

python3 -B tools/browser_trace_summary.py /private/decoded.json \
  --browser 'Mozilla Firefox ESR 153.7.0, Linux x86_64' \
  --output /private/reference.json
/tmp/browser-trace-tools/bin/python -B tests/test_browser_trace_extract_tls.py
python3 -B tests/test_browser_trace_summary.py
```

Port 42451 belongs to this particular recording; use the actual loopback SOCKS forwarding
port when making a new capture. The extractor filters to known Google hosts and rejects
flows with missing bytes or conflicting retransmissions. Its manifest records SHA256 hashes
of both original TLS directions. The normalization tests verify byte preservation.

## Implementation gates

1. **Done:** an HTTP/2 transport (h2: ALPN selection, HPACK, continuation, multiplexing, flow
   control, SETTINGS/PING acknowledgments, GOAWAY; h1 fallback). Its emitted settings, windows,
   stream ids, priority and pseudo-header order are checked against this reference in
   `browser/src/net/http2.rs` tests, and request fields in `browser/src/net/headers.rs` tests.
2. Evaluate a TLS backend/provider that actually supports the hybrid key exchange, relevant
   extensions and certificate compression. Cipher reordering alone leaves major differences.
   Preserve certificate and hostname verification and test rejected certificates as before.
3. **Done:** zstd decoding, advertised as Firefox does. Cookie policy behavior is still to compare.
4. Implement independent iframe JS contexts and origin-checked cross-frame messaging, then
   verify that the actual page generates its identifier RPC.
5. Repeat cold/warm captures of both clients on a controlled route. The native Tor exit here
   was not held identical to the WASM engine's Snowflake exit. Treat Google authentication
   as a separate acceptance test; this reference is not evidence of a completed sign-in.
