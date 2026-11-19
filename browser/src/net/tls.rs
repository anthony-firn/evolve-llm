//! Measurable TLS experiments using only capabilities implemented by rustls/ring.
//! Cipher ordering is not full Firefox/NSS emulation: extension layout, key shares,
//! resumption and ALPN remain rustls/HTTP1 behavior.

use std::sync::Arc;

pub fn config(profile: &str) -> Result<Arc<rustls::ClientConfig>, &'static str> {
    let mut provider = rustls::crypto::ring::default_provider();
    match profile {
        "default" => {},
        "firefox-order" => {
            use rustls::crypto::ring::cipher_suite::*;
            // NSS preference order, intersected with ring's supported AEAD suites.
            // https://github.com/nss-dev/nss/blob/master/lib/ssl/ssl3con.c
            provider.cipher_suites = vec![
                TLS13_AES_128_GCM_SHA256, TLS13_CHACHA20_POLY1305_SHA256,
                TLS13_AES_256_GCM_SHA384,
                TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
                TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
                TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
                TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
            ];
        },
        _ => return Err("Unknown TLS profile"),
    }
    let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
    let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions().expect("TLS versions")
        .with_root_certificates(roots).with_no_client_auth();
    // Advertising h2 before implementing it would break connections that select it.
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_profile_rejects_an_untrusted_server_certificate() {
        use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
        let cert = CertificateDer::from(include_bytes!("../../tests/fixtures/tls/self-signed.der").to_vec());
        let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(include_bytes!("../../tests/fixtures/tls/test-only-key.der").to_vec()));
        let server = rustls::ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions().unwrap().with_no_client_auth()
            .with_single_cert(vec![cert], key).unwrap();
        for profile in ["default", "firefox-order"] {
            let mut client = rustls::ClientConnection::new(config(profile).unwrap(), ServerName::try_from("a.test").unwrap()).unwrap();
            let mut server = rustls::ServerConnection::new(Arc::new(server.clone())).unwrap();
            let mut rejected = false;
            for _ in 0..8 {
                let mut bytes = Vec::new();
                client.write_tls(&mut bytes).unwrap();
                server.read_tls(&mut &bytes[..]).unwrap();
                server.process_new_packets().unwrap();
                bytes.clear();
                server.write_tls(&mut bytes).unwrap();
                client.read_tls(&mut &bytes[..]).unwrap();
                match client.process_new_packets() {
                    Err(rustls::Error::InvalidCertificate(_)) => { rejected = true; break; },
                    Err(other) => panic!("Unexpected TLS error: {other}"),
                    Ok(_) => {},
                }
            }
            assert!(rejected, "{profile} accepted an untrusted certificate");
        }
    }

    #[test]
    fn firefox_order_changes_the_actual_client_hello_without_adding_unsupported_suites() {
        let default = config("default").unwrap();
        let firefox = config("firefox-order").unwrap();
        let suites = |c: &rustls::ClientConfig| c.crypto_provider().cipher_suites.iter()
            .map(|s| u16::from(s.suite())).collect::<Vec<_>>();
        let expected = vec![0x1301, 0x1303, 0x1302, 0xc02b, 0xc02f, 0xcca9, 0xcca8, 0xc02c, 0xc030];
        assert_eq!(suites(&firefox), expected);
        let mut a = suites(&default); a.sort();
        let mut b = suites(&firefox); b.sort();
        assert_eq!(a, b);
        assert_eq!(firefox.alpn_protocols, vec![b"http/1.1".to_vec()]);
        let name = rustls::pki_types::ServerName::try_from("example.com").unwrap();
        let mut conn = rustls::ClientConnection::new(firefox, name).unwrap();
        let mut wire = Vec::new();
        conn.write_tls(&mut wire).unwrap();
        assert_eq!(wire[0], 22); // handshake record
        assert_eq!(wire[5], 1); // ClientHello
        let session_end = 44 + wire[43] as usize;
        let len = u16::from_be_bytes([wire[session_end], wire[session_end + 1]]) as usize;
        let advertised = wire[session_end + 2..session_end + 2 + len].chunks_exact(2)
            .map(|p| u16::from_be_bytes([p[0], p[1]])).filter(|s| *s != 0xff).collect::<Vec<_>>();
        assert_eq!(advertised, expected);
        assert!(config("made-up").is_err());
    }
}
