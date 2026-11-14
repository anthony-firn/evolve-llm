//! Wisp v1 packets: `[type: u8][stream id: u32 LE][payload]`.
//! Kept identical to the engine's client in `browser/src/net/wisp.rs`.

pub const CONNECT: u8 = 0x01;
pub const DATA: u8 = 0x02;
pub const CONTINUE: u8 = 0x03;
pub const CLOSE: u8 = 0x04;
pub const STREAM_TCP: u8 = 0x01;

/// Close reasons, numbered as in the Wisp v1 specification.
pub mod reason {
    pub const UNSPECIFIED: u8 = 0x01;
    pub const VOLUNTARY: u8 = 0x02;
    pub const NETWORK_ERROR: u8 = 0x03;
    /// Invalid stream information, including unsupported stream types
    pub const INVALID_STREAM: u8 = 0x41;
    /// The destination host is unreachable, cannot be resolved, or is not a valid host name
    pub const UNREACHABLE_HOST: u8 = 0x42;
    /// The destination did not accept the connection in time
    pub const CONNECT_TIMEOUT: u8 = 0x43;
    pub const REFUSED: u8 = 0x44;
    /// TCP data transfer timed out
    pub const DATA_TIMEOUT: u8 = 0x47;
    /// The destination address or port is blocked by the relay's policy
    pub const BLOCKED: u8 = 0x48;
    /// Too many streams
    pub const THROTTLED: u8 = 0x49;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Packet {
    Connect { id: u32, kind: u8, port: u16, host: String },
    Data { id: u32, payload: Vec<u8> },
    Continue { id: u32, buffer: u32 },
    Close { id: u32, reason: u8 },
}

pub fn encode(packet: &Packet) -> Vec<u8> {
    let mut out = Vec::new();
    match packet {
        Packet::Connect { id, kind, port, host } => {
            out.push(CONNECT);
            out.extend_from_slice(&id.to_le_bytes());
            out.push(*kind);
            out.extend_from_slice(&port.to_le_bytes());
            out.extend_from_slice(host.as_bytes());
        }
        Packet::Data { id, payload } => {
            out.reserve(5 + payload.len());
            out.push(DATA);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(payload);
        }
        Packet::Continue { id, buffer } => {
            out.push(CONTINUE);
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&buffer.to_le_bytes());
        }
        Packet::Close { id, reason } => {
            out.push(CLOSE);
            out.extend_from_slice(&id.to_le_bytes());
            out.push(*reason);
        }
    }
    out
}

pub fn decode(buf: &[u8]) -> Result<Packet, String> {
    if buf.len() < 5 {
        return Err("Wisp packet is too short".into());
    }
    let id = u32::from_le_bytes(buf[1..5].try_into().unwrap());
    let body = &buf[5..];
    match buf[0] {
        CONNECT if body.len() >= 3 => Ok(Packet::Connect {
            id,
            kind: body[0],
            port: u16::from_le_bytes([body[1], body[2]]),
            host: String::from_utf8(body[3..].to_vec()).map_err(|_| "Invalid host")?,
        }),
        DATA => Ok(Packet::Data { id, payload: body.to_vec() }),
        CONTINUE if body.len() >= 4 => Ok(Packet::Continue { id, buffer: u32::from_le_bytes(body[..4].try_into().unwrap()) }),
        CLOSE if !body.is_empty() => Ok(Packet::Close { id, reason: body[0] }),
        kind => Err(format!("Unknown or truncated Wisp packet type {kind:#x}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_packet_type() {
        let packets = [
            Packet::Connect { id: 7, kind: STREAM_TCP, port: 443, host: "example.com".into() },
            Packet::Data { id: 7, payload: b"hello".to_vec() },
            Packet::Data { id: 9, payload: vec![] },
            Packet::Continue { id: 0, buffer: 128 },
            Packet::Close { id: 7, reason: reason::VOLUNTARY },
        ];
        for p in packets {
            assert_eq!(decode(&encode(&p)).unwrap(), p);
        }
    }

    #[test]
    fn matches_the_client_byte_layout() {
        // Same vector as browser/src/net/wisp.rs
        assert_eq!(encode(&Packet::Connect { id: 1, kind: 1, port: 443, host: "a".into() }), [1, 1, 0, 0, 0, 1, 0xbb, 0x01, b'a']);
        assert_eq!(encode(&Packet::Continue { id: 0, buffer: 128 }), [3, 0, 0, 0, 0, 128, 0, 0, 0]);
        assert_eq!(encode(&Packet::Close { id: 2, reason: 0x48 }), [4, 2, 0, 0, 0, 0x48]);
    }

    #[test]
    fn uses_the_wisp_v1_close_reason_numbers() {
        use reason::*;
        let codes = [UNSPECIFIED, VOLUNTARY, NETWORK_ERROR, INVALID_STREAM, UNREACHABLE_HOST, CONNECT_TIMEOUT, REFUSED, DATA_TIMEOUT, BLOCKED, THROTTLED];
        assert_eq!(codes, [0x01, 0x02, 0x03, 0x41, 0x42, 0x43, 0x44, 0x47, 0x48, 0x49]);
    }

    #[test]
    fn rejects_malformed_packets() {
        assert!(decode(&[9, 0, 0, 0, 0]).is_err());
        assert!(decode(&[1, 0]).is_err());
        assert!(decode(&[1, 1, 0, 0, 0, 1, 0]).is_err());
        assert!(decode(&[3, 1, 0, 0, 0, 1]).is_err());
        assert!(decode(&[4, 1, 0, 0, 0]).is_err());
        assert!(decode(&[1, 1, 0, 0, 0, 1, 80, 0, 0xff, 0xfe]).is_err());
    }
}
