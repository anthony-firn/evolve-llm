//! Extensions specific to the HTTP/2 protocol.

use crate::hpack::BytesStr;

use bytes::Bytes;
use std::fmt;

/// Represents the `:protocol` pseudo-header used by
/// the [Extended CONNECT Protocol].
///
/// [Extended CONNECT Protocol]: https://datatracker.ietf.org/doc/html/rfc8441#section-4
#[derive(Clone, Eq, PartialEq)]
pub struct Protocol {
    value: BytesStr,
}

impl Protocol {
    /// Converts a static string to a protocol name.
    pub const fn from_static(value: &'static str) -> Self {
        Self {
            value: BytesStr::from_static(value),
        }
    }

    /// Returns a str representation of the header.
    pub fn as_str(&self) -> &str {
        self.value.as_str()
    }

    pub(crate) fn try_from(bytes: Bytes) -> Result<Self, std::str::Utf8Error> {
        Ok(Self {
            value: BytesStr::try_from(bytes)?,
        })
    }
}

impl<'a> From<&'a str> for Protocol {
    fn from(value: &'a str) -> Self {
        Self {
            value: BytesStr::from(value),
        }
    }
}

impl AsRef<[u8]> for Protocol {
    fn as_ref(&self) -> &[u8] {
        self.value.as_ref()
    }
}

impl fmt::Debug for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.value.fmt(f)
    }
}

/// Client request framing that browsers vary: pseudo-header order, the HEADERS
/// priority block and the stream's initial receive window. Insert it into the
/// `http::Request` extensions before `SendRequest::send_request`. Every choice
/// here is valid HTTP/2; the default keeps h2's usual framing.
/// (Evolve browser addition; see browser/vendor/README.md.)
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequestFraming {
    /// Encode `:method, :path, :authority, :scheme` instead of
    /// `:method, :scheme, :authority, :path`.
    pub path_before_authority: bool,
    /// Send a priority block in HEADERS: (exclusive, dependency stream id, weight 1-256).
    pub priority: Option<(bool, u32, u16)>,
    /// Send a WINDOW_UPDATE with this increment for the stream directly after its
    /// HEADERS. The peer applies it on top of the advertised initial window, also
    /// when this side's SETTINGS are not yet acknowledged.
    pub stream_window_increment: Option<u32>,
}
