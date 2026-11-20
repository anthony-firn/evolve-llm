//! Fixed diagnostic identities. These change content negotiation and request framing, not
//! the browser engine.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Identity {
    #[default]
    Legacy,
    FirefoxEsr,
    Firefox,
    Chrome,
    Evolve,
}

impl Identity {
    pub fn parse(profile: &str) -> Result<Self, &'static str> {
        match profile {
            "legacy" => Ok(Self::Legacy),
            "firefox-esr" => Ok(Self::FirefoxEsr),
            "firefox" => Ok(Self::Firefox),
            "chrome" => Ok(Self::Chrome),
            "evolve" => Ok(Self::Evolve),
            _ => Err("Unknown browser identity profile"),
        }
    }

    /// Firefox identities also use Firefox's request fields and HTTP/2 framing.
    pub fn is_firefox(self) -> bool {
        matches!(self, Self::FirefoxEsr | Self::Firefox)
    }

    pub fn user_agent(self) -> &'static str {
        match self {
            Self::Legacy => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 EvolveBrowser/0.1",
            Self::FirefoxEsr => "Mozilla/5.0 (X11; Linux x86_64; rv:153.0) Gecko/20100101 Firefox/153.0",
            Self::Firefox => "Mozilla/5.0 (X11; Linux x86_64; rv:160.0) Gecko/20100101 Firefox/160.0",
            Self::Chrome => "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            Self::Evolve => "Mozilla/5.0 (X11; Linux x86_64) EvolveBrowser/0.1",
        }
    }

    pub fn navigator_json(self) -> String {
        let chromium = matches!(self, Self::Legacy | Self::Chrome);
        serde_json::json!({
            "userAgent": self.user_agent(),
            "appVersion": if matches!(self, Self::FirefoxEsr | Self::Firefox) { "5.0 (X11)" } else { &self.user_agent()[8..] },
            "vendor": if chromium { "Google Inc." } else { "" },
            "chromium": chromium,
        }).to_string()
    }
}
