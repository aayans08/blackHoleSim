//! Simulation time: TDB seconds since J2000.0 = 2000-01-01T12:00:00 TT, split into days and seconds.

/// An instant as whole days plus seconds since J2000.0 (TDB). Invariant: 0 ≤ sec < 86400.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Epoch {
    pub day: i64,
    pub sec: f64,
}

impl Epoch {
    /// J2000.0 = JD 2451545.0 TT = 2000-01-01T12:00:00 TT (IERS Conventions 2010).
    pub const J2000: Epoch = Epoch { day: 0, sec: 0.0 };

    /// Builds an epoch from TDB seconds since J2000.0.
    pub fn from_tdb_seconds(s: f64) -> Self {
        todo!()
    }

    /// Returns this epoch moved by `s` seconds, keeping 0 ≤ sec < 86400.
    pub fn add_seconds(self, s: f64) -> Self {
        todo!()
    }

    /// Returns `self − other` in seconds.
    pub fn seconds_since(self, other: Epoch) -> f64 {
        todo!()
    }

    /// Converts Unix UTC seconds using the IERS leap-second table; TDB ≈ TT here.
    pub fn from_unix_utc(unix_s: f64) -> Self {
        todo!()
    }

    /// Inverse of [`Epoch::from_unix_utc`].
    pub fn to_unix_utc(self) -> f64 {
        todo!()
    }

    /// Parses `"YYYY-MM-DDThh:mm:ss[.fff]Z"` (UTC).
    pub fn parse_iso_utc(s: &str) -> Option<Self> {
        todo!()
    }

    /// Julian centuries of 36525 days since J2000.0.
    pub fn julian_centuries(self) -> f64 {
        todo!()
    }
}
