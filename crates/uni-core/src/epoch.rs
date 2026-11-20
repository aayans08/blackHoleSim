//! Simulation time: TDB seconds since J2000.0 = 2000-01-01T12:00:00 TT, split into days and seconds.

use crate::units::DAY;

/// An instant as whole days plus seconds since J2000.0 (TDB). Invariant: 0 ≤ sec < 86400.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
pub struct Epoch {
    pub day: i64,
    pub sec: f64,
}

/// TT − TAI [s], exact by definition (IAU 1991 Resolution A4, Recommendation IV).
const TT_MINUS_TAI: f64 = 32.184;

/// TAI − UTC [s] assumed before 1972-01-01, where the leap-second table starts. UTC had no leap
/// seconds then (it drifted by fractional steps instead), so we hold the 1972 value of 10 s
/// constant; instants before 1972 can be off by up to ~10 s.
const PRE_1972_TAI_MINUS_UTC: f64 = 10.0;

/// The Unix day (days since 1970-01-01) of J2000.0's calendar date, 2000-01-01.
const J2000_UNIX_DAY: i64 = days_from_civil(2000, 1, 1);

/// J2000.0 falls at 12:00:00 TT, half a day into its calendar date.
const J2000_SEC_OF_DAY: f64 = 43_200.0;

/// IERS leap seconds: (Unix day from whose 00:00:00 UTC the value holds, TAI − UTC [s]).
/// Source: IERS Bulletin C, as tabulated in Leap_Second.dat (IERS Earth Orientation Centre,
/// Paris Observatory). The last entry is 2017-01-01 (37 s); extend the table if IERS announces
/// another leap second.
const LEAP_SECONDS: [(i64, f64); 28] = [
    (days_from_civil(1972, 1, 1), 10.0),
    (days_from_civil(1972, 7, 1), 11.0),
    (days_from_civil(1973, 1, 1), 12.0),
    (days_from_civil(1974, 1, 1), 13.0),
    (days_from_civil(1975, 1, 1), 14.0),
    (days_from_civil(1976, 1, 1), 15.0),
    (days_from_civil(1977, 1, 1), 16.0),
    (days_from_civil(1978, 1, 1), 17.0),
    (days_from_civil(1979, 1, 1), 18.0),
    (days_from_civil(1980, 1, 1), 19.0),
    (days_from_civil(1981, 7, 1), 20.0),
    (days_from_civil(1982, 7, 1), 21.0),
    (days_from_civil(1983, 7, 1), 22.0),
    (days_from_civil(1985, 7, 1), 23.0),
    (days_from_civil(1988, 1, 1), 24.0),
    (days_from_civil(1990, 1, 1), 25.0),
    (days_from_civil(1991, 1, 1), 26.0),
    (days_from_civil(1992, 7, 1), 27.0),
    (days_from_civil(1993, 7, 1), 28.0),
    (days_from_civil(1994, 7, 1), 29.0),
    (days_from_civil(1996, 1, 1), 30.0),
    (days_from_civil(1997, 7, 1), 31.0),
    (days_from_civil(1999, 1, 1), 32.0),
    (days_from_civil(2006, 1, 1), 33.0),
    (days_from_civil(2009, 1, 1), 34.0),
    (days_from_civil(2012, 7, 1), 35.0),
    (days_from_civil(2015, 7, 1), 36.0),
    (days_from_civil(2017, 1, 1), 37.0),
];

impl Epoch {
    /// J2000.0 = JD 2451545.0 TT = 2000-01-01T12:00:00 TT (IERS Conventions 2010).
    pub const J2000: Epoch = Epoch { day: 0, sec: 0.0 };

    /// Builds an epoch from TDB seconds since J2000.0.
    pub fn from_tdb_seconds(s: f64) -> Self {
        Epoch::J2000.add_seconds(s)
    }

    /// Returns this epoch moved by `s` seconds, keeping 0 ≤ sec < 86400.
    pub fn add_seconds(self, s: f64) -> Self {
        // Split off whole days first so a large `s` cannot swamp the sub-second part of `sec`.
        let (days, rem) = split_days(s);
        let (carry, sec) = split_days(self.sec + rem);
        Epoch { day: self.day.saturating_add(days).saturating_add(carry), sec }
    }

    /// Returns `self − other` in seconds.
    pub fn seconds_since(self, other: Epoch) -> f64 {
        self.day.saturating_sub(other.day) as f64 * DAY + (self.sec - other.sec)
    }

    /// Converts Unix UTC seconds using the IERS leap-second table; TDB ≈ TT here.
    ///
    /// TT = UTC + (TAI − UTC) + 32.184 s. TDB differs from TT by under 2 ms (IAU 2006
    /// Resolution B3), which this ignores. Before 1972, TAI − UTC is taken as 10 s.
    pub fn from_unix_utc(unix_s: f64) -> Self {
        let (unix_day, utc_sec) = split_days(unix_s);
        Self::from_utc_day(unix_day, utc_sec)
    }

    /// Inverse of [`Epoch::from_unix_utc`].
    ///
    /// `to_unix_utc(from_unix_utc(u)) == u` up to rounding for every `u`. An instant inside an
    /// inserted leap second has no Unix time of its own; as on POSIX clocks it maps onto the
    /// first second of the next day, which therefore occurs twice.
    pub fn to_unix_utc(self) -> f64 {
        let unix_day = self.day.saturating_add(J2000_UNIX_DAY);
        // Undo `from_utc_day` for a given TAI − UTC, giving (Unix day, UTC seconds of day).
        let utc = |tai_minus_utc: f64| {
            let (carry, sec) = split_days(self.sec - utc_to_epoch_sec(tai_minus_utc));
            (unix_day.saturating_add(carry), sec)
        };
        // Use the newest table entry whose UTC result falls on or after its own start date.
        let (day, sec) = LEAP_SECONDS
            .iter()
            .rev()
            .map(|&(start, tai_minus_utc)| (start, utc(tai_minus_utc)))
            .find(|&(start, (day, _))| day >= start)
            .map_or_else(|| utc(PRE_1972_TAI_MINUS_UTC), |(_, day_sec)| day_sec);
        day as f64 * DAY + sec
    }

    /// Parses `"YYYY-MM-DDThh:mm:ss[.fff]Z"` (UTC).
    ///
    /// Any number of fraction digits is accepted. `ss = 60` is accepted only for an actual leap
    /// second (23:59:60 on a day the IERS table ends with one).
    pub fn parse_iso_utc(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        let separators_ok = b.len() >= 20
            && b[4] == b'-'
            && b[7] == b'-'
            && b[10] == b'T'
            && b[13] == b':'
            && b[16] == b':'
            && b[b.len() - 1] == b'Z';
        if !separators_ok {
            return None;
        }
        let number = |at: usize, len: usize| -> Option<i64> {
            b[at..at + len].iter().try_fold(0_i64, |acc, &c| {
                c.is_ascii_digit().then(|| acc * 10 + i64::from(c - b'0'))
            })
        };
        let (year, month, day) = (number(0, 4)?, number(5, 2)?, number(8, 2)?);
        let (hour, minute, second) = (number(11, 2)?, number(14, 2)?, number(17, 2)?);
        let fraction = &b[19..b.len() - 1];
        let fraction = match fraction {
            [] => 0.0,
            [b'.', digits @ ..] if !digits.is_empty() && digits.iter().all(u8::is_ascii_digit) => {
                s.get(19..s.len() - 1)?.parse::<f64>().ok()?
            }
            _ => return None,
        };
        if !(1..=12).contains(&month)
            || !(1..=days_in_month(year, month)).contains(&day)
            || hour > 23
            || minute > 59
        {
            return None;
        }
        let unix_day = days_from_civil(year, month, day);
        let ends_with_leap_second = hour == 23
            && minute == 59
            && tai_minus_utc(unix_day + 1) > tai_minus_utc(unix_day);
        if second > 60 || (second == 60 && !ends_with_leap_second) {
            return None;
        }
        let utc_sec = (hour * 3600 + minute * 60 + second) as f64 + fraction;
        Some(Self::from_utc_day(unix_day, utc_sec))
    }

    /// Julian centuries of 36525 days since J2000.0.
    pub fn julian_centuries(self) -> f64 {
        // Julian century = 36525 days (IAU 1976 System of Astronomical Constants).
        (self.day as f64 + self.sec / DAY) / 36_525.0
    }

    /// The epoch `utc_sec` seconds after 00:00:00 UTC on `unix_day`. `utc_sec` reaches past
    /// 86400 only inside a leap second, which then lands on the right TT instant.
    fn from_utc_day(unix_day: i64, utc_sec: f64) -> Self {
        let midnight = Epoch { day: unix_day.saturating_sub(J2000_UNIX_DAY), sec: 0.0 };
        midnight.add_seconds(utc_sec + utc_to_epoch_sec(tai_minus_utc(unix_day)))
    }
}

/// What to add to UTC seconds-of-day to get seconds after the J2000-aligned day boundary
/// (12:00:00 TT) of the same Unix day: (TAI − UTC) + (TT − TAI) − 12 h.
fn utc_to_epoch_sec(tai_minus_utc: f64) -> f64 {
    tai_minus_utc + TT_MINUS_TAI - J2000_SEC_OF_DAY
}

/// TAI − UTC [s] in force during the UTC day `unix_day`.
fn tai_minus_utc(unix_day: i64) -> f64 {
    LEAP_SECONDS
        .iter()
        .rev()
        .find(|&&(start, _)| unix_day >= start)
        .map_or(PRE_1972_TAI_MINUS_UTC, |&(_, tai_minus_utc)| tai_minus_utc)
}

/// Splits `s` seconds into whole days and a remainder in [0, 86400).
fn split_days(s: f64) -> (i64, f64) {
    let whole = (s / DAY).floor();
    let (mut days, mut rem) = (whole as i64, s - whole * DAY);
    // Rounding in `s / DAY` can leave `rem` a hair outside [0, DAY).
    if rem < 0.0 {
        rem += DAY;
        days = days.saturating_sub(1);
    }
    if rem >= DAY {
        rem -= DAY;
        days = days.saturating_add(1);
    }
    (days, rem)
}

/// Days from 1970-01-01 to the proleptic Gregorian date `year-month-day`
/// (Hinnant 2013, "chrono-Compatible Low-Level Date Algorithms", `days_from_civil`).
const fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let year_of_era = y.rem_euclid(400);
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1; // March 1 = 0
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Days in `month` (1–12) of the proleptic Gregorian `year`.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j2000_from_unix() {
        let e = Epoch::from_unix_utc(946_727_935.816);
        assert!(e.seconds_since(Epoch::J2000).abs() < 1e-3);
    }

    #[test]
    fn parse_2026_11_20() {
        let e = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap();
        assert_eq!(e.day, 9819);
        assert!((e.sec - 43_269.184).abs() < 1e-3);
    }

    #[test]
    fn epoch_add_crosses_days() {
        let e = Epoch::J2000.add_seconds(-1.0);
        assert_eq!(e.day, -1);
        assert!((e.sec - 86_399.0).abs() < 1e-9);
    }

    #[test]
    fn add_seconds_keeps_sec_in_range() {
        for s in [-1e-300, 1e-300, 86_400.0, -86_400.0, 3.0 * 86_400.0 + 5.0, 1e12 + 0.25] {
            let e = Epoch::from_tdb_seconds(s);
            assert!((0.0..86_400.0).contains(&e.sec), "{s} -> {e:?}");
            let tol = (s.abs() * f64::EPSILON).max(1e-12);
            assert!((e.seconds_since(Epoch::J2000) - s).abs() <= tol, "{s} -> {e:?}");
        }
        assert_eq!(Epoch::from_tdb_seconds(3.0 * 86_400.0 + 5.0), Epoch { day: 3, sec: 5.0 });
    }

    #[test]
    fn unix_roundtrip_including_pre_1972() {
        // 1966-10-31, 1970-01-01, 1971-12-31T23:59:59.5, either side of the 2016 leap second,
        // and 2026-11-20.
        for u in [-1e8, 0.0, 63_071_999.5, 1_483_228_799.5, 1_483_228_800.5, 1_795_132_800.0] {
            let back = Epoch::from_unix_utc(u).to_unix_utc();
            assert!((back - u).abs() < 1e-6, "{u} -> {back}");
        }
        let e = Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap();
        assert_eq!(e.to_unix_utc(), 1_795_132_800.0);
    }

    #[test]
    fn tai_minus_utc_is_10_s_before_1972() {
        // At the Unix epoch TT − UTC = 10 s + 32.184 s; J2000.0 is Unix 946 728 000 in TT.
        let e = Epoch::from_unix_utc(0.0);
        assert!((e.seconds_since(Epoch::J2000) - (42.184 - 946_728_000.0)).abs() < 1e-6);
    }

    #[test]
    fn leap_second_2016_adds_one_second() {
        let before = Epoch::from_unix_utc(1_483_228_799.0); // 2016-12-31T23:59:59Z
        let after = Epoch::from_unix_utc(1_483_228_800.0); // 2017-01-01T00:00:00Z
        assert!((after.seconds_since(before) - 2.0).abs() < 1e-6);
        let leap = Epoch::parse_iso_utc("2016-12-31T23:59:60Z").unwrap();
        assert!((leap.seconds_since(before) - 1.0).abs() < 1e-6);
        assert!((after.seconds_since(leap) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_fraction_and_reject_malformed() {
        let e = Epoch::parse_iso_utc("2000-01-01T11:58:55.816Z").unwrap();
        assert!(e.seconds_since(Epoch::J2000).abs() < 1e-6);
        assert!(Epoch::parse_iso_utc("2024-02-29T23:59:59.999999Z").is_some());
        for bad in [
            "",
            "2026-11-20T00:00:00",
            "2026-11-20 00:00:00Z",
            "2026-11-20T00:00:00Zx",
            "2026-11-20T00:00:00.Z",
            "2026-11-20T00:00:00.1e3Z",
            "+026-11-20T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-02-29T00:00:00Z",
            "2026-11-00T00:00:00Z",
            "2026-11-20T24:00:00Z",
            "2026-11-20T00:60:00Z",
            "2016-12-30T23:59:60Z",
        ] {
            assert_eq!(Epoch::parse_iso_utc(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn julian_centuries_count_36525_days() {
        assert_eq!(Epoch::J2000.julian_centuries(), 0.0);
        assert_eq!(Epoch { day: 36_525, sec: 0.0 }.julian_centuries(), 1.0);
        let c = Epoch { day: -36_525, sec: 43_200.0 }.julian_centuries();
        assert!((c + 36_524.5 / 36_525.0).abs() < 1e-15);
    }
}
