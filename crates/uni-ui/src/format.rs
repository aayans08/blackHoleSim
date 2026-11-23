//! Human-readable distances, speeds, masses, durations and dates for the HUD (M0 Task 10; M1
//! Task C4: the mockup's number style, ruling 16).
//!
//! Numbers follow the approved HUD mockup: thousands grouped with a space ("14 200 km",
//! "3 389.5 km"), large and small values as "6.417 × 10²³" with superscript exponents, durations
//! as "24 h 37 m 23 s" or "12 min 40 s", negative numbers with a true minus sign (U+2212).

use uni_core::{AU, C, DAY, Epoch, LY, PC};

/// Formats a distance in metres, picking m, km, AU, pc (with ly), kpc or Mpc by size.
///
/// Each unit is used below its upper threshold (exclusive): 1 km, 0.01 AU, 0.1 pc, 1 kpc, 1 Mpc.
pub fn format_distance(m: f64) -> String {
    if !m.is_finite() {
        "n/a".into()
    } else if m < 1e3 {
        format!("{m:.1} m")
    } else if m < 0.01 * AU {
        format!("{} km", format_grouped(m / 1e3, 0))
    } else if m < 0.1 * PC {
        format!("{} AU", format_grouped(m / AU, 3))
    } else if m < 1e3 * PC {
        format!("{:.2} pc ({:.2} ly)", m / PC, m / LY)
    } else if m < 1e6 * PC {
        format!("{:.2} kpc", m / (1e3 * PC))
    } else {
        format!("{} Mpc", format_grouped(m / (1e6 * PC), 2))
    }
}

/// Formats a speed in m/s as m/s, km/s or a fraction of c.
///
/// Exactly c is still shown as a fraction of c; anything faster is labelled non-physical.
pub fn format_speed(mps: f64) -> String {
    if mps > C {
        format!("{} · FTL (non-physical)", format_speed_value(mps))
    } else {
        format_speed_value(mps)
    }
}

/// [`format_speed`] without the FTL label, for readouts whose header or sub-line says it.
pub(crate) fn format_speed_value(mps: f64) -> String {
    if !mps.is_finite() {
        "n/a".into()
    } else if mps.abs() < 1e3 {
        format!("{} m/s", format_grouped(mps, 1))
    } else if mps.abs() < 0.01 * C {
        format!("{} km/s", format_grouped(mps / 1e3, 2))
    } else if mps.abs() <= C {
        format!("{} c", format_grouped(mps / C, 3))
    } else {
        format!("{} c", format_sci(mps / C, 3))
    }
}

/// A speed as a fraction of c, the Speed panel's sub-line ("4.14 × 10⁻⁵ c"): three significant
/// figures, in scientific notation below 0.01 c and from 1 000 c.
pub fn format_c_fraction(mps: f64) -> String {
    format!("{} c", format_sig(mps / C, 3))
}

/// Formats an epoch as a UTC calendar date and time of day, e.g. `"2026-11-20 00:00:00 UTC"`.
///
/// Seconds are truncated, as on a clock, after rounding to the millisecond so floating-point
/// noise just below a whole second does not show the previous second.
pub fn format_epoch(e: Epoch) -> String {
    let unix_s = ((e.to_unix_utc() * 1e3).round() / 1e3).floor() as i64;
    let day_len = DAY as i64;
    let (year, month, day) = civil_from_days(unix_s.div_euclid(day_len));
    let sec_of_day = unix_s.rem_euclid(day_len);
    let (hour, minute, second) = (sec_of_day / 3600, sec_of_day / 60 % 60, sec_of_day % 60);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

/// `x` with `decimals` decimals and its integer part grouped in threes by spaces.
pub fn format_grouped(x: f64, decimals: usize) -> String {
    if !x.is_finite() {
        return "n/a".into();
    }
    let text = format!("{:.*}", decimals, x.abs());
    let (int, frac) = text
        .split_once('.')
        .map_or((text.as_str(), None), |(i, f)| (i, Some(f)));
    let mut out = String::with_capacity(text.len() + text.len() / 3 + 3);
    // "−0" and "−0.00" read as 0: the sign only where the printed value is not zero.
    if x < 0.0 && text.bytes().any(|b| b.is_ascii_digit() && b != b'0') {
        out.push(MINUS);
    }
    for (k, digit) in int.chars().enumerate() {
        if k > 0 && (int.len() - k) % 3 == 0 {
            out.push(' ');
        }
        out.push(digit);
    }
    if let Some(frac) = frac {
        out.push('.');
        out.push_str(frac);
    }
    out
}

/// The minus sign (U+2212), as in the mockup's "Wheel / = − speed".
const MINUS: char = '\u{2212}';

/// `n` in superscript digits ("²³", "⁻⁵").
pub fn superscript(n: i32) -> String {
    const DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let mut out = String::new();
    if n < 0 {
        out.push('⁻');
    }
    for b in n.unsigned_abs().to_string().bytes() {
        out.push(DIGITS[usize::from(b - b'0')]);
    }
    out
}

/// `s` with every ASCII digit as a subscript ("CO2" → "CO₂").
pub fn subscript_digits(s: &str) -> String {
    const DIGITS: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    s.chars()
        .map(|c| c.to_digit(10).map_or(c, |d| DIGITS[d as usize]))
        .collect()
}

/// `x` to `sig` significant figures in scientific notation.
pub fn format_sci(x: f64, sig: usize) -> String {
    if !x.is_finite() {
        return "n/a".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    // Rust's `{:e}` rounds the mantissa and carries into the exponent (9.9996e2 → 1.00e3).
    let text = format!("{:.*e}", sig.saturating_sub(1), x.abs());
    let (mantissa, exp) = text.split_once('e').unwrap_or((text.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let sign = if x < 0.0 {
        MINUS.to_string()
    } else {
        String::new()
    };
    format!("{sign}{mantissa} × 10{}", superscript(exp))
}

/// `x` to `sig` significant figures: plain (grouped) for 10⁻³ ≤ |x| < 10⁶, otherwise in
/// scientific notation. Integer digits beyond `sig` are kept ("9 200" at three figures).
pub fn format_sig(x: f64, sig: usize) -> String {
    if !x.is_finite() {
        return "n/a".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    let size = x.abs();
    if !(1e-3..1e6).contains(&size) {
        return format_sci(x, sig);
    }
    let magnitude = size.log10().floor() as i32;
    let decimals = (sig as i32 - 1 - magnitude).max(0) as usize;
    format_grouped(x, decimals)
}

/// A mass [kg] to four significant figures.
pub fn format_mass_kg(kg: f64) -> String {
    format!("{} kg", format_sci(kg, 4))
}

/// A duration [s] as hours, minutes and seconds.
pub fn format_hms(s: f64) -> String {
    if !s.is_finite() {
        return "n/a".into();
    }
    let total = s.abs().round() as u64;
    let (h, m, sec) = (total / 3600, total / 60 % 60, total % 60);
    let sign = if s < 0.0 && total > 0 { "\u{2212}" } else { "" };
    if h > 0 {
        format!("{sign}{h} h {m} m {sec} s")
    } else if m > 0 {
        format!("{sign}{m} m {sec} s")
    } else {
        format!("{sign}{sec} s")
    }
}

/// A duration [s], e.g. a light time.
pub fn format_duration(s: f64) -> String {
    /// A Julian year [s] (IAU: 365.25 d).
    const YEAR_S: f64 = 365.25 * DAY;
    if !s.is_finite() || s < 0.0 {
        "n/a".into()
    } else if s < 1.0 {
        format!("{s:.3} s")
    } else if s < 60.0 {
        format!("{s:.1} s")
    } else if s < 3600.0 {
        let total = s.round() as u64;
        format!("{} min {} s", total / 60, total % 60)
    } else if s < DAY {
        let minutes = (s / 60.0).round() as u64;
        format!("{} h {} min", minutes / 60, minutes % 60)
    } else if s < YEAR_S {
        format!("{:.1} d", s / DAY)
    } else {
        format!("{} yr", format_sig(s / YEAR_S, 3))
    }
}

/// An orbital or rotation period given in days: below a day as [`format_hms`], below 1 000 days
/// as "686.98 d", longer in Julian years ("29.45 yr").
pub fn format_period_days(d: f64) -> String {
    if !d.is_finite() {
        "n/a".into()
    } else if d.abs() < 1.0 {
        format_hms(d * DAY)
    } else if d.abs() < 1000.0 {
        format!("{} d", format_grouped(d, 2))
    } else {
        format!("{} yr", format_grouped(d / 365.25, 2))
    }
}

/// The TDB calendar date of `e`, "YYYY-MM-DD": for orbital epochs and perihelion times, which
/// are TDB instants. SBDB's epochs sit at 0 h TDB, 69 s after midnight UTC, so a UTC date would
/// show the day before the catalogue's epoch.
pub fn format_date_tdb(e: Epoch) -> String {
    /// JD (TDB) of 1970-01-01 0 h, the day count's origin (Hinnant's `civil_from_days`).
    const JD_1970: f64 = 2_440_587.5;
    let days = (e.jd_tdb() - JD_1970).floor();
    if !days.is_finite() {
        return "n/a".into();
    }
    let (year, month, day) = civil_from_days(days as i64);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Formats a time-warp rate (simulated seconds per real second).
pub(crate) fn format_time_rate(rate: f64) -> String {
    if rate == 0.0 {
        return "paused".into();
    }
    if !rate.is_finite() {
        return "n/a".into();
    }
    let size = rate.abs();
    let sign = if rate < 0.0 { "\u{2212}" } else { "" };
    if (1e-3..1e4).contains(&size) {
        // n / 1000 prints as its shortest round-trip decimal, so at most three decimals; the
        // integer part is grouped ("× 1 000").
        let rounded = (size * 1e3).round() / 1e3;
        let decimals = format!("{rounded}")
            .split_once('.')
            .map_or(0, |(_, f)| f.len());
        return format!("× {sign}{}", format_grouped(rounded, decimals));
    }
    let exp = size.log10().round() as i32;
    if (size / 10f64.powi(exp) - 1.0).abs() < 1e-9 {
        format!("× {sign}10{}", superscript(exp))
    } else {
        format!("× {}", format_sci(rate, 3))
    }
}

/// The proleptic Gregorian `(year, month, day)` that is `days` after 1970-01-01
/// (Hinnant 2013, "chrono-Compatible Low-Level Date Algorithms", `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Count from 0000-03-01 so the leap day falls at the end of each year; an era is 400 years.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    // In [0, 146096].
    let day_of_era = z.rem_euclid(146_097);
    // In [0, 399].
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    // In [0, 365], with March 1 = 0.
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    // In [0, 11], with March = 0.
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ruling 16: the mockup's grouped numbers, character for character.
    #[test]
    fn grouped_numbers() {
        assert_eq!(format_distance(1.42e7), "14 200 km");
        assert_eq!(format!("{} km", format_grouped(3389.5, 1)), "3 389.5 km");
        assert_eq!(format_mass_kg(6.4169e23), "6.417 × 10²³ kg");
        // Mars's sidereal day, 88 642.663 s (Allison & McEwen 2000, PSS 48, 215).
        assert_eq!(format_hms(88_642.663), "24 h 37 m 23 s");
        // Light time over Mars's 1.524 AU (760.5 s).
        assert_eq!(format_duration(1.524 * AU / C), "12 min 40 s");
        assert_eq!(format_time_rate(1000.0), "× 1 000");
        // Groups, signs and exponents.
        assert_eq!(format_grouped(999.0, 0), "999");
        assert_eq!(format_grouped(1_000_000.0, 0), "1 000 000");
        assert_eq!(format_grouped(-1234.5, 1), "\u{2212}1 234.5");
        assert_eq!(format_grouped(0.25, 2), "0.25");
        assert_eq!(format_sci(4.14e-5, 3), "4.14 × 10⁻⁵");
        assert_eq!(format_sci(4210.0, 3), "4.21 × 10³");
        assert_eq!(format_sci(9.9996e2, 3), "1.00 × 10³");
    }

    #[test]
    fn distances() {
        assert_eq!(format_distance(500.0), "500.0 m");
        assert_eq!(format_distance(1.5e6), "1 500 km");
        assert_eq!(format_distance(AU), "1.000 AU");
        assert_eq!(format_distance(1.3384 * PC), "1.34 pc (4.37 ly)");
        assert_eq!(format_distance(8200.0 * PC), "8.20 kpc");
        assert_eq!(format_distance(16.8e6 * PC), "16.80 Mpc");
        assert_eq!(format_distance(2.32e8 * PC), "232.00 Mpc");
        assert_eq!(format_distance(2.5e10 * PC), "25 000.00 Mpc");
    }

    #[test]
    fn speeds() {
        assert_eq!(format_speed(12.0), "12.0 m/s");
        assert_eq!(format_speed(29_780.0), "29.78 km/s");
        assert_eq!(format_speed(1.2e6), "1 200.00 km/s");
        assert_eq!(format_speed(0.5 * C), "0.500 c");
        assert_eq!(
            format_speed(1000.0 * C),
            "1.00 × 10³ c · FTL (non-physical)"
        );
    }

    #[test]
    fn epoch_format() {
        assert_eq!(
            format_epoch(Epoch::parse_iso_utc("2026-11-20T00:00:00Z").unwrap()),
            "2026-11-20 00:00:00 UTC"
        );
    }

    /// Each threshold is exclusive (`<`) except the speed of light, which is still physical.
    #[test]
    fn unit_thresholds() {
        assert_eq!(format_distance(1e3), "1 km");
        assert_eq!(format_distance(0.01 * AU), "0.010 AU");
        assert_eq!(format_distance(0.1 * PC), "0.10 pc (0.33 ly)");
        assert_eq!(format_distance(1e3 * PC), "1.00 kpc");
        assert_eq!(format_distance(1e6 * PC), "1.00 Mpc");
        assert_eq!(format_speed(1e3), "1.00 km/s");
        assert_eq!(format_speed(0.01 * C), "0.010 c");
        assert_eq!(format_speed(C), "1.000 c");
    }

    /// Times of day, leap days and dates before the Unix epoch.
    #[test]
    fn epoch_format_time_of_day_and_pre_1970() {
        for (iso, shown) in [
            ("2024-02-29T23:59:59Z", "2024-02-29 23:59:59 UTC"),
            ("2026-11-20T13:45:07.25Z", "2026-11-20 13:45:07 UTC"),
            ("1969-07-20T20:17:40Z", "1969-07-20 20:17:40 UTC"),
        ] {
            assert_eq!(format_epoch(Epoch::parse_iso_utc(iso).unwrap()), shown);
        }
        // J2000.0 is 12:00:00 TT, which is 11:58:55.816 UTC (TT − UTC = 64.184 s in 2000).
        assert_eq!(format_epoch(Epoch::J2000), "2000-01-01 11:58:55 UTC");
    }

    /// The mockup's rate row ("× 1 000"); exact powers of ten from 10⁴ as "× 10⁶".
    #[test]
    fn time_rates() {
        assert_eq!(format_time_rate(0.0), "paused");
        assert_eq!(format_time_rate(1.0), "× 1");
        assert_eq!(format_time_rate(0.5), "× 0.5");
        assert_eq!(format_time_rate(1.0 / 3.0), "× 0.333");
        assert_eq!(format_time_rate(1000.0), "× 1 000");
        assert_eq!(format_time_rate(-1.0), "× \u{2212}1");
        assert_eq!(format_time_rate(1e6), "× 10⁶");
        assert_eq!(format_time_rate(1e12), "× 10¹²");
        assert_eq!(format_time_rate(2.5e6), "× 2.50 × 10⁶");
        assert_eq!(format_time_rate(-1e9), "× \u{2212}10⁹");
        assert_eq!(format_time_rate(1e-4), "× 10⁻⁴");
        assert_eq!(format_time_rate(2e-4), "× 2.00 × 10⁻⁴");
    }

    /// Orbital epochs print their TDB date: SBDB's 0 h TDB is 23:58:51 UTC the day before.
    #[test]
    fn tdb_dates() {
        assert_eq!(format_date_tdb(Epoch::J2000), "2000-01-01");
        assert_eq!(
            format_date_tdb(Epoch::from_days_since_j2000(2_461_000.5 - 2_451_545.0)),
            "2025-11-21"
        );
        assert_eq!(
            format_date_tdb(Epoch::from_days_since_j2000(2_440_587.5 - 2_451_545.0)),
            "1970-01-01"
        );
    }

    /// Durations from a fraction of a second (light time to a nearby moon) to years.
    #[test]
    fn durations() {
        assert_eq!(format_duration(0.047_37), "0.047 s");
        assert_eq!(format_duration(4.24), "4.2 s");
        assert_eq!(
            format_duration(3600.0 * 5.0 + 33.0 * 60.0 + 10.0),
            "5 h 33 min"
        );
        assert_eq!(format_duration(2.5 * DAY), "2.5 d");
        assert_eq!(format_duration(4.37 * 365.25 * DAY), "4.37 yr");
        assert_eq!(format_duration(f64::NAN), "n/a");
        assert_eq!(format_hms(59.4), "59 s");
        assert_eq!(
            format_hms(7.0 * 3600.0 + 39.0 * 60.0 + 14.0),
            "7 h 39 m 14 s"
        );
        assert_eq!(format_hms(39.0 * 60.0 + 14.0), "39 m 14 s");
        assert_eq!(format_hms(9.925 * 3600.0), "9 h 55 m 30 s");
    }
}
