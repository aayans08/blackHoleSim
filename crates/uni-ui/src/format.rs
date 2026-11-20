//! Human-readable distances, speeds and dates for the HUD (M0 Task 10).

use uni_core::{AU, C, DAY, Epoch, LY, PC};

/// Formats a distance in metres, picking m, km, AU, pc (with ly), kpc or Mpc by size.
///
/// Each unit is used below its upper threshold (exclusive): 1 km, 0.01 AU, 0.1 pc, 1 kpc, 1 Mpc.
pub fn format_distance(m: f64) -> String {
    if m < 1e3 {
        format!("{m:.1} m")
    } else if m < 0.01 * AU {
        format!("{:.0} km", m / 1e3)
    } else if m < 0.1 * PC {
        format!("{:.3} AU", m / AU)
    } else if m < 1e3 * PC {
        format!("{:.2} pc ({:.2} ly)", m / PC, m / LY)
    } else if m < 1e6 * PC {
        format!("{:.2} kpc", m / (1e3 * PC))
    } else {
        format!("{:.2} Mpc", m / (1e6 * PC))
    }
}

/// Formats a speed in m/s as m/s, km/s or a fraction of c.
///
/// Exactly c is still shown as a fraction of c; anything faster is labelled non-physical.
pub fn format_speed(mps: f64) -> String {
    if mps < 1e3 {
        format!("{mps:.1} m/s")
    } else if mps < 0.01 * C {
        format!("{:.2} km/s", mps / 1e3)
    } else if mps <= C {
        format!("{:.3} c", mps / C)
    } else {
        format!("{:.2e} c · FTL (non-physical)", mps / C)
    }
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

/// Formats a time-warp rate (simulated seconds per real second), e.g. `"×1"`, `"×1.0e12"`.
///
/// Rates from 0.001 up to (not including) 10⁴ in size are shown plainly to at most three
/// decimals; others in scientific notation; zero is `"paused"`.
pub(crate) fn format_time_rate(rate: f64) -> String {
    if rate == 0.0 {
        "paused".into()
    } else if (1e-3..1e4).contains(&rate.abs()) {
        // n / 1000 prints as its shortest round-trip decimal, so at most three decimals.
        format!("×{}", (rate * 1e3).round() / 1e3)
    } else {
        format!("×{rate:.1e}")
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

    #[test]
    fn distances() {
        assert_eq!(format_distance(500.0), "500.0 m");
        assert_eq!(format_distance(1.5e6), "1500 km");
        assert_eq!(format_distance(AU), "1.000 AU");
        assert_eq!(format_distance(1.3384 * PC), "1.34 pc (4.37 ly)");
        assert_eq!(format_distance(8200.0 * PC), "8.20 kpc");
        assert_eq!(format_distance(16.8e6 * PC), "16.80 Mpc");
    }

    #[test]
    fn speeds() {
        assert_eq!(format_speed(12.0), "12.0 m/s");
        assert_eq!(format_speed(29_780.0), "29.78 km/s");
        assert_eq!(format_speed(0.5 * C), "0.500 c");
        assert_eq!(format_speed(1000.0 * C), "1.00e3 c · FTL (non-physical)");
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

    #[test]
    fn time_rates() {
        assert_eq!(format_time_rate(0.0), "paused");
        assert_eq!(format_time_rate(1.0), "×1");
        assert_eq!(format_time_rate(0.5), "×0.5");
        assert_eq!(format_time_rate(1.0 / 3.0), "×0.333");
        assert_eq!(format_time_rate(1000.0), "×1000");
        assert_eq!(format_time_rate(-1.0), "×-1");
        assert_eq!(format_time_rate(1e4), "×1.0e4");
        assert_eq!(format_time_rate(1e12), "×1.0e12");
        assert_eq!(format_time_rate(1e-4), "×1.0e-4");
    }
}
