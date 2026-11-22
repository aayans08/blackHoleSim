//! A single-PNG contact sheet of every capture in a directory, for cheap visual checks.
//!
//! `uni-verify contact --captures <dir> --out <png> [--width 2400]` lists every `*.png` in the
//! captures directory, sorted by file name (the HTML sheet's order), downscales each to fit one
//! tile of a grid that is `--width` pixels wide, and labels each tile with its 1-based position
//! in that order, drawn in black on a yellow box in the tile's top-left corner. The legend
//! (position → file name) is returned so the caller can print it. Tiles keep their aspect
//! ratio on a mid-grey background, so black space stays distinguishable from the gaps.
//!
//! The downscale filter is image's `Triangle` (a tent filter widened by the scale factor, so it
//! averages every source pixel), applied to the sRGB-encoded values. Point-like stars lose
//! brightness when averaged; the sheet is for composition and gross errors, not photometry.

use std::path::{Path, PathBuf};

use anyhow::Context;
use image::{Rgb, RgbImage, imageops};

/// Default sheet width [px].
pub const DEFAULT_WIDTH: u32 = 2400;
/// Gap between tiles and around the sheet [px].
const GAP_PX: u32 = 8;
/// Sheet background (mid grey): visible against black sky.
const BACKGROUND: Rgb<u8> = Rgb([96, 96, 96]);
/// Label box and digit colours.
const LABEL_BOX: Rgb<u8> = Rgb([255, 200, 0]);
const LABEL_INK: Rgb<u8> = Rgb([0, 0, 0]);
/// Each font pixel is drawn as a square this many sheet pixels wide.
const FONT_SCALE: u32 = 4;
/// Margin inside the label box, in font pixels.
const LABEL_PAD: u32 = 1;

/// A 3×5 bitmap font for the digits 0–9: five rows of three bits each, most significant bit
/// on the left.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111],
    [0b010, 0b110, 0b010, 0b010, 0b111],
    [0b111, 0b001, 0b111, 0b100, 0b111],
    [0b111, 0b001, 0b111, 0b001, 0b111],
    [0b101, 0b101, 0b111, 0b001, 0b001],
    [0b111, 0b100, 0b111, 0b001, 0b111],
    [0b111, 0b100, 0b111, 0b101, 0b111],
    [0b111, 0b001, 0b001, 0b001, 0b001],
    [0b111, 0b101, 0b111, 0b101, 0b111],
    [0b111, 0b101, 0b111, 0b001, 0b111],
];

/// Columns of the grid for `n` tiles: ⌈√n⌉, so the sheet is roughly square in tiles.
fn columns(n: usize) -> u32 {
    let mut c = 1_u32;
    while (c as usize) * (c as usize) < n {
        c += 1;
    }
    c
}

/// Draws `n` (1-based position) in the top-left corner of the tile at (`x0`, `y0`).
fn draw_label(sheet: &mut RgbImage, x0: u32, y0: u32, n: usize) {
    let text = n.to_string();
    let glyphs = text.len() as u32;
    // Glyph 3 wide + 1 space; padding on both sides.
    let box_w = (glyphs * 4 - 1 + 2 * LABEL_PAD) * FONT_SCALE;
    let box_h = (5 + 2 * LABEL_PAD) * FONT_SCALE;
    let fill = |sheet: &mut RgbImage, x: u32, y: u32, w: u32, h: u32, c: Rgb<u8>| {
        for yy in y..(y + h).min(sheet.height()) {
            for xx in x..(x + w).min(sheet.width()) {
                sheet.put_pixel(xx, yy, c);
            }
        }
    };
    fill(sheet, x0, y0, box_w, box_h, LABEL_BOX);
    for (i, ch) in text.bytes().enumerate() {
        let glyph = DIGITS[usize::from(ch - b'0')];
        let gx = x0 + (LABEL_PAD + 4 * i as u32) * FONT_SCALE;
        let gy = y0 + LABEL_PAD * FONT_SCALE;
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    fill(
                        sheet,
                        gx + col * FONT_SCALE,
                        gy + row as u32 * FONT_SCALE,
                        FONT_SCALE,
                        FONT_SCALE,
                        LABEL_INK,
                    );
                }
            }
        }
    }
}

/// Tiles `images` (in order) into one sheet `width` pixels wide; see the module docs.
pub fn tile(images: &[RgbImage], width: u32) -> anyhow::Result<RgbImage> {
    anyhow::ensure!(!images.is_empty(), "no captures to tile");
    let cols = columns(images.len());
    let rows = (images.len() as u32).div_ceil(cols);
    let tile_w = width
        .checked_sub(GAP_PX * (cols + 1))
        .map(|w| w / cols)
        .filter(|&w| w >= 16)
        .with_context(|| format!("--width {width} is too narrow for {cols} columns"))?;
    // Tile height from the tallest aspect ratio, so every image fits its tile.
    let tile_h = images
        .iter()
        .map(|im| {
            (f64::from(tile_w) * f64::from(im.height()) / f64::from(im.width().max(1))).ceil()
                as u32
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let height = GAP_PX + rows * (tile_h + GAP_PX);
    let mut sheet = RgbImage::from_pixel(width, height, BACKGROUND);
    for (i, im) in images.iter().enumerate() {
        let (c, r) = (i as u32 % cols, i as u32 / cols);
        let x0 = GAP_PX + c * (tile_w + GAP_PX);
        let y0 = GAP_PX + r * (tile_h + GAP_PX);
        // Fit inside the tile, keeping the aspect ratio.
        let scale = (f64::from(tile_w) / f64::from(im.width().max(1)))
            .min(f64::from(tile_h) / f64::from(im.height().max(1)));
        let w = ((f64::from(im.width()) * scale).round() as u32).clamp(1, tile_w);
        let h = ((f64::from(im.height()) * scale).round() as u32).clamp(1, tile_h);
        let small = imageops::resize(im, w, h, imageops::FilterType::Triangle);
        imageops::replace(&mut sheet, &small, i64::from(x0), i64::from(y0));
        draw_label(&mut sheet, x0, y0, i + 1);
    }
    Ok(sheet)
}

/// Writes the contact sheet of every PNG in `captures` to `out` (PNG), skipping `out` itself
/// if it lies in `captures`. Returns the legend: (1-based position, file name).
pub fn write_contact(
    captures: &Path,
    out: &Path,
    width: u32,
) -> anyhow::Result<Vec<(usize, String)>> {
    let out_canon = std::fs::canonicalize(out).ok();
    let pngs: Vec<PathBuf> = crate::sheet::list_pngs(captures)?
        .into_iter()
        .filter(|p| out_canon.is_none() || std::fs::canonicalize(p).ok() != out_canon)
        .collect();
    anyhow::ensure!(!pngs.is_empty(), "no PNG files in {}", captures.display());
    let images = pngs
        .iter()
        .map(|p| crate::shadow::load_rgb8(p))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let sheet = tile(&images, width)?;
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    sheet
        .save(out)
        .with_context(|| format!("cannot write {}", out.display()))?;
    Ok(pngs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            (i + 1, name.into_owned())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_are_the_ceiling_square_root() {
        let got: Vec<u32> = [1, 2, 4, 5, 6, 9, 10].map(columns).to_vec();
        assert_eq!(got, [1, 2, 2, 3, 3, 3, 4]);
    }

    #[test]
    fn tiles_six_16_by_9_captures_in_a_3_by_2_grid() {
        let images: Vec<RgbImage> = (0..6)
            .map(|i| RgbImage::from_pixel(192, 108, Rgb([i * 40, 0, 0])))
            .collect();
        let sheet = tile(&images, 2400).unwrap();
        // 3 columns: (2400 − 4·8)/3 = 789 px tiles, 789·108/192 → 444 px tall.
        assert_eq!(sheet.width(), 2400);
        assert_eq!(sheet.height(), 8 + 2 * (444 + 8));
        // Tile 6 (row 1, column 2) holds the last image, away from its label.
        let (x, y) = (8 + 2 * (789 + 8) + 400, 8 + (444 + 8) + 300);
        assert_eq!(*sheet.get_pixel(x, y), Rgb([200, 0, 0]));
        // A gap pixel is background.
        assert_eq!(*sheet.get_pixel(3, 3), BACKGROUND);
        // Every tile has its yellow label box in the corner.
        assert_eq!(*sheet.get_pixel(8, 8), LABEL_BOX);
        // The "1" glyph's top-centre pixel is ink: box padding 1, column 1, row 0.
        assert_eq!(
            *sheet.get_pixel(8 + (1 + 1) * FONT_SCALE, 8 + FONT_SCALE),
            LABEL_INK
        );
    }

    #[test]
    fn rejects_empty_input_and_a_too_narrow_sheet() {
        assert!(tile(&[], 2400).is_err());
        let one = vec![RgbImage::new(4, 4); 4];
        assert!(tile(&one, 40).is_err());
    }

    #[test]
    fn writes_a_sheet_and_legend_for_a_directory() {
        let dir = std::env::temp_dir().join(format!("uni-verify-{}-contact", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["b", "a", "c"] {
            RgbImage::from_pixel(32, 18, Rgb([10, 20, 30]))
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        }
        // The output inside the captures directory is not tiled into a later sheet.
        let out = dir.join("contact.png");
        let legend = write_contact(&dir, &out, 400).unwrap();
        assert_eq!(
            legend,
            [
                (1, "a.png".into()),
                (2, "b.png".into()),
                (3, "c.png".into())
            ]
        );
        let again = write_contact(&dir, &out, 400).unwrap();
        assert_eq!(again.len(), 3);
        assert_eq!(image::open(&out).unwrap().width(), 400);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
