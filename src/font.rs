//! On-canvas text rendering for the whiteboard.
//!
//! gpui paints text as glyph sprites whose transform is fixed to the identity,
//! so native text can't rotate or follow the camera the way the board needs.
//! Instead we render text as **vector outlines**: `ttf-parser` gives glyph
//! contours, which we lay out into board-local space and (in `lib.rs`) feed to a
//! `PathBuilder` fill — so text rotates, scales, and z-orders exactly like the
//! shapes. A face is just bytes, so a host can swap in a user-uploaded font; the
//! default (JetBrains Mono, OFL — see `assets/JetBrainsMono-OFL.txt`) is bundled
//! so the crate works standalone.

use std::cell::OnceCell;
use std::sync::{Arc, OnceLock};

/// The bundled default face.
const DEFAULT_FONT: &[u8] = include_bytes!("../assets/JetBrainsMono-Regular.ttf");

/// System faces tried in order for a char the board's face lacks (Arabic and
/// Hebrew in a CJK face, Hangul in a Chinese one, anything in an uploaded
/// face). Each is read only the first time some char needs it. Missing paths
/// are skipped, so one list serves every platform.
const FALLBACK_FACES: &[(&str, u32)] = &[
    // Windows
    ("C:/Windows/Fonts/segoeui.ttf", 0),
    ("C:/Windows/Fonts/malgun.ttf", 0),
    ("C:/Windows/Fonts/YuGothR.ttc", 0),
    ("C:/Windows/Fonts/msyh.ttc", 0),
    ("C:/Windows/Fonts/simsun.ttc", 0),
    ("C:/Windows/Fonts/seguisym.ttf", 0),
    // macOS
    ("/System/Library/Fonts/SFArabic.ttf", 0),
    ("/System/Library/Fonts/SFHebrew.ttf", 0),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 0),
    // Linux
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    (
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        0,
    ),
    (
        "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
        0,
    ),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
];

static FALLBACK_BYTES: [OnceLock<Option<Vec<u8>>>; FALLBACK_FACES.len()] =
    [const { OnceLock::new() }; FALLBACK_FACES.len()];

/// Fallback `i`, read and parsed (`None` if it's absent or unusable).
fn fallback_face(i: usize) -> Option<ttf_parser::Face<'static>> {
    let (path, index) = FALLBACK_FACES[i];
    let bytes = FALLBACK_BYTES[i].get_or_init(|| std::fs::read(path).ok());
    ttf_parser::Face::parse(bytes.as_deref()?, index).ok()
}

/// The face fontconfig picks for `ch`, for systems whose fonts aren't at the
/// paths in [`FALLBACK_FACES`] (NixOS keeps them in the store, Fedora and Arch
/// use their own folders). Faces found stay loaded and are tried first for
/// later chars, so a script costs one `fc-list` run, not one per char.
#[cfg(all(unix, not(target_os = "macos")))]
fn fontconfig_face(ch: char) -> Option<(&'static ttf_parser::Face<'static>, ttf_parser::GlyphId)> {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // ponytail: leaks each found face for the process's life; one per script.
    static FOUND: Mutex<Vec<(String, &'static ttf_parser::Face<'static>)>> = Mutex::new(Vec::new());
    static MISSES: OnceLock<Mutex<HashSet<char>>> = OnceLock::new();
    let mut found = FOUND.lock().ok()?;
    let arabic = is_arabic(ch);
    // A loaded face that has `ch` and can shape it; for Arabic, an Arabic
    // family only (see the sort below), any face once fc-list has no better.
    let loaded = |found: &[(String, &'static ttf_parser::Face<'static>)], any: bool| {
        found.iter().find_map(|(p, f)| {
            let named = !arabic || any || p.to_lowercase().contains("arab");
            Some((*f, f.glyph_index(ch).filter(|_| named && can_shape(f, ch))?))
        })
    };
    if let Some(hit) = loaded(&found, false) {
        return Some(hit);
    }
    let mut misses = MISSES.get_or_init(Default::default).lock().ok()?;
    if misses.contains(&ch) {
        return loaded(&found, true);
    }
    let hit = (|| {
        // fc-list, not fc-match: fc-match returns its best overall match even
        // when that face lacks `ch`, so a face already loaded for one script
        // came back for the next and the char counted as missing.
        let out = std::process::Command::new("fc-list")
            .args([
                &format!(":charset={:x}", ch as u32),
                "-f",
                "%{file}\t%{index}\n",
            ])
            .output()
            .ok()?;
        let out = String::from_utf8(out.stdout).ok()?;
        let mut faces: Vec<(&str, u32)> = out
            .lines()
            .filter_map(|l| l.split_once('\t'))
            .filter(|(p, _)| !found.iter().any(|(f, _)| f == p))
            .map(|(p, i)| (p, i.trim().parse::<u32>().unwrap_or(0) & 0xFFFF))
            .collect();
        // fc-list's order is arbitrary: catch-all faces (Unifont covers the
        // whole BMP with no shaping tables) last; for Arabic, an Arabic family
        // first, since math and number faces carry a few Arabic letters and a
        // word split across faces can't join; then upright regular faces.
        faces.sort_by_key(|(p, _)| {
            let p = p.to_lowercase();
            (
                p.contains("unifont") || p.contains("lastresort"),
                !(arabic && p.contains("arab")),
                !p.contains("regular"),
            )
        });
        // The first face that can shape `ch`; failing that, the first at all.
        // ponytail: tries 8 faces at most, each a full file read.
        let mut first = None;
        for (path, index) in faces.into_iter().take(8) {
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            let Ok(face) = ttf_parser::Face::parse(&bytes, index) else {
                continue;
            };
            if face.glyph_index(ch).is_some() && can_shape(&face, ch) {
                return Some(keep(&mut found, path, bytes, index, ch));
            }
            first.get_or_insert((path, bytes, index));
        }
        let (path, bytes, index) = first?;
        Some(keep(&mut found, path, bytes, index, ch))
    })()
    .flatten();
    if hit.is_none() {
        misses.insert(ch);
    }
    hit.or_else(|| loaded(&found, true))
}

/// Leak `bytes` as a face kept for the process's life, add it to `found`.
#[cfg(all(unix, not(target_os = "macos")))]
fn keep(
    found: &mut Vec<(String, &'static ttf_parser::Face<'static>)>,
    path: &str,
    bytes: Vec<u8>,
    index: u32,
    ch: char,
) -> Option<(&'static ttf_parser::Face<'static>, ttf_parser::GlyphId)> {
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let face: &'static _ = Box::leak(Box::new(ttf_parser::Face::parse(bytes, index).ok()?));
    found.push((path.to_string(), face));
    Some((face, face.glyph_index(ch)?))
}

/// Whether `face` can draw `ch` properly: a char that needs shaping (Arabic
/// joining, Indic, marks) needs the face's OpenType substitutions, or every
/// letter comes out in its isolated form.
fn can_shape(face: &ttf_parser::Face, ch: char) -> bool {
    if !needs_shaping(ch) {
        return true;
    }
    let Some(gsub) = face.tables().gsub else {
        return false;
    };
    // Arabic needs its own GSUB script: FreeMono has a GSUB table and Arabic
    // glyphs but no Arabic rules, so its letters stay isolated.
    // ponytail: only Arabic is checked by script; add tags as scripts need it.
    !is_arabic(ch)
        || gsub
            .scripts
            .index(ttf_parser::Tag::from_bytes(b"arab"))
            .is_some()
}

fn is_arabic(ch: char) -> bool {
    matches!(ch as u32, 0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF)
}

/// Which of a [`Faces`]' faces a glyph comes from.
#[derive(Clone, Copy, Debug)]
enum FaceId {
    Primary,
    Fallback(usize),
    #[cfg(all(unix, not(target_os = "macos")))]
    System(&'static ttf_parser::Face<'static>),
}

impl PartialEq for FaceId {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Primary, Self::Primary) => true,
            (Self::Fallback(a), Self::Fallback(b)) => a == b,
            #[cfg(all(unix, not(target_os = "macos")))]
            (Self::System(a), Self::System(b)) => std::ptr::eq(*a, *b),
            _ => false,
        }
    }
}

/// A font's face plus the system fallbacks (parsed on first use), so a layout
/// draws each char from the first face that has it. Metrics (ascent, line
/// height) stay the primary face's.
struct Faces<'a> {
    primary: ttf_parser::Face<'a>,
    fallbacks: [OnceCell<Option<ttf_parser::Face<'static>>>; FALLBACK_FACES.len()],
}

impl<'a> Faces<'a> {
    fn get(&self, id: FaceId) -> &ttf_parser::Face<'a> {
        match id {
            FaceId::Primary => &self.primary,
            FaceId::Fallback(i) => self.fallbacks[i]
                .get_or_init(|| fallback_face(i))
                .as_ref()
                .unwrap_or(&self.primary),
            #[cfg(all(unix, not(target_os = "macos")))]
            FaceId::System(f) => f,
        }
    }

    /// The face that has `ch` and its glyph; the primary's `.notdef` when no
    /// face has it.
    fn glyph(&self, ch: char) -> (FaceId, ttf_parser::GlyphId) {
        // A face that has `ch` and can shape it, then one that only has it.
        let shaped = |f: &ttf_parser::Face| f.glyph_index(ch).filter(|_| can_shape(f, ch));
        if let Some(g) = shaped(&self.primary) {
            return (FaceId::Primary, g);
        }
        if !ch.is_whitespace() && !ch.is_control() {
            for i in 0..self.fallbacks.len() {
                // An absent fallback resolves to the primary, which lacks `ch`.
                if let Some(g) = shaped(self.get(FaceId::Fallback(i))) {
                    return (FaceId::Fallback(i), g);
                }
            }
            #[cfg(all(unix, not(target_os = "macos")))]
            if let Some((f, g)) = fontconfig_face(ch) {
                return (FaceId::System(f), g);
            }
        }
        if let Some(g) = self.primary.glyph_index(ch) {
            return (FaceId::Primary, g);
        }
        if !ch.is_whitespace() && !ch.is_control() {
            for i in 0..self.fallbacks.len() {
                if let Some(g) = self.get(FaceId::Fallback(i)).glyph_index(ch) {
                    return (FaceId::Fallback(i), g);
                }
            }
        }
        (FaceId::Primary, ttf_parser::GlyphId(0))
    }
}

/// A face's units → world scale at `font_size`.
fn scale_of(face: &ttf_parser::Face, font_size: f32) -> f32 {
    let upem = face.units_per_em() as f32;
    if upem > 0.0 { font_size / upem } else { 0.0 }
}

/// Whether a run with `ch` needs the shaper: combining marks and the scripts
/// whose letters change form or order (Arabic, Hebrew points, Indic, Thai…).
/// ponytail: a block list, not script data; Latin, Greek, Cyrillic and CJK
/// skip shaping, so they keep the plain cmap path (and no code ligatures).
fn needs_shaping(ch: char) -> bool {
    matches!(ch as u32, 0x0300..=0x036F | 0x0483..=0x0489 | 0x0590..=0x1DFF | 0x200C..=0x200D | 0x20D0..=0x20FF | 0xA800..=0xABFF | 0xFB1D..=0xFDFF | 0xFE20..=0xFE2F | 0xFE70..=0xFEFF)
}

/// One shaped glyph: its face and id, the content byte its cluster starts at,
/// and its advance and offsets in world units (offsets y-up).
#[derive(Clone, Copy, Debug)]
struct Glyph {
    face: FaceId,
    id: ttf_parser::GlyphId,
    cluster: usize,
    adv: f32,
    dx: f32,
    dy: f32,
}

/// A run of one face and one direction: its content byte range and its glyphs
/// in visual (left-to-right) order.
struct Run {
    range: std::ops::Range<usize>,
    glyphs: Vec<Glyph>,
}

/// Shape the `range` of `line` (content byte offset `base`) in one face and
/// direction.
fn shape_run(
    faces: &Faces,
    line: &str,
    base: usize,
    range: std::ops::Range<usize>,
    (face, rtl): (FaceId, bool),
    font_size: f32,
) -> Run {
    let f = faces.get(face);
    let scale = scale_of(f, font_size);
    let text = &line[range.clone()];
    let at = base + range.start;
    let glyphs = if rtl || text.chars().any(needs_shaping) {
        let hb = rustybuzz::Face::from_face(f.clone());
        let mut buf = rustybuzz::UnicodeBuffer::new();
        buf.push_str(text);
        buf.guess_segment_properties();
        buf.set_direction(if rtl {
            rustybuzz::Direction::RightToLeft
        } else {
            rustybuzz::Direction::LeftToRight
        });
        let out = rustybuzz::shape(&hb, &[], buf);
        out.glyph_infos()
            .iter()
            .zip(out.glyph_positions())
            .map(|(i, p)| Glyph {
                face,
                id: ttf_parser::GlyphId(i.glyph_id as u16),
                cluster: at + i.cluster as usize,
                adv: p.x_advance as f32 * scale,
                dx: p.x_offset as f32 * scale,
                dy: p.y_offset as f32 * scale,
            })
            .collect()
    } else {
        text.char_indices()
            .map(|(i, ch)| {
                let id = f.glyph_index(ch).unwrap_or(ttf_parser::GlyphId(0));
                Glyph {
                    face,
                    id,
                    cluster: at + i,
                    adv: f.glyph_hor_advance(id).unwrap_or(0) as f32 * scale,
                    dx: 0.0,
                    dy: 0.0,
                }
            })
            .collect()
    };
    Run {
        range: at..base + range.end,
        glyphs,
    }
}

/// Split a hard line into face × direction runs and shape each. A char that
/// isn't a letter (space, punctuation, digit, mark) stays in the current face
/// when that face has it, so a word's spaces and marks shape with it.
fn shape_line(
    faces: &Faces,
    line: &str,
    base: usize,
    levels: &[unicode_bidi::Level],
    font_size: f32,
) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut cur: Option<(FaceId, bool)> = None;
    let mut start = 0;
    for (i, ch) in line.char_indices() {
        let rtl = levels[i].is_rtl();
        let face = match cur {
            Some((f, _)) if !ch.is_alphabetic() && faces.get(f).glyph_index(ch).is_some() => f,
            _ => faces.glyph(ch).0,
        };
        if cur != Some((face, rtl)) {
            if let Some(c) = cur {
                runs.push(shape_run(faces, line, base, start..i, c, font_size));
            }
            start = i;
            cur = Some((face, rtl));
        }
    }
    if let Some(c) = cur {
        runs.push(shape_run(
            faces,
            line,
            base,
            start..line.len(),
            c,
            font_size,
        ));
    }
    runs
}

/// Floor for shape-label auto-shrink (world units): a label never shrinks below
/// this, even if it then slightly overflows a very small box. See [`Font::fit_size`].
const MIN_LABEL_SIZE: f32 = 1.0;

/// One glyph-outline command in text-local space: origin at the block's
/// top-left, x to the right, y *down* (screen-like), in the same world units as
/// `font_size`. Curves keep their control points so they transform under
/// rotation / the camera before being flattened by the path builder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Seg {
    Move([f32; 2]),
    Line([f32; 2]),
    /// Quadratic Bézier: control, end.
    Quad([f32; 2], [f32; 2]),
    /// Cubic Bézier: control1, control2, end.
    Cubic([f32; 2], [f32; 2], [f32; 2]),
    Close,
}

/// A laid-out block of text: glyph outline segments plus metrics, all in
/// text-local space (origin = the block's top-left corner).
#[derive(Clone, Debug)]
pub struct TextLayout {
    pub segs: Vec<Seg>,
    /// Width of the widest line.
    pub width: f32,
    /// Total height (line count × line height).
    pub height: f32,
    /// Distance between successive line tops.
    pub line_height: f32,
    /// Top-left of the caret (just past the content), for the editing cursor.
    pub caret: [f32; 2],
}

/// A font backing whiteboard text. Holds raw TTF/OTF bytes (parsed on demand) so
/// it's cheap to clone and a host can supply its own face.
#[derive(Clone)]
pub struct Font {
    bytes: Arc<Vec<u8>>,
    index: u32,
}

impl Default for Font {
    fn default() -> Self {
        Self::system_cjk_fallback().unwrap_or_else(|| Self {
            bytes: Arc::new(DEFAULT_FONT.to_vec()),
            index: 0,
        })
    }
}

impl Font {
    fn system_cjk_fallback() -> Option<Self> {
        static SYSTEM_CJK: OnceLock<Option<(Arc<Vec<u8>>, u32)>> = OnceLock::new();

        SYSTEM_CJK
            .get_or_init(Self::load_system_cjk_fallback)
            .as_ref()
            .map(|(bytes, index)| Self {
                bytes: bytes.clone(),
                index: *index,
            })
    }

    fn load_system_cjk_fallback() -> Option<(Arc<Vec<u8>>, u32)> {
        // The editor uses GPUI's text stack, which gets system font fallback for
        // free. Whiteboard text is converted to vector outlines, so we need a
        // single face that actually contains CJK glyphs. Prefer broad Unicode / CJK
        // system fonts, then fall back to the bundled Latin-only JetBrains Mono.
        // This lookup is cached by `system_cjk_fallback`: CJK collections are often
        // tens of megabytes and must not be re-read for every whiteboard entity.
        let candidates: &[(&str, u32)] = &[
            // macOS
            ("/Library/Fonts/Arial Unicode.ttf", 0),
            ("/System/Library/Fonts/PingFang.ttc", 0),
            ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
            ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
            ("/System/Library/Fonts/STHeiti Light.ttc", 0),
            // Windows
            ("C:/Windows/Fonts/msyh.ttc", 0),
            ("C:/Windows/Fonts/simhei.ttf", 0),
            ("C:/Windows/Fonts/simsun.ttc", 0),
            // Linux common CJK packages
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
            ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
            (
                "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
                0,
            ),
        ];

        candidates.iter().find_map(|(path, index)| {
            let bytes = std::fs::read(path).ok()?;
            ttf_parser::Face::parse(&bytes, *index).ok()?;
            Some((Arc::new(bytes), *index))
        })
    }

    /// Build a font from raw face bytes (e.g. a user-uploaded `.ttf`/`.otf`),
    /// returning `None` if they don't parse. `index` selects a face within a
    /// collection (`.ttc`); pass 0 otherwise.
    pub fn from_bytes(bytes: Vec<u8>, index: u32) -> Option<Self> {
        ttf_parser::Face::parse(&bytes, index).ok()?;
        Some(Self {
            bytes: Arc::new(bytes),
            index,
        })
    }

    fn face(&self) -> Option<ttf_parser::Face<'_>> {
        ttf_parser::Face::parse(&self.bytes, self.index).ok()
    }

    fn faces(&self) -> Option<Faces<'_>> {
        Some(Faces {
            primary: self.face()?,
            fallbacks: Default::default(),
        })
    }

    /// The line height (top-to-top spacing) at `font_size`, world units.
    fn line_height_of(face: &ttf_parser::Face, scale: f32) -> f32 {
        (face.ascender() as f32 - face.descender() as f32 + face.line_gap() as f32) * scale
    }

    /// Lay out `content` at `font_size` (world units) into local-space outlines.
    /// Newlines break lines; the block's origin is its top-left corner.
    pub fn layout(&self, content: &str, font_size: f32) -> TextLayout {
        self.layout_wrapped(content, font_size, None)
    }

    /// Like [`layout`](Self::layout) but word-wraps to `max_width` (world units)
    /// when `Some` — for fitting a label inside a shape. Glyph positions come
    /// from the same [`line_stops`](Self::line_stops) the caret uses, so the
    /// caret always lands exactly between the rendered glyphs.
    pub fn layout_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
    ) -> TextLayout {
        let l = self.layout_styled(content, font_size, max_width, |_| GlyphStyle::default());
        TextLayout {
            segs: l.segs,
            width: l.width,
            height: l.height,
            line_height: l.line_height,
            caret: l.caret,
        }
    }

    /// Like [`layout_wrapped`](Self::layout_wrapped) but applies a per-character
    /// [`GlyphStyle`] from `style_at` (keyed by byte offset): synthetic italic
    /// (shear) and bold (a doubled, offset pass) are baked into the glyph
    /// outlines, and underline / strikethrough / highlight runs become
    /// [`Decoration`]s. Wrapping — and therefore the caret geometry — is identical
    /// to the unstyled path (synthetic styling doesn't change advances).
    pub fn layout_styled(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
        style_at: impl Fn(usize) -> GlyphStyle,
    ) -> StyledLayout {
        let Some(faces) = self.faces() else {
            return StyledLayout {
                segs: Vec::new(),
                bold_segs: Vec::new(),
                bold_width: 0.0,
                decorations: Vec::new(),
                width: 0.0,
                height: font_size,
                line_height: font_size,
                caret: [0.0, 0.0],
            };
        };
        let ascent = faces.primary.ascender() as f32 * scale_of(&faces.primary, font_size);
        let (lines, line_height) = self.line_stops(content, font_size, max_width);

        const SHEAR: f32 = 0.22; // ~12° oblique
        let bar = (font_size * 0.07).max(1.0); // underline / strike thickness

        let mut segs = Vec::new();
        // Bold glyphs' outlines, stroked over the solid fill (`bold_width`). A
        // doubled fill would cancel under even-odd winding → hollow glyphs.
        let mut bold_segs = Vec::new();
        let mut decorations = Vec::new();
        for line in &lines {
            let baseline = line.top + ascent;
            // The current decoration run: its style and starting local x. Only the
            // underline/strike/highlight bits matter (bold/italic are glyph-baked).
            // Walked in visual order, so a run is one contiguous stretch of x.
            let mut run: Option<(GlyphStyle, f32)> = None;
            for g in &line.glyphs {
                let st = style_at(g.cluster);
                let face = faces.get(g.face);
                let start = segs.len();
                face.outline_glyph(
                    g.id,
                    &mut Outliner {
                        segs: &mut segs,
                        pen: g.x + g.dx,
                        baseline: baseline - g.dy,
                        scale: scale_of(face, font_size),
                        shear: if st.italic { SHEAR } else { 0.0 },
                    },
                );
                if st.bold {
                    bold_segs.extend_from_slice(&segs[start..]);
                }
                // Track the decoration run; flush when underline/strike/highlight change.
                let deco = GlyphStyle {
                    bold: false,
                    italic: false,
                    ..st
                };
                let has_deco = deco.underline || deco.strike || deco.highlight.is_some();
                let same = matches!(&run, Some((rs, _)) if *rs == deco);
                if !same {
                    if let Some((rs, x0)) = run.take() {
                        flush_deco(
                            &mut decorations,
                            rs,
                            x0,
                            g.x,
                            line.top,
                            line_height,
                            baseline,
                            bar,
                        );
                    }
                    run = has_deco.then_some((deco, g.x));
                }
            }
            if let Some((rs, x0)) = run.take() {
                let x1 = line.glyphs.last().map_or(x0, |g| g.x + g.adv);
                flush_deco(
                    &mut decorations,
                    rs,
                    x0,
                    x1,
                    line.top,
                    line_height,
                    baseline,
                    bar,
                );
            }
        }
        StyledLayout {
            segs,
            bold_segs,
            bold_width: font_size * 0.06,
            decorations,
            width: lines.iter().fold(0.0, |m, l| m.max(l.width)),
            height: lines.len().max(1) as f32 * line_height,
            line_height,
            caret: end_caret(&lines),
        }
    }

    /// The block's `(width, height)` at `font_size` without building outlines —
    /// for selection bounds and hit-testing (no `Window` needed).
    pub fn measure(&self, content: &str, font_size: f32) -> (f32, f32) {
        self.measure_wrapped(content, font_size, None)
    }

    /// Like [`measure`](Self::measure) but wraps to `max_width` (world units)
    /// when `Some` — the height then reflects the wrapped line count.
    pub fn measure_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
    ) -> (f32, f32) {
        let (lines, line_height) = self.line_stops(content, font_size, max_width);
        let width = lines.iter().fold(0.0_f32, |m, l| m.max(l.width));
        (width, lines.len().max(1) as f32 * line_height)
    }

    /// The largest font size in `(0, max_size]` at which `content`, wrapped to
    /// `max_w`, fits within a `max_w × max_h` box (world units) — so a shape
    /// label can shrink until it no longer crosses the shape's border. Returns
    /// `max_size` for empty content or a non-positive box, and never goes below
    /// [`MIN_LABEL_SIZE`] (a tiny box may then still be overflowed).
    pub fn fit_size(&self, content: &str, max_w: f32, max_h: f32, max_size: f32) -> f32 {
        if content.is_empty() || max_w <= 0.0 || max_h <= 0.0 {
            return max_size;
        }
        let fits = |size: f32| {
            let (w, h) = self.measure_wrapped(content, size, Some(max_w));
            w <= max_w + 0.5 && h <= max_h + 0.5
        };
        if fits(max_size) {
            return max_size;
        }
        let (mut lo, mut hi) = (MIN_LABEL_SIZE, max_size.max(MIN_LABEL_SIZE));
        for _ in 0..20 {
            let mid = 0.5 * (lo + hi);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Per-visual-line glyphs and caret stops for `content`. Lines break on
    /// `'\n'` always and, when `max_width` is `Some(w)` (world units), also wrap
    /// greedily at word boundaries so each visual line stays within `w`; a single
    /// word wider than `w` is split between clusters so it never overflows.
    ///
    /// Each hard line is one bidi paragraph (direction from its first strong
    /// char): it's shaped in face × direction runs, wrapped in logical order,
    /// then each visual line is reordered (UAX #9 L2) and its glyphs placed left
    /// to right. Lines of a right-to-left paragraph align to the block's right
    /// edge. Soft wraps consume no byte, so offsets stay contiguous across
    /// visual lines. The backbone of drawing, caret placement, click
    /// hit-testing, and selection rects. Byte offsets index into `content`.
    fn line_stops(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
    ) -> (Vec<LineStops>, f32) {
        let Some(faces) = self.faces() else {
            return (
                vec![LineStops {
                    top: 0.0,
                    stops: vec![(0, 0.0)],
                    glyphs: Vec::new(),
                    width: 0.0,
                    rtl: false,
                }],
                font_size,
            );
        };
        let line_height = Self::line_height_of(&faces.primary, scale_of(&faces.primary, font_size));
        let wrap = max_width.filter(|w| *w > 0.0);

        let mut lines = Vec::new();
        let mut base = 0usize;
        let mut top = 0.0_f32;
        for line in content.split('\n') {
            let bidi = unicode_bidi::ParagraphBidiInfo::new(line, None);
            let runs = shape_line(&faces, line, base, &bidi.levels, font_size);
            // Logical cluster starts (line-relative) and their advances, for wrapping.
            let mut adv = vec![None::<f32>; line.len()];
            for g in runs.iter().flat_map(|r| &r.glyphs) {
                *adv[g.cluster - base].get_or_insert(0.0) += g.adv;
            }
            // Caret stops at cluster starts (x cumulative from the line's start,
            // logical order), plus whether each cluster ends in a space.
            let mut flat: Vec<(usize, f32)> = Vec::new();
            let mut space_after: Vec<bool> = Vec::new();
            let mut pen = 0.0_f32;
            for (i, a) in adv.iter().enumerate() {
                if let Some(a) = a {
                    if !flat.is_empty() {
                        space_after.push(line[..i].ends_with(' '));
                    }
                    flat.push((i, pen));
                    pen += a;
                }
            }
            if !flat.is_empty() {
                space_after.push(line.ends_with(' '));
            }
            flat.push((line.len(), pen));

            // Greedy word-wrap of `flat` into visual segments ≤ `w`.
            let mut segments = Vec::new();
            let n = flat.len();
            let mut s = 0usize; // segment start index into `flat`
            if let Some(w) = wrap {
                let mut e = 1usize; // current end index
                let mut last_break: Option<usize> = None; // index just past a space
                while e < n {
                    if flat[e].1 - flat[s].1 > w && e - s > 1 {
                        // Break at the last space if any, else split the word.
                        let brk = match last_break {
                            Some(lb) if lb > s => lb,
                            _ => e - 1,
                        };
                        segments.push(flat[s].0..flat[brk].0);
                        s = brk;
                        e = s + 1;
                        last_break = None;
                        continue;
                    }
                    if space_after.get(e - 1).copied().unwrap_or(false) {
                        last_break = Some(e);
                    }
                    e += 1;
                }
            }
            segments.push(flat[s].0..line.len());

            for seg in segments {
                lines.push(visual_line(content, &bidi, &runs, base, seg, top));
                top += line_height;
            }
            base += line.len() + 1; // +1 for the '\n' (harmless past the last line)
        }
        // A right-to-left paragraph's lines hug the block's right edge.
        let width = lines.iter().fold(0.0_f32, |m, l| m.max(l.width));
        for l in lines.iter_mut().filter(|l| l.rtl) {
            let shift = width - l.width;
            l.glyphs.iter_mut().for_each(|g| g.x += shift);
            l.stops.iter_mut().for_each(|s| s.1 += shift);
        }
        (lines, line_height)
    }

    /// Local-space top-left of the caret at content byte offset `at`. Out-of-range
    /// offsets clamp to the end of the text.
    pub fn caret_pos(&self, content: &str, font_size: f32, at: usize) -> [f32; 2] {
        self.caret_pos_wrapped(content, font_size, None, at)
    }

    /// [`caret_pos`](Self::caret_pos) honoring a `max_width` wrap (label editing).
    pub fn caret_pos_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
        at: usize,
    ) -> [f32; 2] {
        let (lines, _) = self.line_stops(content, font_size, max_width);
        for line in &lines {
            // A boundary byte belongs to exactly one line (the '\n' separates
            // line i's end from line i+1's start), so the first match is correct.
            for &(b, x) in &line.stops {
                if b == at {
                    return [x, line.top];
                }
            }
        }
        lines
            .last()
            .map(|l| {
                let &(_, x) = l.stops.last().unwrap();
                [x, l.top]
            })
            .unwrap_or([0.0, 0.0])
    }

    /// The caret stop one step left (`dir < 0`) or right of offset `at` on its
    /// visual line: the nearest stop strictly that way on screen (ties go to the
    /// logically nearest). `None` at the line's edge. Walking stops by x, not
    /// logical neighbors, keeps arrows moving through mixed-direction text.
    pub fn caret_step_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
        at: usize,
        dir: i32,
    ) -> Option<usize> {
        let (lines, _) = self.line_stops(content, font_size, max_width);
        let i = lines
            .iter()
            .position(|l| l.stops.iter().any(|s| s.0 == at))?;
        let line = &lines[i];
        let x = line.stops.iter().find(|s| s.0 == at)?.1;
        // A soft wrap's boundary offset sits on both lines but the caret draws
        // it on the first; only stops a line owns are places to land there.
        let owns = |j: usize, b: usize| j == 0 || !lines[j - 1].stops.iter().any(|s| s.0 == b);
        let step = line
            .stops
            .iter()
            .filter(|s| owns(i, s.0) && (s.1 - x) * dir as f32 > 0.01)
            .min_by(|a, b| {
                let key = |s: &(usize, f32)| ((s.1 - x).abs(), s.0.abs_diff(at));
                key(a).partial_cmp(&key(b)).unwrap()
            });
        if let Some(s) = step {
            return Some(s.0);
        }
        // At the edge of a soft-wrapped line, continue on the paragraph's
        // previous (reading back) or next visual line, entering from the side
        // the caret is moving away from. Soft wraps share their boundary
        // offset; a hard '\n' doesn't (→ `None`, the caller steps logically).
        let back = (dir < 0) != line.rtl;
        let j = if back { i.checked_sub(1)? } else { i + 1 };
        let adj = lines.get(j)?;
        let soft = adj
            .stops
            .iter()
            .any(|s| line.stops.iter().any(|t| t.0 == s.0));
        if !soft {
            return None;
        }
        adj.stops
            .iter()
            .filter(|s| owns(j, s.0))
            .min_by(|a, b| (a.1 * dir as f32).partial_cmp(&(b.1 * dir as f32)).unwrap())
            .map(|s| s.0)
    }

    /// The content byte offset whose caret sits nearest the local point `p`
    /// (text-local space). Picks the line by y, then the closest caret stop by x —
    /// so a click lands the caret between letters like a real text field.
    pub fn index_at(&self, content: &str, font_size: f32, p: [f32; 2]) -> usize {
        self.index_at_wrapped(content, font_size, None, p)
    }

    /// [`index_at`](Self::index_at) honoring a `max_width` wrap (label editing).
    pub fn index_at_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
        p: [f32; 2],
    ) -> usize {
        let (lines, line_height) = self.line_stops(content, font_size, max_width);
        if lines.is_empty() {
            return 0;
        }
        let li = if line_height > 0.0 {
            (p[1] / line_height)
                .floor()
                .clamp(0.0, (lines.len() - 1) as f32) as usize
        } else {
            0
        };
        let line = &lines[li];
        let mut best = line.stops[0];
        for &(b, x) in &line.stops {
            if (x - p[0]).abs() < (best.1 - p[0]).abs() {
                best = (b, x);
            }
        }
        best.0
    }

    /// Local-space highlight rectangles `[x, y, w, h]` for the selection
    /// `[start, end)` (byte offsets), one per line it covers. Lines fully inside a
    /// multi-line selection get a small trailing stub so the selected newline reads.
    pub fn selection_rects(
        &self,
        content: &str,
        font_size: f32,
        start: usize,
        end: usize,
    ) -> Vec<[f32; 4]> {
        self.selection_rects_wrapped(content, font_size, None, start, end)
    }

    /// [`selection_rects`](Self::selection_rects) honoring a `max_width` wrap.
    pub fn selection_rects_wrapped(
        &self,
        content: &str,
        font_size: f32,
        max_width: Option<f32>,
        start: usize,
        end: usize,
    ) -> Vec<[f32; 4]> {
        if start >= end {
            return Vec::new();
        }
        let (lines, line_height) = self.line_stops(content, font_size, max_width);
        let stub = font_size * 0.3; // trailing width shown for a selected newline
        let mut rects = Vec::new();
        for line in &lines {
            let lstart = line.stops[0].0;
            let (lend, end_x) = line.stops[line.stops.len() - 1];
            if end <= lstart || start > lend {
                continue; // selection doesn't touch this line
            }
            // The selected glyphs' spans, merged where they touch — a logical
            // range can split into several stretches across a direction change.
            let mut spans: Vec<(f32, f32)> = line
                .glyphs
                .iter()
                .filter(|g| (start..end).contains(&g.cluster))
                .map(|g| (g.x, g.x + g.adv))
                .collect();
            // The newline (and anything below) is selected when `end` runs past
            // this line — show a stub so an end-of-line/empty-line selection reads.
            if spans.is_empty() && end > lend {
                spans.push(if line.rtl {
                    (end_x - stub, end_x)
                } else {
                    (end_x, end_x + stub)
                });
            }
            let mut merged: Vec<(f32, f32)> = Vec::new();
            for (x0, x1) in spans {
                match merged.last_mut() {
                    Some(m) if x0 <= m.1 + 0.01 => m.1 = m.1.max(x1),
                    _ => merged.push((x0, x1)),
                }
            }
            for (x0, x1) in merged {
                if x1 > x0 {
                    rects.push([x0, line.top, x1 - x0, line_height]);
                }
            }
        }
        rects
    }
}

/// One visual line: its top y, its glyphs placed left to right, and the
/// `(content byte offset, local x)` caret stop at every char boundary in
/// logical order (the first and last are the line's start and end). A stop's x
/// is the *leading* edge of the char after it in reading order (the right edge
/// for right-to-left text); the line end's is the trailing edge of its
/// logically last char. See [`Font::line_stops`].
struct LineStops {
    top: f32,
    stops: Vec<(usize, f32)>,
    glyphs: Vec<Placed>,
    /// Total advance of the line's glyphs.
    width: f32,
    /// Whether the line's paragraph runs right to left.
    rtl: bool,
}

/// A shaped glyph placed on a line: `x` is the start of its advance.
#[derive(Clone, Copy, Debug)]
struct Placed {
    face: FaceId,
    id: ttf_parser::GlyphId,
    cluster: usize,
    x: f32,
    adv: f32,
    dx: f32,
    dy: f32,
    rtl: bool,
}

/// Lay out the visual line `seg` (line-relative bytes) of the hard line that
/// starts at content byte `base`: reorder its runs (UAX #9 L1–L2), place the
/// glyphs left to right, and derive the caret stops.
fn visual_line(
    content: &str,
    bidi: &unicode_bidi::ParagraphBidiInfo,
    runs: &[Run],
    base: usize,
    seg: std::ops::Range<usize>,
    top: f32,
) -> LineStops {
    let mut glyphs = Vec::new();
    let mut pen = 0.0_f32;
    if !seg.is_empty() {
        let (levels, vruns) = bidi.visual_runs(seg.clone());
        for vr in vruns {
            let rtl = levels[vr.start].is_rtl();
            let abs = base + vr.start..base + vr.end;
            let mut parts: Vec<&Run> = runs
                .iter()
                .filter(|r| r.range.start < abs.end && abs.start < r.range.end)
                .collect();
            if rtl {
                parts.reverse();
            }
            for g in parts.iter().flat_map(|r| &r.glyphs) {
                if abs.contains(&g.cluster) {
                    glyphs.push(Placed {
                        face: g.face,
                        id: g.id,
                        cluster: g.cluster,
                        x: pen,
                        adv: g.adv,
                        dx: g.dx,
                        dy: g.dy,
                        rtl,
                    });
                    pen += g.adv;
                }
            }
        }
    }
    // Clusters: (start byte, left x, right x, rtl), in logical order.
    let mut clusters: Vec<(usize, f32, f32, bool)> = Vec::new();
    for g in &glyphs {
        match clusters.last_mut() {
            Some(c) if c.0 == g.cluster => c.2 = g.x + g.adv,
            _ => clusters.push((g.cluster, g.x, g.x + g.adv, g.rtl)),
        }
    }
    clusters.sort_by_key(|c| c.0);
    let end = base + seg.end;
    let mut stops = Vec::with_capacity(seg.len() + 1);
    let mut trailing = 0.0;
    for (k, &(b, x0, x1, rtl)) in clusters.iter().enumerate() {
        let (lead, trail) = if rtl { (x1, x0) } else { (x0, x1) };
        let next = clusters.get(k + 1).map_or(end, |c| c.0);
        // Boundaries inside a multi-char cluster (a ligature, a base + marks)
        // split it evenly.
        let chars = content[b..next].char_indices().map(|(i, _)| b + i);
        let n = content[b..next].chars().count() as f32;
        for (j, at) in chars.enumerate() {
            stops.push((at, lead + (trail - lead) * j as f32 / n));
        }
        trailing = trail;
    }
    stops.push((end, trailing));
    LineStops {
        top,
        stops,
        glyphs,
        width: pen,
        rtl: bidi.paragraph_level.is_rtl(),
    }
}

/// The caret just past the content: the last line's last stop.
fn end_caret(lines: &[LineStops]) -> [f32; 2] {
    lines.last().map_or([0.0, 0.0], |l| {
        let &(_, x) = l.stops.last().unwrap();
        [x, l.top]
    })
}

/// The renderable style of one character — what [`Font::layout_styled`] needs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlyphStyle {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strike: bool,
    /// Highlight color behind the glyphs, packed `0xRRGGBBAA`.
    pub highlight: Option<u32>,
}

/// A non-glyph text decoration in text-local space — transformed like the
/// glyphs at paint time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoration {
    /// `[x, y, w, h]`.
    pub rect: [f32; 4],
    pub kind: DecoKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecoKind {
    /// Filled rect behind the glyphs, in this color (`0xRRGGBBAA`).
    Highlight(u32),
    /// A bar drawn in the text color.
    Underline,
    Strike,
}

/// A laid-out styled block: glyph outlines (italic / bold baked in) plus
/// decoration rects, all text-local. Like [`TextLayout`] otherwise.
#[derive(Clone, Debug)]
pub struct StyledLayout {
    pub segs: Vec<Seg>,
    /// The bold glyphs' outlines, stroked over the fill for synthetic bold.
    pub bold_segs: Vec<Seg>,
    /// Local stroke width for `bold_segs` (scale to screen px by the zoom).
    pub bold_width: f32,
    pub decorations: Vec<Decoration>,
    pub width: f32,
    pub height: f32,
    pub line_height: f32,
    pub caret: [f32; 2],
}

/// Emit the highlight / underline / strikethrough rects for a decoration run
/// spanning local x `[x0, x1)` on one line.
// Each argument is a distinct glyph-run geometry/style value; bundling them into
// a struct would obscure more than it clarifies.
#[allow(clippy::too_many_arguments)]
fn flush_deco(
    out: &mut Vec<Decoration>,
    st: GlyphStyle,
    x0: f32,
    x1: f32,
    top: f32,
    line_height: f32,
    baseline: f32,
    bar: f32,
) {
    if x1 <= x0 {
        return;
    }
    let w = x1 - x0;
    if let Some(c) = st.highlight {
        out.push(Decoration {
            rect: [x0, top, w, line_height],
            kind: DecoKind::Highlight(c),
        });
    }
    if st.underline {
        out.push(Decoration {
            rect: [x0, baseline + bar, w, bar],
            kind: DecoKind::Underline,
        });
    }
    if st.strike {
        // Through the x-height (about halfway from the line top to the baseline).
        out.push(Decoration {
            rect: [x0, (top + baseline) * 0.5, w, bar],
            kind: DecoKind::Strike,
        });
    }
}

/// Accumulates a glyph's outline into local space as `ttf-parser` walks it.
struct Outliner<'a> {
    segs: &'a mut Vec<Seg>,
    pen: f32,
    baseline: f32,
    scale: f32,
    /// Synthetic-italic slant: local x shifts right with height above the
    /// baseline (`0.0` = upright).
    shear: f32,
}

impl Outliner<'_> {
    /// Font units (y-up, baseline origin) → text-local (y-down, top-left origin),
    /// applying the italic shear.
    fn pt(&self, x: f32, y: f32) -> [f32; 2] {
        [
            self.pen + (x + self.shear * y) * self.scale,
            self.baseline - y * self.scale,
        ]
    }
}

impl ttf_parser::OutlineBuilder for Outliner<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.segs.push(Seg::Move(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        self.segs.push(Seg::Line(p));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let c = self.pt(x1, y1);
        let e = self.pt(x, y);
        self.segs.push(Seg::Quad(c, e));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let c1 = self.pt(x1, y1);
        let c2 = self.pt(x2, y2);
        let e = self.pt(x, y);
        self.segs.push(Seg::Cubic(c1, c2, e));
    }
    fn close(&mut self) {
        self.segs.push(Seg::Close);
    }
}

#[cfg(test)]
mod tests {
    /// The bundled face is Latin-only: Hangul, Hebrew and Arabic must come
    /// from a system fallback instead of drawing `.notdef` boxes.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn missing_glyphs_fall_back_to_system_faces() {
        let font = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        let faces = font.faces().unwrap();
        for ch in ['가', 'א', 'ب', '東'] {
            assert_ne!(faces.glyph(ch).1, ttf_parser::GlyphId(0), "{ch}");
        }
        assert_eq!(faces.glyph('a').0, FaceId::Primary);
    }

    use super::*;

    /// Arabic letters take their joined forms (initial, medial, final differ)
    /// and read right to left, from the system Arabic fallback.
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    #[test]
    fn arabic_is_shaped_and_joined() {
        let f = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        let (lines, _) = f.line_stops("ببب", 20.0, None);
        assert!(lines[0].glyphs.iter().all(|g| g.rtl));
        // The letters (a face may add the dots as zero-advance marks).
        let g: Vec<_> = lines[0].glyphs.iter().filter(|g| g.adv > 0.0).collect();
        assert_eq!(g.len(), 3, "{g:?}");
        let isolated = f.faces().unwrap().glyph('ب').1;
        assert!(g.iter().all(|g| g.id != isolated), "{g:?}");
        assert!(g[0].id != g[1].id && g[1].id != g[2].id, "{g:?}");
        // Visual order is reversed: the logically first letter draws rightmost.
        assert_eq!(
            g.iter().map(|g| g.cluster).collect::<Vec<_>>(),
            vec![4, 2, 0]
        );
    }

    /// A Hebrew run inside a Latin line reads right to left, and every caret
    /// stop has its own x that a click there gives back.
    #[test]
    fn mixed_line_orders_runs_and_caret() {
        let f = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        let s = "abc אבג"; // א at byte 4, ב 6, ג 8, end 10
        let x = |b| f.caret_pos(s, 20.0, b)[0];
        assert!(x(0) < x(1) && x(1) < x(2) && x(2) < x(3));
        assert!(
            x(4) > x(6) && x(6) > x(8) && x(8) > x(3),
            "{:?}",
            [4, 6, 8, 10].map(x)
        );
        assert!(
            (x(4) - f.measure(s, 20.0).0).abs() < 0.01,
            "alef leads at the right end"
        );
        for b in [0, 1, 2, 3, 4, 6, 8, 10] {
            assert_eq!(f.index_at(s, 20.0, [x(b), 0.0]), b, "byte {b}");
        }
        // Selecting "c א" covers two stretches: "c " and, at the far right, "א".
        let r = f.selection_rects(s, 20.0, 2, 6);
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r[1][0] > r[0][0] + r[0][2], "{r:?}");
    }

    /// A right-to-left paragraph starts at the right: its trailing `!` draws
    /// leftmost, and its lines align to the block's right edge.
    #[test]
    fn rtl_paragraph_starts_at_the_right() {
        let f = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        let s = "שלום!";
        let w = f.measure(s, 20.0).0;
        let x = |b| f.caret_pos(s, 20.0, b)[0];
        assert!(
            (x(0) - w).abs() < 0.01,
            "first letter at the right: {}",
            x(0)
        );
        assert!(x(s.len() - 1) < x(2), "! left of the letters");
        assert!(x(s.len()) < 0.01, "end caret at the left");
        let two = "שלום\nabcdefghij";
        let w = f.measure(two, 20.0).0;
        assert!((f.caret_pos(two, 20.0, 0)[0] - w).abs() < 0.01);
        assert!(
            f.caret_pos(two, 20.0, 9)[0] < 0.01,
            "the Latin line stays left"
        );
    }

    /// Stepping left from the visual right edge of mixed RTL + Latin text
    /// reaches the left edge (no ping-pong inside the Latin run).
    #[test]
    fn caret_steps_cross_mixed_direction_text() {
        let f = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        let s = "שלום 25 dollars טוב";
        let x = |b| f.caret_pos(s, 20.0, b)[0];
        let mut at = 0; // logical start = visual right edge
        let mut n = 0;
        while let Some(next) = f.caret_step_wrapped(s, 20.0, None, at, -1) {
            assert!(x(next) < x(at), "{at} -> {next}");
            at = next;
            n += 1;
            assert!(n <= s.len(), "stuck");
        }
        assert!(x(at) < 0.01, "ended at {at}, x {}", x(at));
    }

    /// Arrows cross the soft wrap of a mixed RTL paragraph both ways and
    /// reach the far end (no ping-pong at the wrap).
    #[test]
    fn caret_steps_cross_a_soft_wrap() {
        let f = Font::from_bytes(DEFAULT_FONT.to_vec(), 0).unwrap();
        for (s, rtl) in [
            ("שלום 25 dollars טוב", true),
            ("hello wide world again", false),
            ("שלום עולם גדול מאוד", true),
        ] {
            wrap_walk(&f, s, rtl);
        }
    }

    fn wrap_walk(f: &Font, s: &str, rtl: bool) {
        let w = Some(f.measure(s, 20.0).0 * 0.55);
        let walk = |from: usize, dir: i32| {
            let mut at = from;
            for _ in 0..2 * s.len() {
                match f.caret_step_wrapped(s, 20.0, w, at, dir) {
                    Some(n) => at = n,
                    None => return at,
                }
            }
            panic!("stuck walking {dir} from {from}");
        };
        assert!(f.line_stops(s, 20.0, w).0.len() > 1, "wraps");
        let d = if rtl { -1 } else { 1 }; // the reading direction on screen
        assert_eq!(walk(0, d), s.len(), "{s}: reaches the logical end");
        assert_eq!(walk(s.len(), -d), 0, "{s}: reaches the logical start");
    }

    #[test]
    fn default_font_parses_and_lays_out() {
        let f = Font::default();
        let l = f.layout("A", 20.0);
        assert!(l.width > 0.0, "width {}", l.width);
        assert!(l.line_height > 0.0);
        assert!(
            l.segs.iter().any(|s| matches!(s, Seg::Move(_))),
            "expected contours"
        );
        // JetBrains Mono is monospace: "MM" is twice the advance of "M".
        let one = f.measure("M", 20.0).0;
        let two = f.measure("MM", 20.0).0;
        assert!(one > 0.0 && (two - 2.0 * one).abs() < 0.5, "{one} {two}");
    }

    #[test]
    fn newlines_stack_lines_and_place_the_caret() {
        let f = Font::default();
        let one = f.measure("x", 20.0).1;
        let two = f.measure("x\ny", 20.0).1;
        assert!((two - 2.0 * one).abs() < 0.5, "{one} {two}");
        // The caret of a two-line block sits on the lower line.
        let l = f.layout("x\ny", 20.0);
        assert!(l.caret[1] > 0.0, "caret {:?}", l.caret);
    }

    #[test]
    fn empty_content_has_no_segments_but_a_caret() {
        let f = Font::default();
        let l = f.layout("", 20.0);
        assert!(l.segs.is_empty());
        assert_eq!(l.caret, [0.0, 0.0]);
        assert!(l.height > 0.0);
    }

    #[test]
    fn bad_bytes_are_rejected() {
        assert!(Font::from_bytes(vec![0, 1, 2, 3], 0).is_none());
    }

    #[test]
    fn caret_pos_walks_per_char_and_across_lines() {
        let f = Font::default();
        let a = f.measure("M", 20.0).0; // monospace advance
        let lh = f.layout("M", 20.0).line_height;
        // One stop per boundary on a line.
        assert_eq!(f.caret_pos("MMM", 20.0, 0), [0.0, 0.0]);
        assert!((f.caret_pos("MMM", 20.0, 1)[0] - a).abs() < 0.5);
        assert!((f.caret_pos("MMM", 20.0, 3)[0] - 3.0 * a).abs() < 0.5);
        // Byte 2 of "M\nM" is the start of the second line (col 0, lower row).
        let c = f.caret_pos("M\nM", 20.0, 2);
        assert!(c[0].abs() < 0.5 && (c[1] - lh).abs() < 0.5, "{c:?}");
        // Byte 3 is one advance into that second line.
        let c = f.caret_pos("M\nM", 20.0, 3);
        assert!((c[0] - a).abs() < 0.5 && (c[1] - lh).abs() < 0.5, "{c:?}");
    }

    #[test]
    fn index_at_picks_the_nearest_boundary() {
        let f = Font::default();
        let a = f.measure("M", 20.0).0;
        let lh = f.layout("M", 20.0).line_height;
        // Just past the first glyph's midpoint rounds to the next boundary.
        assert_eq!(f.index_at("MMM", 20.0, [0.4 * a, 0.0]), 0);
        assert_eq!(f.index_at("MMM", 20.0, [0.6 * a, 0.0]), 1);
        // Way off to the right clamps to the line end; clicking the lower line
        // (by y) lands on it.
        assert_eq!(f.index_at("MMM", 20.0, [99.0 * a, 0.0]), 3);
        assert_eq!(f.index_at("M\nMM", 20.0, [0.0, lh]), 2); // start of line 2
    }

    #[test]
    fn selection_rects_span_lines() {
        let f = Font::default();
        let a = f.measure("M", 20.0).0;
        // Single line, two chars selected → one rect ~2 advances wide.
        let r = f.selection_rects("MMMM", 20.0, 1, 3);
        assert_eq!(r.len(), 1);
        assert!(
            (r[0][0] - a).abs() < 0.5 && (r[0][2] - 2.0 * a).abs() < 0.5,
            "{r:?}"
        );
        // Across a newline → one rect per line.
        let r = f.selection_rects("M\nMM", 20.0, 0, 4);
        assert_eq!(r.len(), 2, "{r:?}");
        // Empty selection → nothing.
        assert!(f.selection_rects("MM", 20.0, 2, 2).is_empty());
    }

    #[test]
    fn wrap_breaks_long_lines_at_spaces() {
        let f = Font::default();
        let lh = f.measure("x", 20.0).1;
        let full = f.measure("hello world", 20.0).0;
        // Wrapping below the phrase width forces a second line.
        let (w, h) = f.measure_wrapped("hello world", 20.0, Some(full * 0.6));
        assert!(
            (h - 2.0 * lh).abs() < 0.5,
            "expected 2 lines: h={h} lh={lh}"
        );
        assert!(w <= full * 0.6 + 0.5, "lines stay within the width: {w}");
        // A width wider than the text wraps nothing — identical to no wrap.
        assert_eq!(
            f.measure_wrapped("hello world", 20.0, Some(10_000.0)),
            f.measure("hello world", 20.0),
        );
    }

    #[test]
    fn wrap_keeps_byte_offsets_contiguous() {
        let f = Font::default();
        let s = "alpha beta gamma";
        let full = f.measure(s, 20.0).0;
        // The end-of-content caret is still reachable (no byte dropped at a soft
        // break) and sits lower than the single-line end (more visual lines).
        let end = f.caret_pos_wrapped(s, 20.0, Some(full * 0.5), s.len());
        let plain = f.caret_pos(s, 20.0, s.len());
        assert!(end[1] > plain[1], "wrapped end lower: {end:?} vs {plain:?}");
    }

    #[test]
    fn fit_size_shrinks_for_a_small_box() {
        let f = Font::default();
        let big = f.fit_size("hello world", 1000.0, 1000.0, 40.0);
        let small = f.fit_size("hello world", 60.0, 60.0, 40.0);
        assert_eq!(big, 40.0, "fits at max in a big box");
        assert!(small < big, "shrinks in a small box: {small} vs {big}");
        // The shrunk label actually fits inside its box.
        let (w, h) = f.measure_wrapped("hello world", small, Some(60.0));
        assert!(w <= 60.5 && h <= 60.5, "fits: {w}x{h}");
    }

    #[test]
    fn styled_layout_bolds_and_decorates() {
        let f = Font::default();
        let plain = f.layout_styled("Ab", 20.0, None, |_| GlyphStyle::default());
        let bold = f.layout_styled("Ab", 20.0, None, |_| GlyphStyle {
            bold: true,
            ..Default::default()
        });
        assert!(
            !bold.bold_segs.is_empty() && plain.bold_segs.is_empty(),
            "bold collects outlines to stroke"
        );
        assert!(plain.decorations.is_empty());
        let und = f.layout_styled("Ab", 20.0, None, |_| GlyphStyle {
            underline: true,
            ..Default::default()
        });
        assert!(
            und.decorations
                .iter()
                .any(|d| d.kind == DecoKind::Underline)
        );
        let hi = f.layout_styled("Ab", 20.0, None, |_| GlyphStyle {
            highlight: Some(0xffff00ff),
            ..Default::default()
        });
        assert!(
            hi.decorations
                .iter()
                .any(|d| d.kind == DecoKind::Highlight(0xffff00ff))
        );
    }

    #[test]
    fn styled_layout_decoration_runs_split() {
        let f = Font::default();
        // Underline only the first of two chars → one underline run ~1 advance wide.
        let l = f.layout_styled("MM", 20.0, None, |b| GlyphStyle {
            underline: b == 0,
            ..Default::default()
        });
        let unders: Vec<_> = l
            .decorations
            .iter()
            .filter(|d| d.kind == DecoKind::Underline)
            .collect();
        assert_eq!(unders.len(), 1, "{:?}", l.decorations);
        let adv = f.measure("M", 20.0).0;
        assert!(
            (unders[0].rect[2] - adv).abs() < 0.5,
            "≈1 advance: {}",
            unders[0].rect[2]
        );
    }
}
