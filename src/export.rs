//! A board as a plain drawing — paths and images in world units — for export.
//! The same shapes, strokes and glyph outlines the canvas paints, so an export
//! looks like the board. [`Drawing::to_svg`] writes it as SVG; a host can
//! rasterize the same list (Zorite does, for PNG).

use std::fmt::Write as _;

use gpui::Hsla;

use crate::font::{Font, Seg};
use crate::geometry::bbox;
use crate::paint::{
    DIAMOND_UNIT, TRIANGLE_UNIT, hexagon_unit, hsla_to_u32, star_unit, thumbnail_text_outline,
    u32_to_hsla,
};
use crate::scene::{
    BoxGeom, Element, ElementKind, MindMapConnectorStyle, Scene, SegGeom, SegmentStyle,
};

/// A 2-D affine transform `[a, b, c, d, e, f]` (SVG's `matrix(...)`): a point
/// maps to `(a·x + c·y + e, b·x + d·y + f)`.
pub type Affine = [f32; 6];

const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// One thing to draw, in board order (later items on top).
#[derive(Clone, Debug)]
pub enum DrawItem {
    /// A path, filled and/or stroked (round caps and joins), after `transform`.
    Path {
        segs: Vec<Seg>,
        fill: Option<u32>,
        /// Color, width, dashed.
        stroke: Option<(u32, f32, bool)>,
        transform: Affine,
    },
    /// A host-managed image `src` stretched over `rect` (`[x, y, w, h]`), after
    /// `transform` (rotation about its center).
    Image {
        src: String,
        rect: [f32; 4],
        transform: Affine,
    },
}

/// A whole board ready to export: what to draw and the area it covers
/// (`[x, y, w, h]`, world units, padded).
#[derive(Clone, Debug)]
pub struct Drawing {
    pub bounds: [f32; 4],
    pub items: Vec<DrawItem>,
}

/// Room around the content in an export.
const PAD: f32 = 32.0;

/// Rotation by `rad` about `(cx, cy)`.
fn rotation(rad: f32, cx: f32, cy: f32) -> Affine {
    if rad == 0.0 {
        return IDENTITY;
    }
    let (s, c) = rad.sin_cos();
    [c, s, -s, c, cx - c * cx + s * cy, cy - s * cx - c * cy]
}

/// `m` after `n` (apply `n` first).
fn then(m: Affine, n: Affine) -> Affine {
    [
        m[0] * n[0] + m[2] * n[1],
        m[1] * n[0] + m[3] * n[1],
        m[0] * n[2] + m[2] * n[3],
        m[1] * n[2] + m[3] * n[3],
        m[0] * n[4] + m[2] * n[5] + m[4],
        m[1] * n[4] + m[3] * n[5] + m[5],
    ]
}

fn polygon(pts: impl IntoIterator<Item = [f32; 2]>) -> Vec<Seg> {
    let mut out = Vec::new();
    for (i, p) in pts.into_iter().enumerate() {
        out.push(if i == 0 { Seg::Move(p) } else { Seg::Line(p) });
    }
    out.push(Seg::Close);
    out
}

fn ellipse(b: &BoxGeom) -> Vec<Seg> {
    // Four cubic quarter-arcs.
    const K: f32 = 0.552_284_8;
    let (cx, cy, rx, ry) = (b.x + b.w / 2.0, b.y + b.h / 2.0, b.w / 2.0, b.h / 2.0);
    let (kx, ky) = (rx * K, ry * K);
    vec![
        Seg::Move([cx + rx, cy]),
        Seg::Cubic([cx + rx, cy + ky], [cx + kx, cy + ry], [cx, cy + ry]),
        Seg::Cubic([cx - kx, cy + ry], [cx - rx, cy + ky], [cx - rx, cy]),
        Seg::Cubic([cx - rx, cy - ky], [cx - kx, cy - ry], [cx, cy - ry]),
        Seg::Cubic([cx + kx, cy - ry], [cx + rx, cy - ky], [cx + rx, cy]),
        Seg::Close,
    ]
}

fn round_rect(b: &BoxGeom) -> Vec<Seg> {
    let r = b.w.abs().min(b.h.abs()) * 0.2;
    let (x0, y0, x1, y1) = (b.x, b.y, b.x + b.w, b.y + b.h);
    vec![
        Seg::Move([x0 + r, y0]),
        Seg::Line([x1 - r, y0]),
        Seg::Quad([x1, y0], [x1, y0 + r]),
        Seg::Line([x1, y1 - r]),
        Seg::Quad([x1, y1], [x1 - r, y1]),
        Seg::Line([x0 + r, y1]),
        Seg::Quad([x0, y1], [x0, y1 - r]),
        Seg::Line([x0, y0 + r]),
        Seg::Quad([x0, y0], [x0 + r, y0]),
        Seg::Close,
    ]
}

fn unit_polygon(b: &BoxGeom, unit: &[(f32, f32)]) -> Vec<Seg> {
    let (cx, cy, rx, ry) = (b.x + b.w / 2.0, b.y + b.h / 2.0, b.w / 2.0, b.h / 2.0);
    polygon(unit.iter().map(|u| [cx + u.0 * rx, cy + u.1 * ry]))
}

/// A line or arrow: the stroke, plus the filled head for an arrow.
fn segment(s: &SegGeom, arrow: bool, style: MindMapConnectorStyle, ink: u32) -> Vec<DrawItem> {
    let (p1, p2) = ([s.x1, s.y1], [s.x2, s.y2]);
    let dx = s.x2 - s.x1;
    let (path, end_dir) = match style {
        MindMapConnectorStyle::Straight => (
            vec![Seg::Move(p1), Seg::Line(p2)],
            [s.x2 - s.x1, s.y2 - s.y1],
        ),
        MindMapConnectorStyle::Bezier => {
            let (c1, c2) = ([s.x1 + dx * 0.35, s.y1], [s.x2 - dx * 0.35, s.y2]);
            (
                vec![Seg::Move(p1), Seg::Cubic(c1, c2, p2)],
                [p2[0] - c2[0], p2[1] - c2[1]],
            )
        }
        MindMapConnectorStyle::Orthogonal => {
            let mx = s.x1 + dx * 0.5;
            (
                vec![
                    Seg::Move(p1),
                    Seg::Line([mx, s.y1]),
                    Seg::Line([mx, s.y2]),
                    Seg::Line(p2),
                ],
                [s.x2 - mx, 0.0],
            )
        }
    };
    let mut out = vec![DrawItem::Path {
        segs: path,
        fill: None,
        stroke: Some((ink, s.width, s.style == SegmentStyle::Dashed)),
        transform: IDENTITY,
    }];
    let len = end_dir[0].hypot(end_dir[1]);
    if arrow && len > 0.01 {
        let (ux, uy) = (end_dir[0] / len, end_dir[1] / len);
        let head = (s.width * 6.0).max(8.0);
        let barb = |a: f32| {
            let (sn, c) = a.sin_cos();
            [
                p2[0] + head * (-ux * c + uy * sn),
                p2[1] + head * (-ux * sn - uy * c),
            ]
        };
        out.push(DrawItem::Path {
            segs: polygon([p2, barb(0.45), barb(-0.45)]),
            fill: Some(ink),
            stroke: None,
            transform: IDENTITY,
        });
    }
    out
}

impl Scene {
    /// The whole board as a [`Drawing`] for export (SVG via
    /// [`Drawing::to_svg`], or rasterized by the host), with text set in
    /// `font`. `ink` colors whatever has no color of its own — pass a dark one
    /// for a light background, since a dark theme's ink is white. `None` for
    /// an empty board.
    pub fn to_drawing(&self, font: &Font, ink: Hsla) -> Option<Drawing> {
        drawing(self, font, ink)
    }
}

fn drawing(scene: &Scene, font: &Font, ink: Hsla) -> Option<Drawing> {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for e in &scene.elements {
        let (a, b, c, d) = bbox(&e.kind);
        (x0, y0, x1, y1) = (x0.min(a), y0.min(b), x1.max(c), y1.max(d));
    }
    if x0 > x1 {
        return None;
    }
    let ink_u = hsla_to_u32(ink);
    let mut items = Vec::new();
    for e in &scene.elements {
        items.extend(element_items(scene, e, font, ink_u));
    }
    Some(Drawing {
        bounds: [x0 - PAD, y0 - PAD, x1 - x0 + PAD * 2.0, y1 - y0 + PAD * 2.0],
        items,
    })
}

fn element_items(scene: &Scene, e: &Element, font: &Font, ink: u32) -> Vec<DrawItem> {
    let stroke = e.stroke.unwrap_or(ink);
    let mut out = Vec::new();
    let shape = |segs: Vec<Seg>, b: &BoxGeom| DrawItem::Path {
        segs,
        fill: e.fill,
        stroke: Some((stroke, b.width, false)),
        transform: rotation(b.rotation, b.x + b.w / 2.0, b.y + b.h / 2.0),
    };
    match &e.kind {
        ElementKind::Draw(s) if s.points.len() > 1 => out.push(DrawItem::Path {
            segs: s
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| if i == 0 { Seg::Move(*p) } else { Seg::Line(*p) })
                .collect(),
            fill: None,
            stroke: Some((stroke, s.width, false)),
            transform: IDENTITY,
        }),
        ElementKind::Draw(_) => {}
        ElementKind::Rect(b) => out.push(shape(
            polygon([
                [b.x, b.y],
                [b.x + b.w, b.y],
                [b.x + b.w, b.y + b.h],
                [b.x, b.y + b.h],
            ]),
            b,
        )),
        ElementKind::Ellipse(b) => out.push(shape(ellipse(b), b)),
        ElementKind::RoundRect(b) => out.push(shape(round_rect(b), b)),
        ElementKind::Diamond(b) => out.push(shape(unit_polygon(b, &DIAMOND_UNIT), b)),
        ElementKind::Triangle(b) => out.push(shape(unit_polygon(b, &TRIANGLE_UNIT), b)),
        ElementKind::Star(b) => out.push(shape(unit_polygon(b, &star_unit()), b)),
        ElementKind::Hexagon(b) => out.push(shape(unit_polygon(b, &hexagon_unit()), b)),
        ElementKind::Sticky(b) => {
            let (body, flap) = crate::paint::sticky_parts(b);
            let paper = e.fill.unwrap_or(crate::paint::STICKY_COLORS[0]);
            let flap_color = hsla_to_u32(crate::paint::sticky_flap(u32_to_hsla(paper)));
            let transform = rotation(b.rotation, b.x + b.w / 2.0, b.y + b.h / 2.0);
            let side = b.w.abs().min(b.h.abs());
            out.push(DrawItem::Path {
                segs: polygon(body.map(|[x, y]| [x + side * 0.01, y + side * 0.025])),
                fill: Some(0x00000024),
                stroke: None,
                transform,
            });
            for (pts, color) in [(polygon(body), paper), (polygon(flap), flap_color)] {
                out.push(DrawItem::Path {
                    segs: pts,
                    fill: Some(color),
                    stroke: None,
                    transform,
                });
            }
        }
        ElementKind::Line(s) | ElementKind::Arrow(s) => {
            let style = mindmap_style(scene, s);
            out.extend(segment(
                s,
                matches!(e.kind, ElementKind::Arrow(_)),
                style,
                stroke,
            ));
        }
        ElementKind::Embed(em) => {
            out.push(DrawItem::Path {
                segs: polygon([
                    [em.x, em.y],
                    [em.x + em.w, em.y],
                    [em.x + em.w, em.y + em.h],
                    [em.x, em.y + em.h],
                ]),
                fill: Some(0xFFFFFFFF),
                stroke: Some((stroke, 1.5, false)),
                transform: IDENTITY,
            });
            let title = font.layout(&em.title, 14.0);
            out.push(DrawItem::Path {
                segs: title.segs,
                fill: Some(stroke),
                stroke: None,
                transform: [1.0, 0.0, 0.0, 1.0, em.x + 12.0, em.y + 12.0],
            });
        }
        ElementKind::Image(im) => out.push(DrawItem::Image {
            src: im.src.clone(),
            rect: [im.x, im.y, im.w, im.h],
            transform: rotation(im.rotation, im.x + im.w / 2.0, im.y + im.h / 2.0),
        }),
        ElementKind::Text(_) => {}
    }
    // Text, and a shape's label, as the canvas lays them out (readable on a
    // fill, like the canvas). ponytail: underline / strike / highlight
    // decorations aren't exported yet.
    let label_color = e.label_color.or_else(|| {
        e.fill
            .and_then(|f| crate::paint::readable_on(f, u32_to_hsla(stroke)))
    });
    if let Some(t) = thumbnail_text_outline(
        font,
        &e.kind,
        u32_to_hsla(stroke),
        e.label.as_deref(),
        label_color,
        &e.styles,
        e.id,
        None,
        None,
    ) {
        let place = then(
            rotation(t.rotation, t.pivot[0], t.pivot[1]),
            [1.0, 0.0, 0.0, 1.0, t.x, t.y],
        );
        let color = hsla_to_u32(t.color);
        out.push(DrawItem::Path {
            segs: t.segs.to_vec(),
            fill: Some(color),
            stroke: None,
            transform: place,
        });
        if !t.bold_segs.is_empty() {
            out.push(DrawItem::Path {
                segs: t.bold_segs.to_vec(),
                fill: Some(color),
                stroke: Some((color, t.bold_width, false)),
                transform: place,
            });
        }
    }
    out
}

/// A mind-map edge keeps its node's connector style; other segments are straight.
fn mindmap_style(scene: &Scene, s: &SegGeom) -> MindMapConnectorStyle {
    s.end_anchor
        .and_then(|a| scene.elements.iter().find(|e| e.id == a.element_id))
        .and_then(|e| e.mindmap.as_ref())
        .map_or(MindMapConnectorStyle::Straight, |m| m.connector_style)
}

fn svg_color(c: u32) -> (String, f32) {
    (format!("#{:06x}", c >> 8), (c & 0xFF) as f32 / 255.0)
}

fn svg_path(segs: &[Seg]) -> String {
    let mut d = String::new();
    for s in segs {
        let _ = match s {
            Seg::Move(p) => write!(d, "M{} {}", p[0], p[1]),
            Seg::Line(p) => write!(d, "L{} {}", p[0], p[1]),
            Seg::Quad(c, p) => write!(d, "Q{} {} {} {}", c[0], c[1], p[0], p[1]),
            Seg::Cubic(a, b, p) => {
                write!(d, "C{} {} {} {} {} {}", a[0], a[1], b[0], b[1], p[0], p[1])
            }
            Seg::Close => write!(d, "Z"),
        };
    }
    d
}

impl Drawing {
    /// The drawing as an SVG document. `image_href` turns an image `src` into
    /// something an `<image href>` can show (a `data:` URI keeps the file
    /// self-contained); images it returns `None` for are left out.
    /// `background` paints behind everything (`None` = transparent).
    pub fn to_svg(
        &self,
        background: Option<u32>,
        image_href: impl Fn(&str) -> Option<String>,
    ) -> String {
        let [bx, by, bw, bh] = self.bounds;
        let mut out = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{bx} {by} {bw} {bh}\" width=\"{bw}\" height=\"{bh}\">\n"
        );
        if let Some(bg) = background {
            let (c, a) = svg_color(bg);
            let _ = writeln!(
                out,
                "<rect x=\"{bx}\" y=\"{by}\" width=\"{bw}\" height=\"{bh}\" fill=\"{c}\" fill-opacity=\"{a}\"/>"
            );
        }
        for item in &self.items {
            match item {
                DrawItem::Path {
                    segs,
                    fill,
                    stroke,
                    transform: t,
                } => {
                    let fill = fill.map_or("fill=\"none\"".to_string(), |f| {
                        let (c, a) = svg_color(f);
                        format!("fill=\"{c}\" fill-opacity=\"{a}\" fill-rule=\"nonzero\"")
                    });
                    let stroke = stroke.map_or(String::new(), |(s, w, dashed)| {
                        let (c, a) = svg_color(s);
                        let dash = if dashed {
                            format!(" stroke-dasharray=\"{} {}\"", (w * 4.5).max(10.0), (w * 2.5).max(6.0))
                        } else {
                            String::new()
                        };
                        format!(" stroke=\"{c}\" stroke-opacity=\"{a}\" stroke-width=\"{w}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"{dash}")
                    });
                    let _ = writeln!(
                        out,
                        "<path d=\"{}\" {fill}{stroke} transform=\"matrix({} {} {} {} {} {})\"/>",
                        svg_path(segs),
                        t[0],
                        t[1],
                        t[2],
                        t[3],
                        t[4],
                        t[5]
                    );
                }
                DrawItem::Image {
                    src,
                    rect,
                    transform: t,
                } => {
                    if let Some(href) = image_href(src) {
                        let _ = writeln!(
                            out,
                            "<image href=\"{href}\" x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"none\" transform=\"matrix({} {} {} {} {} {})\"/>",
                            rect[0], rect[1], rect[2], rect[3], t[0], t[1], t[2], t[3], t[4], t[5]
                        );
                    }
                }
            }
        }
        out.push_str("</svg>\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_then_translation_composes() {
        // Rotate a quarter turn about the origin after moving by (1, 0):
        // (0, 0) → (1, 0) → (0, 1).
        let m = then(
            rotation(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        );
        let (x, y) = (m[4], m[5]);
        assert!(x.abs() < 1e-5 && (y - 1.0).abs() < 1e-5, "{m:?}");
    }
}
