use crate::{max, min, prelude::*};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Channels {
    Mono,
    RGB,
}

#[derive(Debug, Clone)]
pub struct Color {
    pub channels: Channels,
    pub values: Vec<f64>,
}

impl Default for Color {
    fn default() -> Self {
        Color {
            channels: Channels::Mono,
            values: vec![0.0],
        }
    }
}

impl Color {
    pub fn new_rgb(r: f64, g: f64, b: f64) -> Color {
        Color {
            channels: Channels::RGB,
            values: vec![r, g, b],
        }
    }
    pub fn new_mono(v: f64) -> Color {
        Color {
            channels: Channels::Mono,
            values: vec![v],
        }
    }
    pub fn get_channel_value(&self, c: usize) -> f64 {
        if c >= self.values.len() {
            panic!("Invalid color channel index: {}", c);
        } else {
            self.values[c]
        }
    }

    pub fn is_nonzero(&self) -> bool {
        match self.channels {
            Channels::Mono => self.get_channel_value(0) > 0.0,
            Channels::RGB => {
                self.get_channel_value(0) > 0.0
                    || self.get_channel_value(1) > 0.0
                    || self.get_channel_value(2) > 0.0
            }
        }
    }
}

/// A single two-dimensional point on a raster. Contains the x/y coordinate and the RGB values to be placed
/// into the buffer.
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub color: Color,
}

impl Default for Point {
    fn default() -> Self {
        Point {
            x: 0.0,
            y: 0.0,
            color: Color::default(),
        }
    }
}

impl Point {
    /// Simple creation for a Point.
    pub fn create_rgb(x: f64, y: f64, r: f64, g: f64, b: f64) -> Self {
        Point {
            x,
            y,
            color: Color::new_rgb(r, g, b),
        }
    }

    pub fn create_mono(x: f64, y: f64, v: f64) -> Self {
        Point {
            x,
            y,
            color: Color::new_mono(v),
        }
    }

    pub fn create(x: f64, y: f64) -> Self {
        Point {
            x,
            y,
            color: Color::default(),
        }
    }
}

/// A triangle (polygon) of three two-dimensional points
pub struct Triangle {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
}

impl Triangle {
    /// Determine if a two dimensional point is contained within the  area bounded by the triangle
    pub fn contains(&self, x: f64, y: f64) -> bool {
        let p = Point::create(x, y);
        let b0 = Triangle::sign(&p, &self.p0, &self.p1) <= 0.0;
        let b1 = Triangle::sign(&p, &self.p1, &self.p2) <= 0.0;
        let b2 = Triangle::sign(&p, &self.p2, &self.p0) <= 0.0;

        (b0 == b1) && (b1 == b2)
    }

    pub fn sign(p0: &Point, p1: &Point, p2: &Point) -> f64 {
        (p0.x - p2.x) * (p1.y - p2.y) - (p1.x - p2.x) * (p0.y - p2.y)
    }

    pub fn x_min(&self) -> f64 {
        min!(self.p0.x, self.p1.x, self.p2.x)
    }

    pub fn x_max(&self) -> f64 {
        max!(self.p0.x, self.p1.x, self.p2.x)
    }

    pub fn y_min(&self) -> f64 {
        min!(self.p0.y, self.p1.y, self.p2.y)
    }

    pub fn y_max(&self) -> f64 {
        max!(self.p0.y, self.p1.y, self.p2.y)
    }

    /// Determines an interpolated single-channel color value for a point in the triangle
    pub fn interpolate_color_channel(&self, x: f64, y: f64, c0: f64, c1: f64, c2: f64) -> f64 {
        let det = self.p0.x * self.p1.y - self.p1.x * self.p0.y + self.p1.x * self.p2.y
            - self.p2.x * self.p1.y
            + self.p2.x * self.p0.y
            - self.p0.x * self.p2.y;
        let a = ((self.p1.y - self.p2.y) * c0
            + (self.p2.y - self.p0.y) * c1
            + (self.p0.y - self.p1.y) * c2)
            / det;
        let b = ((self.p2.x - self.p1.x) * c0
            + (self.p0.x - self.p2.x) * c1
            + (self.p1.x - self.p0.x) * c2)
            / det;
        let c = ((self.p1.x * self.p2.y - self.p2.x * self.p1.y) * c0
            + (self.p2.x * self.p0.y - self.p0.x * self.p2.y) * c1
            + (self.p0.x * self.p1.y - self.p1.x * self.p0.y) * c2)
            / det;

        a * x + b * y + c
    }

    /// Determines an interpolated three-channel (RGB) color value for a point in the triangle
    pub fn interpolate_color_rgb(&self, x: f64, y: f64) -> Color {
        Color::new_rgb(
            self.interpolate_color_channel(
                x,
                y,
                self.p0.color.get_channel_value(0),
                self.p1.color.get_channel_value(0),
                self.p2.color.get_channel_value(0),
            ),
            self.interpolate_color_channel(
                x,
                y,
                self.p0.color.get_channel_value(1),
                self.p1.color.get_channel_value(1),
                self.p2.color.get_channel_value(1),
            ),
            self.interpolate_color_channel(
                x,
                y,
                self.p0.color.get_channel_value(2),
                self.p1.color.get_channel_value(2),
                self.p2.color.get_channel_value(2),
            ),
        )
    }

    /// Determines an interpolated three-channel (RGB) color value for a point in the triangle
    pub fn interpolate_color_mono(&self, x: f64, y: f64) -> Color {
        Color::new_mono(self.interpolate_color_channel(
            x,
            y,
            self.p0.color.get_channel_value(0),
            self.p1.color.get_channel_value(0),
            self.p2.color.get_channel_value(0),
        ))
    }

    /// Determines an interpolated three-channel (RGB) color value for a point in the triangle
    pub fn interpolate_color(&self, x: f64, y: f64) -> Color {
        self.interpolate_color_rgb(x, y)
        // match self.p0.color.channels {
        //     Channels::Mono => self.interpolate_color_mono(x, y),
        //     Channels::RGB => self.interpolate_color_rgb(x, y),
        // }
    }
}

/// Defines a buffer that can be drawn on using triangle or square polygons.
pub trait Drawable {
    /// Create a simple three-channel image buffer
    fn create(width: usize, height: usize) -> Self;

    /// Create a simple three-channel image buffer
    fn create_masked(width: usize, height: usize, mask_value: bool) -> Self;

    /// Paint a triangle on the buffer.
    fn paint_tri(&mut self, tri: &Triangle, avg_pixels: bool);

    /// Paint a triangle on the buffer.
    fn paint_tri_with_channel_rule<F: Fn(usize) -> bool>(
        &mut self,
        tri: &Triangle,
        avg_pixels: bool,
        channel_rule: F,
    );

    /// Paint a square on the buffer using four points
    fn paint_square(&mut self, tl: &Point, bl: &Point, br: &Point, tr: &Point, avg_pixels: bool);

    /// Paint a square on the buffer using four points
    fn paint_square_with_channel_rule<F: Fn(usize) -> bool>(
        &mut self,
        tl: &Point,
        bl: &Point,
        br: &Point,
        tr: &Point,
        avg_pixels: bool,
        channel_rule: F,
    );

    /// Width of the buffer
    fn get_width(&self) -> usize;

    /// Height of the buffer
    fn get_height(&self) -> usize;

    /// Converts color to mono
    fn to_mono(&mut self);
}

/// Implements the Drawable trait for the RgbImage class. This is probably later be merged fully into RgbImage
/// in the sciimg crate.
impl Drawable for Image {
    fn create(width: usize, height: usize) -> Self {
        Image::new_with_bands_masked(width, height, 3, ImageMode::U16BIT, true).unwrap()
    }

    fn create_masked(width: usize, height: usize, mask_value: bool) -> Self {
        Image::new_with_bands_masked(width, height, 3, ImageMode::U16BIT, mask_value).unwrap()
    }

    fn get_width(&self) -> usize {
        self.width
    }

    fn get_height(&self) -> usize {
        self.height
    }

    fn paint_tri(&mut self, tri: &Triangle, avg_pixels: bool) {
        self.paint_tri_with_channel_rule(tri, avg_pixels, |_| true);
    }

    //pub fn unwrap_or_else<F: FnOnce(E) -> T>(self, op: F) -> T {
    fn paint_tri_with_channel_rule<F: Fn(usize) -> bool>(
        &mut self,
        tri: &Triangle,
        avg_pixels: bool,
        channel_rule: F,
    ) {
        let min_x = tri.x_min().floor() as usize;
        let max_x = tri.x_max().ceil() as usize;
        let min_y = tri.y_min().floor() as usize;
        let max_y = tri.y_max().ceil() as usize;

        let num_channels = match tri.p0.color.channels {
            Channels::Mono => 1,
            Channels::RGB => 3,
        };

        // Gonna limit the max dimension of a poly to just 100x100
        // to prevent those that wrap the entire image.
        // Until I plan out a better control to handle polygons that
        // wrap the cut-off azimuth
        if max_x - min_x < 100 && max_y - min_y < 100 {
            for y in min_y..=max_y {
                for x in min_x..=max_x {
                    if x < self.width && y < self.height && tri.contains(x as f64, y as f64) {
                        let interpolated_color: Color = tri.interpolate_color(x as f64, y as f64);

                        let point_mask = self.get_band(0).get_mask_at_point(x, y);
                        self.put_alpha(x, y, true);

                        for channel in 0..num_channels {
                            let mut v = interpolated_color.get_channel_value(channel);
                            let v0 = self.get_band(channel).get(x, y).unwrap() as f64;
                            if point_mask && avg_pixels && v0 > 0.0 {
                                v = (v + v0) / 2.0;
                            }

                            if channel_rule(channel) {
                                self.put(x, y, v as f32, channel);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Paints a square on the image by breaking it into two triangles.
    fn paint_square(&mut self, tl: &Point, bl: &Point, br: &Point, tr: &Point, avg_pixels: bool) {
        self.paint_square_with_channel_rule(tl, bl, br, tr, avg_pixels, |_| true)
    }

    /// Paints a square on the image by breaking it into two triangles.
    #[allow(clippy::redundant_closure)]
    fn paint_square_with_channel_rule<F: Fn(usize) -> bool>(
        &mut self,
        tl: &Point,
        bl: &Point,
        br: &Point,
        tr: &Point,
        avg_pixels: bool,
        channel_rule: F,
    ) {
        self.paint_tri_with_channel_rule(
            &Triangle {
                p0: tl.clone(),
                p1: bl.clone(),
                p2: tr.clone(),
            },
            avg_pixels,
            |c| channel_rule(c),
        );
        self.paint_tri_with_channel_rule(
            &Triangle {
                p0: tr.clone(),
                p1: bl.clone(),
                p2: br.clone(),
            },
            avg_pixels,
            |c| channel_rule(c),
        );
    }

    fn to_mono(&mut self) {
        if self.num_bands() != 3 {
            panic!("Cannot convert to mono: Already mono or unsupported number of bands");
        }

        let r = self.get_band(0).scale(0.2125).unwrap();
        let g = self.get_band(1).scale(0.7154).unwrap();
        let b = self.get_band(2).scale(0.0721).unwrap();

        let m = r.add(&g).unwrap().add(&b).unwrap();

        self.set_band(&m, 0);
        self.set_band(&m, 1);
        self.set_band(&m, 2);
    }
}
