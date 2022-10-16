//! Attempt at hot pixel detection and removal.
//! Method:
//!     For each pixel (excluding image border pixels):
//!         1. Compute the standard deviation of a window of pixels (3x3, say)
//!         2. Compute the z-score for the target pixel
//!         3. If the z-score exceeds a threshold variance from the mean
//!            we replace the pixel value with a median filter

use crate::{error, imagebuffer::ImageBuffer, stats};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub struct ReplacedPixel {
    x: usize,
    y: usize,
    pixel_value: f32,
    z_score: f32,
}

pub struct HpcResults {
    pub buffer: ImageBuffer,
    pub replaced_pixels: Vec<ReplacedPixel>,
}

fn isolate_window(buffer: &ImageBuffer, window_size: i32, x: usize, y: usize) -> Vec<f32> {
    let mut v: Vec<f32> = Vec::with_capacity(36);
    let start = window_size / 2 * -1;
    let end = window_size / 2 + 1;
    for _y in start..end {
        for _x in start..end {
            let get_x = x as i32 + _x;
            let get_y = y as i32 + _y;
            if get_x >= 0
                && get_x < buffer.width as i32
                && get_y >= 0
                && get_y < buffer.height as i32
            {
                v.push(buffer.get(get_x as usize, get_y as usize).unwrap());
            }
        }
    }
    v
}

//TODO: needs proper error-handling
pub fn hot_pixel_detection(
    buffer: &ImageBuffer,
    window_size: i32,
    threshold: f32,
) -> error::Result<HpcResults> {
    let mtx_map = Arc::new(Mutex::new(
        ImageBuffer::new(buffer.width, buffer.height).unwrap(),
    ));

    let replaced_pixels = (1..buffer.height - 1)
        .into_par_iter()
        .map(|y| {
            let cl_map = mtx_map.clone();
            (1..buffer.height - 1)
                .into_iter()
                .flat_map(|x| {
                    let pixel_value = buffer.get(x, y).unwrap();
                    let window = isolate_window(buffer, window_size, x, y);
                    let z_score = stats::z_score(pixel_value, &window[0..]).unwrap();
                    let mut map = cl_map.lock().unwrap();
                    if z_score > threshold {
                        let m = stats::mean(&window[0..]).unwrap();

                        map.put(x, y, m);

                        Some(ReplacedPixel {
                            x,
                            y,
                            pixel_value,
                            z_score,
                        })
                    } else {
                        map.put(x, y, pixel_value);
                        None
                    }
                })
                .collect::<Vec<ReplacedPixel>>()
        })
        .flatten()
        .collect::<Vec<ReplacedPixel>>();

    let lock = Arc::try_unwrap(mtx_map).expect("Lock still has multiple owners");
    let map = lock.into_inner().expect("Mutex cannot be locked");

    Ok(HpcResults {
        buffer: map,
        replaced_pixels,
    })
}
// Serial version
// on the hot_pixel_correction test, the parallel version (above) is consistently faster by about
// ~17 to 20%. (mileage may vary based on the image size etc...)
pub fn st_hot_pixel_detection(
    buffer: &ImageBuffer,
    window_size: i32,
    threshold: f32,
) -> error::Result<HpcResults> {
    let mut replaced_pixels: Vec<ReplacedPixel> = Vec::new();
    let mut map = ImageBuffer::new(buffer.width, buffer.height).unwrap();

    for y in 1..buffer.height - 1 {
        for x in 1..buffer.width - 1 {
            let pixel_value = buffer.get(x, y).unwrap();
            let window = isolate_window(buffer, window_size, x, y);
            let z_score = stats::z_score(pixel_value, &window[0..]).unwrap();
            if z_score > threshold {
                let m = stats::mean(&window[0..]).unwrap();
                map.put(x, y, m);

                replaced_pixels.push(ReplacedPixel {
                    x,
                    y,
                    pixel_value,
                    z_score,
                });
            } else {
                map.put(x, y, buffer.get(x, y).unwrap());
            }
        }
    }
    Ok(HpcResults {
        buffer: map,
        replaced_pixels,
    })
}
