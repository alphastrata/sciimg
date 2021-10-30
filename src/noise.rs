
use fastblur::gaussian_blur;
extern crate lab;
use lab::{
    rgbs_to_labs,
    labs_to_rgbs,
    Lab
};

use crate::{
    error, 
    rgbimage::RgbImage,
    enums,
    imagebuffer::ImageBuffer
};

struct SplitLab {
    l: Vec<[u8; 3]>,
    a: Vec<[u8; 3]>,
    b: Vec<[u8; 3]>
}

fn split_lab_channels(lab_array:&[Lab]) -> SplitLab {
    let mut l: Vec<[u8; 3]> = Vec::with_capacity(lab_array.len());
    l.resize(lab_array.len(), [0, 0, 0]);

    let mut a: Vec<[u8; 3]> = Vec::with_capacity(lab_array.len());
    a.resize(lab_array.len(), [0, 0, 0]);

    let mut b: Vec<[u8; 3]> = Vec::with_capacity(lab_array.len());
    b.resize(lab_array.len(), [0, 0, 0]);

    for i in 0..lab_array.len() {
        l[i][0] = lab_array[i].l as u8;
        l[i][1] = lab_array[i].l as u8;
        l[i][2] = lab_array[i].l as u8;

        a[i][0] = lab_array[i].a as u8;
        a[i][1] = lab_array[i].a as u8;
        a[i][2] = lab_array[i].a as u8;

        b[i][0] = lab_array[i].b as u8;
        b[i][1] = lab_array[i].b as u8;
        b[i][2] = lab_array[i].b as u8;
    }

    SplitLab{l, 
        a, 
        b
    }
}

fn combine_lab_channels(splitlab:&SplitLab) -> Vec<Lab> {

    let mut lab_array:Vec<Lab> = Vec::with_capacity(splitlab.a.len());
    lab_array.resize(splitlab.a.len(), Lab{l:0.0, a:0.0, b:0.0});

    for i in 0..splitlab.a.len() {
        lab_array[i].l = splitlab.l[i][0] as f32;
        lab_array[i].a = splitlab.a[i][0] as f32;
        lab_array[i].b = splitlab.b[i][0] as f32;
    }

    lab_array
}

pub fn color_noise_reduction(image:&mut RgbImage, amount:i32) -> error::Result<RgbImage> {
    //let orig_mode = image.get_mode();

    if image.get_mode() != enums::ImageMode::U8BIT {
        //image.normalize_to_8bit_with_max(decompanding::get_max_for_instrument(image.get_instrument()) as f32).unwrap();
    }

    // We're juggling a couple different data structures here so we need to
    // convert the imagebuffer to a vec that's expected by lab and fastblur...

    let mut data: Vec<[u8; 3]> = Vec::with_capacity(image.width * image.height);
    data.resize(image.width * image.height, [0, 0, 0]);

    for y in 0..image.height {
        for x in 0..image.width {
            let r = image.get_band(0).unwrap().get(x, y).unwrap() as u8;
            let g = image.get_band(1).unwrap().get(x, y).unwrap() as u8;
            let b = image.get_band(2).unwrap().get(x, y).unwrap() as u8;
            let idx = (y * image.width) + x;
            data[idx][0] = r;
            data[idx][1] = g;
            data[idx][2] = b;
        }
    }

    let labs = rgbs_to_labs(&data);

    let mut split_channels = split_lab_channels(&labs);
    gaussian_blur(&mut split_channels.a, image.width, image.height, amount as f32);
    gaussian_blur(&mut split_channels.b, image.width, image.height, amount as f32);
    
    let labs_recombined = combine_lab_channels(&split_channels);

    let rgbs = labs_to_rgbs(&labs_recombined);

    let mut red = ImageBuffer::new_with_mask(image.width, image.height, &image.get_band(0).unwrap().mask).unwrap();
    let mut green = ImageBuffer::new_with_mask(image.width, image.height, &image.get_band(1).unwrap().mask).unwrap();
    let mut blue = ImageBuffer::new_with_mask(image.width, image.height, &image.get_band(2).unwrap().mask).unwrap();

    for y in 0..image.height {
        for x in 0..image.width {
            let idx = (y * image.width) + x;
            red.put(x, y, rgbs[idx][0] as f32);
            green.put(x, y, rgbs[idx][1] as f32);
            blue.put(x, y, rgbs[idx][2] as f32);
        }
    }

    let newimage = RgbImage::new_from_buffers_rgb(&red, &green, &blue, enums::ImageMode::U8BIT).unwrap();

    // TODO: Check value stretching!!
    // if orig_mode == enums::ImageMode::U12BIT {
    //     newimage.normalize_to_12bit_with_max(255.0).unwrap();
    // } else if orig_mode == enums::ImageMode::U16BIT {
    //     newimage.normalize_to_16bit_with_max(255.0).unwrap();
    // }

    Ok(newimage)
}

