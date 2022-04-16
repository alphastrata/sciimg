use crate::{
    error,
    vector::Vector,
    camera::model::*
};

use serde::{
    Deserialize, 
    Serialize
};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cahv {

    // Camera center vector C
    #[serde(with = "crate::vector::vector_format")]
    pub c: Vector,

    // Camera axis unit vector A
    #[serde(with = "crate::vector::vector_format")]
    pub a: Vector,

    // Horizontal information vector H
    #[serde(with = "crate::vector::vector_format")]
    pub h: Vector,

    // Vertical information vector V
    #[serde(with = "crate::vector::vector_format")]
    pub v: Vector
}

impl Cahv {
    pub fn default() -> Self {
        Cahv{
            c:Vector::default(),
            a:Vector::default(),
            h:Vector::default(),
            v:Vector::default()
        }
    }
}


impl CameraModelTrait for Cahv {

    fn model_type(&self) -> ModelType {
        ModelType::CAHV
    }

    fn c(&self) -> Vector {
        self.c.clone()
    }

    fn a(&self) -> Vector {
        self.a.clone()
    }

    fn h(&self) -> Vector {
        self.h.clone()
    }

    fn v(&self) -> Vector {
        self.v.clone()
    }

    fn o(&self) -> Vector {
        Vector::default()
    }

    fn r(&self) -> Vector {
        Vector::default()
    }

    fn e(&self) -> Vector {
        Vector::default()
    }

    fn box_clone(&self) -> Box<dyn CameraModelTrait + 'static> {
        Box::new((*self).clone())
    }

    fn f(&self) -> f64 {
        self.a.cross_product(&self.h).len()
    }

    // Adapted from https://github.com/NASA-AMMOS/VICAR/blob/master/vos/java/jpl/mipl/mars/pig/PigCoreCAHV.java
    fn ls_to_look_vector(&self, coordinate:&ImageCoordinate) -> error::Result<LookVector> {
        let line = coordinate.line;
        let samp = coordinate.sample;

        let origin = self.c.clone();

        let f = self.v.subtract(&self.a.scale(line));
        let g = self.h.subtract(&self.a.scale(samp));

        let mut look_direction = f.cross_product(&g).normalized();

        let t = self.v.cross_product(&self.h);
        if t.dot_product(&self.a) < 0.0 {
            look_direction = look_direction.inversed();
        }

        Ok(LookVector{
            origin:origin,
            look_direction:look_direction
        })
    }

    // Adapted from https://github.com/NASA-AMMOS/VICAR/blob/master/vos/java/jpl/mipl/mars/pig/PigCoreCAHV.java
    fn xyz_to_ls(&self, xyz:&Vector, infinity:bool) -> ImageCoordinate {
        if infinity {
            let x = xyz.dot_product(&self.a);
            ImageCoordinate {
                sample: xyz.dot_product(&self.h) / x,
                line: xyz.dot_product(&self.v) / x
            }
        } else {
            let d = xyz.subtract(&self.c);
            let range = d.dot_product(&self.a);
            let r_1 = 1.0 / range;
            ImageCoordinate {
                sample: d.dot_product(&self.h) * r_1,
                line: d.dot_product(&self.v) * r_1
            }
        }
    }

    fn pixel_angle_horiz(&self) -> f64 {
        let a = self.v.dot_product(&self.a);
        let s = self.a.scale(a);
        let f = self.v.subtract(&s).len();
        (1.0 / f).atan()
    }

    fn pixel_angle_vert(&self) -> f64 {
        let a = self.h.dot_product(&self.a);
        let s = self.a.scale(a);
        let f = self.h.subtract(&s).len();
        (1.0 / f).atan()
    }
}