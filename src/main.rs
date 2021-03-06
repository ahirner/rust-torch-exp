mod cam;

use cam::CameraCV;
use opencv::core;
use opencv::highgui;
use opencv::prelude::*;

use tch::Tensor;

/// Convert core::Mat types to torch Kinds,
/// this doesn't take into account the number of channels (use .channels to get this dimensionality)
fn dtype_to_kind(dtype: i32) -> tch::Kind {
    /*
    pub const CV_16SC1: i32 = 0x3; // 3
    pub const CV_16SC2: i32 = 0xb; // 11
    pub const CV_16SC3: i32 = 0x13; // 19
    pub const CV_16SC4: i32 = 0x1b; // 27
    pub const CV_16UC1: i32 = 0x2; // 2
    pub const CV_16UC2: i32 = 0xa; // 10
    pub const CV_16UC3: i32 = 0x12; // 18
    pub const CV_16UC4: i32 = 0x1a; // 26
    pub const CV_32FC1: i32 = 0x5; // 5
    pub const CV_32FC2: i32 = 0xd; // 13
    pub const CV_32FC3: i32 = 0x15; // 21
    pub const CV_32FC4: i32 = 0x1d; // 29
    pub const CV_32SC1: i32 = 0x4; // 4
    pub const CV_32SC2: i32 = 0xc; // 12
    pub const CV_32SC3: i32 = 0x14; // 20
    pub const CV_32SC4: i32 = 0x1c; // 28
    pub const CV_64FC1: i32 = 0x6; // 6
    pub const CV_64FC2: i32 = 0xe; // 14
    pub const CV_64FC3: i32 = 0x16; // 22
    pub const CV_64FC4: i32 = 0x1e; // 30
    pub const CV_8SC1: i32 = 0x1; // 1
    pub const CV_8SC2: i32 = 0x9; // 9
    pub const CV_8SC3: i32 = 0x11; // 17
    pub const CV_8SC4: i32 = 0x19; // 25
    pub const CV_8UC1: i32 = 0x0; // 0
    pub const CV_8UC2: i32 = 0x8; // 8
    pub const CV_8UC3: i32 = 0x10; // 16
    pub const CV_8UC4: i32 = 0x18; // 24
    */

    use opencv::core::*;

    match dtype {
        CV_8SC1 | CV_8SC2 | CV_8SC3 | CV_8SC4 => tch::Kind::Int8,
        CV_8UC1 | CV_8UC2 | CV_8UC3 | CV_8UC4 => tch::Kind::Uint8,
        CV_32FC1 | CV_32FC2 | CV_32FC3 | CV_32FC4 => tch::Kind::Float,
        _ => todo!("Datatype CV2 not supported {}", dtype),
    }
}

fn kind_to_dtype(k: tch::Kind, chans: i64) -> i32 {
    use opencv::core::*;

    match (k, chans) {
        (tch::Kind::Uint8, 3) => CV_8UC3,
        _ => todo!("TCH Kind to CV2 not supported {:?}", k),
    }
}

fn tensor_from(mat: &core::Mat) -> Tensor {
    let size = mat.size().unwrap();
    let chans = mat.channels().unwrap(); // Todo: does that return 0 or 1 for grayscale?
    let dtype_cv = mat.typ().unwrap();

    // Note: data_typed::<Vec3b> -> &[Vec3b] cannot be implemented for kind for Tensor::of_slice
    // hence we dance with the unsafe slice
    let data_pointer = unsafe { mat.data_typed_unchecked() }.unwrap();
    let kind_tch = dtype_to_kind(dtype_cv);
    let shape = [size.height as i64, size.width as i64, chans as i64];
    Tensor::of_data_size(data_pointer, &shape, kind_tch)
}

fn tensor_into(t: &Tensor) -> core::Mat {
    let shape = t.size();
    let ndim = shape.len();
    let chans = match ndim {
        2 => 1,
        3 => shape[ndim - 1],
        _ => panic!("Only 2- or 3-dimensional Tensor supported, got {:?}", shape),
    };

    let dtype = kind_to_dtype(t.kind(), chans);

    let mut mat =
        unsafe { core::Mat::new_rows_cols(shape[0] as i32, shape[1] as i32, dtype) }.unwrap();
    assert!(
        mat.is_continuous().unwrap(),
        "Underlying opencv Mat needs to be continuous for Tensor::copy_data"
    );

    // Note: Mat::data_typed_mut() returns the wrong slice length, namely without element size,
    // hence for 3 channel images 1/3 of its length. Thus, copy_data asserts an error. We work
    // around that by calculating the number of bytes manually and copying with this type.
    //println!("DEST {:?}", mat);
    let num_bytes = (shape[0] * shape[1] * mat.elem_size().unwrap() as i64) as usize;
    let dest_buf = unsafe { std::slice::from_raw_parts_mut(mat.data_mut() as *mut u8, num_bytes) };
    t.copy_data::<u8>(dest_buf, num_bytes);

    // Todo: need other than u8 types
    // Note: this is how conversion is done from C++ Mat: https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#implementing-the-custom-operator-in-c
    mat
}

trait FromMat {
    fn from_mat(mat: &core::Mat) -> Self;
}

trait ToMat {
    fn to_mat(self) -> core::Mat;
}

impl FromMat for Tensor {
    fn from_mat(mat: &core::Mat) -> Self {
        tensor_from(mat)
    }
}

impl ToMat for Tensor {
    // why self cannot be ref?
    fn to_mat(self) -> core::Mat {
        tensor_into(&self)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cam = CameraCV::open(0, None, None).expect("Cannot open camera");

    for img_maybe in cam {
        let img = img_maybe?;
        let mut tens: Tensor = Tensor::from_mat(&img);
        highgui::imshow("orig", &img)?;

        tens = tens.transpose(2, 0).to_kind(tch::Kind::Float).unsqueeze(0);
        println!("Tensor from img {:?}", tens);
        tens = tens.upsample_bilinear2d_out(&tens, &[320, 240], true, None, None);

        let img_tens = tens.to_kind(tch::Kind::Uint8).squeeze1(0).transpose(0, 2);
        let mat = img_tens.to_mat();
        println!("Mat from tens {:?}", mat);

        highgui::imshow("processed", &mat)?;
        let key = highgui::wait_key(1)?;
        if key == 27 {
            break;
        }
    }

    Ok(())
}
