use opencv::core;
use opencv::highgui;
use opencv::videoio;

use genawaiter::{generator_mut, stack::{Co}};
use tch::Tensor;
use opencv::core::{DataType};


struct CameraCV {
    cam: videoio::VideoCapture,
}

impl CameraCV {
    fn open(device: i32) -> Result<CameraCV, opencv::Error> {
        let mut cam = videoio::VideoCapture::new_with_backend(device, videoio::CAP_ANY)?;
        let opened = videoio::VideoCapture::is_opened(&cam)?;
        if !opened {
            let msg = format!("Cannot open device {}", device);
            return Err(opencv::Error::new(0, String::from(msg)));
        }

        Ok(CameraCV { cam })
    }

    fn read_one(&mut self, buf: &mut core::Mat) -> Option<Result<(), opencv::Error>> {
        let res = self.cam.read(buf);

        match res {
            Err(e) => {
                return Some(Err(e));
            }
            Ok(res_ok) => {
                if !res_ok {
                    eprintln!("Camera read bad result: {}", res_ok);
                    return None;
                }
            }
        }

        let size = buf.size();
        match size {
            Err(e) => return Some(Err(e)),
            Ok(size) => {
                if size.width <= 0 {
                    eprintln!(
                        "Camera produced bad image with size: {}x{}",
                        size.width, size.height
                    );
                    return None;
                }
            }
        }

        Some(Ok(()))
    }
}

impl Iterator for CameraCV {
    // Cannot use Iterators for streaming easily without GADTs (we allocate a copy on read instead)
    // https://stackoverflow.com/questions/30422177/how-do-i-write-an-iterator-that-returns
    // -references-to-itself
    // http://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html

    type Item = opencv::Result<core::Mat>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = core::Mat::default().unwrap();
        let _res = self.read_one(&mut buf)?;

        let buf_copied = buf.clone().unwrap();
        return Some(Ok(buf_copied));
    }
}



/// Convert core::Mat types to torch Kinds,
/// this doesn't take into account the number of channels (use .channels to get this dimensionality)
fn Datatype_to_Kind(dtype: i32) -> tch::Kind
{
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

    match dtype
        {
            CV_8SC1 | CV_8SC2 | CV_8SC3 | CV_8SC4 => {tch::Kind::Int8},
            CV_8UC1 | CV_8UC2 | CV_8UC3 | CV_8UC4 => {tch::Kind::Uint8},
            CV_32FC1 | CV_32FC2 | CV_32FC3 | CV_32FC4 => {tch::Kind::Float},
            _ => panic!("Datatype CV2 not supported {}", dtype),
        }

}

fn kind_to_dtype(k: tch::Kind, last_dim: i64) -> i32 {

    use opencv::core::*;

    match(k, last_dim)
        {
            (tch::Kind::Uint8, 3) => CV_8UC3,
            _ => panic!("TCH Kind to CV2 not supported {:?}", k),
        }
}

fn tensor_from(mat: core::Mat) -> Tensor {

    let size = mat.size().unwrap();
    let chans = mat.channels().unwrap(); // Todo: does that return 0 or 1 for grayscale?
    let data_pointer = mat.data_typed().unwrap();
    let dtype_cv = mat.typ().unwrap();
    let dtype_tch = Datatype_to_Kind(dtype_cv);

    let shape = [size.height as i64, size.width as i64, chans as i64];
    Tensor::of_data_size(data_pointer, &shape, dtype_tch)

}


fn tensor_into(t: &Tensor) -> core::Mat {

    let shape = t.size();
    let ndim = shape.len();
    let last_dim = match ndim {
        2 => 1,
        3 => shape[ndim-1],
        _ => panic!("Only 2- or 3-dimensional Tensor supported, got {:?}", shape)
    };

    let dtype = kind_to_dtype(t.kind(), last_dim);

    // Todo: this mut pointer passing leads to invalid mem access
    /*

    #[repr(C)]
    pub struct C_tensor {
        _private: [u8; 0],
    }

    struct UglyTensor
    {
        pub c_tensor: *mut C_tensor,
    }

    let t_cont = t.contiguous(); // Todo: lazy contigous instead of deriving steps

    let mut t_exposed: UglyTensor = unsafe { std::mem::transmute(t_cont )};
    let mut ptr = unsafe {std::mem::transmute(t_exposed.c_tensor)};

    let mat = core::Mat::new_rows_cols_with_data(shape[0] as i32, shape[1] as i32,
                                                 dtype,
                                                 ptr,
                                                 core::Mat_AUTO_STEP);
    */

    // ... so we have to copy
    let mut mat = unsafe { core::Mat::new_rows_cols(shape[0] as i32, shape[1] as i32,
                                                    dtype).unwrap()  };
    let num_el = (shape[0] * shape[1]) as usize;
    t.copy_data::<u8>(mat.data_typed_mut().unwrap(), num_el);

    mat
}


async fn imgs(co: Co<'_, core::Mat>) {
    let cam = CameraCV::open(0).expect("Cannot open camera");
    for img in cam {
        co.yield_(img.unwrap()).await;
    }

}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    generator_mut!(gen, imgs);

    for img in gen {

        let tens: Tensor = tensor_from(img);
        println!("I can print tensors like {:?}", tens);
        let mat = tensor_into(&tens);
        println!("I can convert tensors to {:?}", mat.size().unwrap());
        highgui::imshow("generated", &mat)?;

        let key = highgui::wait_key(1).unwrap();
                if key == 27 {
                    break;
                }
    }

    Ok(())
}
