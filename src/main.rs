extern crate opencv;
extern crate tch;
extern crate tokio;

use std::env;
use std::error::Error;

use opencv::core;
use opencv::highgui;
use opencv::videoio;

use tch::Tensor;


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


async fn run_stream() -> Result<(), Box<dyn Error>> {
    println!("I'm in async");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let cam = CameraCV::open(0).expect("Cannot open camera");

    for img in cam {
        let img_ok = img?;

        highgui::imshow("threaded", &img_ok)?;

        let key = highgui::wait_key(1).unwrap();
        if key == 27 {
            break;
        }
    }

    let t = Tensor::of_slice(&[10, 10]);
    println!("I can print tensors like {:?}", t);

    let _async_res = run_stream().await?;

    Ok(())
}