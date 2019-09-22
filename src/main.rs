#[feature(async_await)]
extern crate opencv;
extern crate tch;

use opencv::core;
use opencv::highgui;
use opencv::videoio;

use tch::Tensor;

struct CameraCV {
    cam: videoio::VideoCapture,
    buf: core::Mat,
}

impl CameraCV {
    fn open(device: i32) -> Result<CameraCV, opencv::Error> {
        let mut cam = videoio::VideoCapture::new_with_backend(device, videoio::CAP_ANY)?;
        let opened = videoio::VideoCapture::is_opened(&cam)?;
        if !opened {
            let msg = format!("Cannot open device {}", device);
            return Err(opencv::Error::new(0, String::from(msg)));
        }
        let mut buf = core::Mat::default()?;

        Ok(CameraCV { cam, buf })
    }
}

// Cannot use Iterators for streaming easily without GADTs:
// https://stackoverflow.com/questions/30422177/how-do-i-write-an-iterator-that-returns
// -references-to-itself
// http://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html

impl Iterator for CameraCV {
    type Item = opencv::Result<&core::Mat>;

    fn next(&mut self) -> Option<Self::Item> {
        // Why is there no ? for Some(Err)
        let res = self.cam.read(&mut self.buf);

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

        let size = self.buf.size();
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

        Some(Ok(&self.buf))
    }
}

fn main() {
    println!("Hello, world!");

    let cam = CameraCV::open(0).expect("Cannot open camera");

    for img in cam {
        highgui::imshow("threaded", &img.unwrap()).unwrap();

        let key = highgui::wait_key(1).unwrap();
        if key == 27 {
            break;
        }
    }

    let t = Tensor::of_slice(&[10, 10]);
    println!("I can print tensors like {:?}", t);
}
