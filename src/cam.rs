use opencv::prelude::*;
use opencv::videoio::{VideoCapture, VideoCaptureTrait};
use opencv::{core, videoio};

pub struct CameraCV {
    cam: videoio::VideoCapture,
}

impl CameraCV {
    pub fn open(
        device: i32,
        width: Option<usize>,
        height: Option<usize>,
    ) -> Result<CameraCV, opencv::Error> {
        let mut cam = VideoCapture::new(device, videoio::CAP_ANY)?;
        let opened = cam.is_opened()?;
        if !opened {
            let msg = format!("Cannot open device {}", device);
            return Err(opencv::Error::new(0, String::from(msg)));
        }

        // Set dimensions
        if let Some(w) = width {
            cam.set(opencv::videoio::CAP_PROP_FRAME_WIDTH, w as f64).unwrap();
            let true_width =
                cam.get(opencv::videoio::CAP_PROP_FRAME_WIDTH).unwrap().round() as usize;
            if w != true_width {
                return Err(opencv::Error::new(
                    0,
                    format!("Desired width {} wasn't set, got {}", w, true_width),
                ));
            }
        }

        if let Some(h) = height {
            cam.set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, h as f64).unwrap();
            let true_height =
                cam.get(opencv::videoio::CAP_PROP_FRAME_HEIGHT).unwrap().round() as usize;
            if h != true_height {
                return Err(opencv::Error::new(
                    0,
                    format!("Desired height {} wasn't set, got {}", h, true_height),
                ));
            }
        }

        Ok(CameraCV { cam })
    }

    fn read_one(&mut self, buf: &mut core::Mat) -> Option<Result<core::Size, opencv::Error>> {
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
                    eprintln!("Camera produced bad image with size: {:?}", size);
                    return None;
                }
                return Some(Ok(size));
            }
        }
    }
}

impl Iterator for CameraCV {
    type Item = opencv::Result<core::Mat>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = core::Mat::default().unwrap();
        let _res = self.read_one(&mut buf)?;
        Some(Ok(buf))
    }
}
