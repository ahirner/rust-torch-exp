extern crate opencv;
extern crate tch;

use opencv::core;
use opencv::videoio;
use opencv::highgui;

use tch::Tensor;

fn run(dev: i32, winname: &str) -> opencv::Result<()> {

    let mut cam = videoio::VideoCapture::new_with_backend(dev, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;

    if !opened
    {
        panic!(format!("Unable to open catpure device {}", dev));
    }

    println!("Opened VideoCapture device: {}", dev);

    highgui::named_window(winname, highgui::WINDOW_AUTOSIZE)?;

    let mut img = core::Mat::default()?;
    loop {

        let res = cam.read(&mut img)?;
        if !res | (img.size()?.width <= 0) {
            println!("Got no image or no good result");
            break;
        }
        highgui::imshow(winname, &img)?;

        let key = highgui::wait_key(1)?;

        if key == 27 { break; }
    }

    highgui::destroy_window(winname)?;

    Ok(())
}

fn main() {
    println!("Hello, world!");

    run(0, "test").unwrap();

    let t = Tensor::of_slice(&[10,10]);
    println!("I can print tensors like {:?}", t);
}
