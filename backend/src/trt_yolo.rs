#![cfg(feature = "trt_engine")]

use anyhow::{anyhow, Result};
use opencv::{
    core::{self, BorderTypes, Mat, Scalar, Size},
    imgproc,
    prelude::{MatTraitConst, MatTraitConstManual, MatTraitManual},
};

use crate::state::CONF_THRESH;
use crate::trt::TrtRunner;

#[derive(Clone, Copy)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub in_w: i32,
    pub in_h: i32,
}

#[derive(Clone, Copy)]
pub struct RawDet {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i64,
}

pub struct TrtYolo {
    runner: TrtRunner,
    input_w: i32,
    input_h: i32,
    num_classes: i32,
}

impl TrtYolo {
    pub fn new(engine_path: &str) -> Result<Self> {
        let runner = TrtRunner::new(engine_path)?;
        let (_, c, h, w) = runner.input_dims()?;
        if c != 3 {
            return Err(anyhow!("TensorRT engine input channels must be 3 (got {c})."));
        }
        let (_, out_c, _, _) = runner.output_dims()?;
        if out_c < 5 {
            return Err(anyhow!("Unexpected output channel count: {out_c}."));
        }
        let num_classes = out_c - 4;
        Ok(Self {
            runner,
            input_w: w,
            input_h: h,
            num_classes,
        })
    }

    pub fn predict(&self, frame: &Mat) -> Result<Vec<RawDet>> {
        let (input, letterbox) = preprocess(frame, self.input_w, self.input_h)?;
        let mut output = vec![0f32; self.runner.output_count()];
        self.runner.infer(&input, &mut output)?;
        decode_yolo(&output, self.num_classes as usize, &letterbox)
    }
}

fn preprocess(frame: &Mat, input_w: i32, input_h: i32) -> Result<(Vec<f32>, Letterbox)> {
    let in_w = frame.cols();
    let in_h = frame.rows();
    let scale = (input_w as f32 / in_w as f32).min(input_h as f32 / in_h as f32);
    let new_w = (in_w as f32 * scale).round() as i32;
    let new_h = (in_h as f32 * scale).round() as i32;
    let pad_x = (input_w - new_w) as f32 / 2.0;
    let pad_y = (input_h - new_h) as f32 / 2.0;

    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &resized,
        &mut padded,
        pad_y.floor() as i32,
        (input_h - new_h) - pad_y.floor() as i32,
        pad_x.floor() as i32,
        (input_w - new_w) - pad_x.floor() as i32,
        BorderTypes::BORDER_CONSTANT as i32,
        Scalar::all(114.0),
    )?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    let data = rgb.data_bytes()?;
    let hw = (input_w * input_h) as usize;
    let mut input = vec![0f32; hw * 3];
    for y in 0..input_h as usize {
        for x in 0..input_w as usize {
            let idx = (y * input_w as usize + x) * 3;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;
            let base = y * input_w as usize + x;
            input[base] = r;
            input[base + hw] = g;
            input[base + 2 * hw] = b;
        }
    }

    Ok((
        input,
        Letterbox {
            scale,
            pad_x,
            pad_y,
            in_w,
            in_h,
        },
    ))
}

fn decode_yolo(output: &[f32], num_classes: usize, lb: &Letterbox) -> Result<Vec<RawDet>> {
    let channels = num_classes + 4;
    if output.len() % channels != 0 {
        return Err(anyhow!("Unexpected output size: {}", output.len()));
    }
    let num = output.len() / channels;
    let mut dets = Vec::with_capacity(num);

    for i in 0..num {
        let x = output[i] ;
        let y = output[num + i];
        let w = output[2 * num + i];
        let h = output[3 * num + i];

        let mut best = 0f32;
        let mut best_id = 0usize;
        for c in 0..num_classes {
            let score = output[(4 + c) * num + i];
            if score > best {
                best = score;
                best_id = c;
            }
        }
        if best < CONF_THRESH {
            continue;
        }

        let mut x1 = x - w * 0.5;
        let mut y1 = y - h * 0.5;
        let mut x2 = x + w * 0.5;
        let mut y2 = y + h * 0.5;

        x1 = (x1 - lb.pad_x) / lb.scale;
        y1 = (y1 - lb.pad_y) / lb.scale;
        x2 = (x2 - lb.pad_x) / lb.scale;
        y2 = (y2 - lb.pad_y) / lb.scale;

        x1 = x1.clamp(0.0, (lb.in_w - 1) as f32);
        y1 = y1.clamp(0.0, (lb.in_h - 1) as f32);
        x2 = x2.clamp(0.0, (lb.in_w - 1) as f32);
        y2 = y2.clamp(0.0, (lb.in_h - 1) as f32);

        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        dets.push(RawDet {
            x1,
            y1,
            x2,
            y2,
            score: best,
            class_id: best_id as i64,
        });
    }

    Ok(nms(dets, 0.45))
}

fn nms(mut dets: Vec<RawDet>, iou_thresh: f32) -> Vec<RawDet> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept = Vec::new();
    while let Some(det) = dets.pop() {
        let mut keep = true;
        for k in &kept {
            if iou(&det, k) > iou_thresh {
                keep = false;
                break;
            }
        }
        if keep {
            kept.push(det);
        }
    }
    kept
}

fn iou(a: &RawDet, b: &RawDet) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (area_a + area_b - inter + 1e-6)
}
