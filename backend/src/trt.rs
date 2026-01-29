#![cfg(feature = "trt_engine")]

use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

#[repr(C)]
struct TrtRunnerOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn trt_create(path: *const c_char) -> *mut TrtRunnerOpaque;
    fn trt_destroy(runner: *mut TrtRunnerOpaque);
    fn trt_get_input_dims(runner: *mut TrtRunnerOpaque, n: *mut c_int, c: *mut c_int, h: *mut c_int, w: *mut c_int) -> c_int;
    fn trt_get_output_dims(runner: *mut TrtRunnerOpaque, n: *mut c_int, c: *mut c_int, h: *mut c_int, w: *mut c_int) -> c_int;
    fn trt_get_output_count(runner: *mut TrtRunnerOpaque) -> usize;
    fn trt_infer(runner: *mut TrtRunnerOpaque, input: *const f32, output: *mut f32) -> c_int;
    fn trt_last_error() -> *const c_char;
    fn trt_build_engine(onnx: *const c_char, engine: *const c_char) -> c_int;
}

pub struct TrtRunner {
    ptr: *mut TrtRunnerOpaque,
}

impl TrtRunner {
    pub fn build_engine(onnx_path: &str, engine_path: &str) -> Result<()> {
        let c_onnx = CString::new(onnx_path)?;
        let c_engine = CString::new(engine_path)?;
        let res = unsafe { trt_build_engine(c_onnx.as_ptr(), c_engine.as_ptr()) };
        if res != 0 {
            return Err(anyhow!(last_error()));
        }
        Ok(())
    }
    pub fn new(path: &str) -> Result<Self> {
        let c_path = CString::new(path)?;
        let ptr = unsafe { trt_create(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err(anyhow!(last_error()));
        }
        Ok(Self { ptr })
    }

    pub fn input_dims(&self) -> Result<(i32, i32, i32, i32)> {
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        let res = unsafe { trt_get_input_dims(self.ptr, &mut n, &mut c, &mut h, &mut w) };
        if res != 0 {
            return Err(anyhow!(last_error()));
        }
        Ok((n, c, h, w))
    }

    pub fn output_dims(&self) -> Result<(i32, i32, i32, i32)> {
        let mut n = 0;
        let mut c = 0;
        let mut h = 0;
        let mut w = 0;
        let res = unsafe { trt_get_output_dims(self.ptr, &mut n, &mut c, &mut h, &mut w) };
        if res != 0 {
            return Err(anyhow!(last_error()));
        }
        Ok((n, c, h, w))
    }

    pub fn output_count(&self) -> usize {
        unsafe { trt_get_output_count(self.ptr) }
    }

    pub fn infer(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let res = unsafe { trt_infer(self.ptr, input.as_ptr(), output.as_mut_ptr()) };
        if res != 0 {
            return Err(anyhow!(last_error()));
        }
        Ok(())
    }
}

impl Drop for TrtRunner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { trt_destroy(self.ptr) };
        }
    }
}

fn last_error() -> String {
    unsafe {
        let ptr = trt_last_error();
        if ptr.is_null() {
            "TensorRT error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}
