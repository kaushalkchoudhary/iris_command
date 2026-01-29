use std::env;

fn main() {
    if env::var("CARGO_FEATURE_TRT_ENGINE").is_err() {
        return;
    }

    println!("cargo:rerun-if-changed=trt_wrapper/trt_wrapper.cpp");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let mut build = cc::Build::new();
    build.cpp(true).file("trt_wrapper/trt_wrapper.cpp");
    build.flag_if_supported("-std=c++17");

    if let Ok(inc) = env::var("TENSORRT_INCLUDE_DIR") {
        build.include(inc);
    } else {
        build.include("/usr/include/x86_64-linux-gnu");
        build.include("/usr/include");
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        build.include(format!("{cuda_home}/include"));
    } else {
        build.include("/usr/local/cuda/include");
    }

    build.compile("trt_wrapper");

    if let Ok(lib) = env::var("TENSORRT_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib}");
    } else {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }

    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }

    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvinfer_plugin");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=cudart");
}
