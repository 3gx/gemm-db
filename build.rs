extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
	let cuda_path = PathBuf::from(match env::var("CUDA_PATH") {
		Ok(chome) => chome,
		Err(_) => "/usr/local/cuda".to_string()
	});
	let cuda = match cuda_path.to_str() {
		Some(c) => c,
		None => "blas: error creating string from cuda path",
	};
	for libdir in vec!["lib64", "lib"] {
		let mut clib_path = cuda_path.clone();
		clib_path.push(libdir);
		// Don't check if the path exists first.  If someone is having issues and
		// turns verbosity on, this will at least clue them in how to hack it.
		println!("cargo:rustc-link-search=native={}/{}", cuda, libdir);
	}
	println!("cargo:rustc-link-lib=cublas"); // link against cublas

	// The general rules on what to whitelist:
	//   - never anything from the runtime API,
	//   - nothing concerning implicit state (i.e. no CU_STREAM_PER_THREAD),
	//   - debatably, nothing that is marked deprecated,
	//   - unquestioningly, nothing that is deprecated in CUDA 10.0 or earlier.
	let bindings = bindgen::Builder::default()
			// Tell clang where to find cuda.h.
			.clang_arg(format!("-I{}/include", cuda))
			.header("blas.h")
			.layout_tests(false)
			// prepend_enum_name and constified_enum together make it so that the
			// bindings create a constant for e.g. "CU_CTX_SCHED_AUTO", instead of
			// something that needs to be qualified (a la
			// "CUctx_flags::CU_CTX_SCHED_AUTO").
			.prepend_enum_name(false)
			.constified_enum("cublasGemmAlgo_t")
			.constified_enum("cublasOperation_t")
			.constified_enum("cublasStatus_t")
			.constified_enum("cudaDataType_t")
			.whitelist_recursively(false)
			.whitelisted_type("cudaError_enum")
			.whitelisted_type("CU[A-Za-z0-9_]+_enum")
			.whitelisted_type("CU[A-Za-z0-9_]+_st")
			// Keep these alphabetized.
			.whitelisted_type("cublasContext")
			.whitelisted_type("cublasGemmAlgo_t")
			.whitelisted_type("cublasHandle_t")
			.whitelisted_type("cublasOperation_t")
			.whitelisted_type("cublasStatus_t")
			.whitelisted_type("cudaDataType")
			.whitelisted_type("cudaDataType_t")
			// Keep these alphabetized.
			.whitelisted_function("cublasCreate_v2")
			.whitelisted_function("cublasDestroy_v2")
			.whitelisted_function("cublasGemmEx")
			.generate()
			.expect("Unable to generate blas bindings");
	
	let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
	bindings
		.write_to_file(out_path.join("blas.rs"))
		.expect("Couldn't write blas bindings!");

	println!("cargo:rerun-if-changed=build.rs");
}
