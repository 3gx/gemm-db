#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/blas.rs"));

#[cfg(test)]
mod test {
	#![deny(dead_code)]
	use std::os::raw;
	use cu;
	use cuda_driver_sys::*;
	use super::*;

	macro_rules! blas {
		($call:expr) => (unsafe{
			let res: cublasStatus_t = $call;
			assert_eq!(res, CUBLAS_STATUS_SUCCESS);
		})
	}

	#[test]
	fn simple_gemm() {
		let M = 64 as usize;
		let N = 128 as usize;
		let K = 256 as usize;
		let lda = 64;
		let ldb = 256;
		let ldc = 64;
		// how many bytes are we going to need to hold all three matrices?
		// For this we assume packed data w/o special alignment criterion.
		let a_size = M*K * ::std::mem::size_of::<f32>();
		let b_size = K*N * ::std::mem::size_of::<f32>();
		let c_size = M*N * ::std::mem::size_of::<f32>();
		let required = a_size + b_size + c_size;
		let ctx = cu::Context::new(0);
		let mem: cuda::CUdeviceptr = ctx.allocate(required).unwrap();
		let A: cuda::CUdeviceptr = mem;
		let B: cuda::CUdeviceptr = mem + a_size as u64;
		let C: cuda::CUdeviceptr = mem + a_size as u64 + b_size as u64;
		let mut hdl: cublasHandle_t = 0 as cublasHandle_t;
		blas!(cublasCreate_v2(&mut hdl));
		let alpha: f32 = 1.0;
		let beta: f32 = 0.0;
		let compute_type = CUDA_R_32F;
		blas!(cublasGemmEx(
			hdl, CUBLAS_OP_N, CUBLAS_OP_N, M as i32, N as i32, K as i32,
			&alpha as *const _ as *const raw::c_void,
			A as *const raw::c_void, CUDA_R_32F, lda,
			B as *const raw::c_void, CUDA_R_32F, ldb,
			&beta as *const _ as *const raw::c_void,
			C as *mut raw::c_void, CUDA_R_32F, ldc,
			compute_type, CUBLAS_GEMM_DEFAULT
		));
		blas!(cublasDestroy_v2(hdl));
		ctx.deallocate(mem);
	}
}
