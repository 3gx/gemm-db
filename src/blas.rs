#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/blas.rs"));

use std::os::raw;
use cuda_driver_sys::*;
use cu;

#[derive(Debug)]
pub enum BlasError {
	NotInitialized = CUBLAS_STATUS_NOT_INITIALIZED as isize,
	Allocation = CUBLAS_STATUS_ALLOC_FAILED as isize,
	InvalidValue = CUBLAS_STATUS_INVALID_VALUE as isize,
	Architecture= CUBLAS_STATUS_ARCH_MISMATCH as isize,
	Mapping = CUBLAS_STATUS_MAPPING_ERROR as isize,
	Execution = CUBLAS_STATUS_EXECUTION_FAILED as isize,
	Internal = CUBLAS_STATUS_INTERNAL_ERROR as isize,
	Unsupported = CUBLAS_STATUS_NOT_SUPPORTED as isize,
	License = CUBLAS_STATUS_LICENSE_ERROR as isize,
	Unknown, /* if we somehow get an undefined status code */
}

fn cublas_error(e: u32) -> BlasError {
	match e {
		CUBLAS_STATUS_NOT_INITIALIZED => BlasError::NotInitialized,
		CUBLAS_STATUS_ALLOC_FAILED => BlasError::Allocation,
		CUBLAS_STATUS_INVALID_VALUE => BlasError::InvalidValue,
		CUBLAS_STATUS_ARCH_MISMATCH => BlasError::Architecture,
		CUBLAS_STATUS_MAPPING_ERROR => BlasError::Mapping,
		CUBLAS_STATUS_EXECUTION_FAILED => BlasError::Execution,
		CUBLAS_STATUS_INTERNAL_ERROR => BlasError::Internal,
		CUBLAS_STATUS_NOT_SUPPORTED => BlasError::Unsupported,
		CUBLAS_STATUS_LICENSE_ERROR => BlasError::License,
		_ => BlasError::Unknown,
	}
}

// A BLAS handle; all BLAS APIs hang off the handle.
pub struct Blas<'a> {
	hdl: cublasHandle_t,
	ctx: &'a cu::Context,
}

impl<'a> Blas<'a> {
	// A cuBLAS handle requires a cu::Context, which must live longer than the
	// handle itself.  Unlike cuBLAS' C API, the device to run on cannot be
	// dynamically reconfigured: if clients want to run on N devices, they must
	// create N contexts and associate N handles with them.
	fn new(ctx: &'a cu::Context) -> Result<Self, BlasError> {
		let mut hdl = 0 as cublasHandle_t;
		let res = unsafe { cublasCreate_v2(&mut hdl) };
		if res != CUBLAS_STATUS_SUCCESS {
			return Err(cublas_error(res))
		}
		return Ok(Blas{hdl: hdl, ctx: ctx});
	}

	// Basically drop, but can return an error.  This should be preferred over
	// a normal drop, because it allows error handling.
	// It is always safe to drop a destroy()ed Blas.
	fn destroy(&mut self) -> Result<(), BlasError> {
		if self.hdl == 0 as cublasHandle_t {
			return Ok(());
		}
		let res = unsafe { cublasDestroy_v2(self.hdl) };
		self.hdl = 0 as cublasHandle_t;
		if res != CUBLAS_STATUS_SUCCESS {
			return Err(cublas_error(res));
		}
		return Ok(());
	}

	/// @todo add type parameter.  for now assuming f32.
	/// @todo alpha + beta parameters, or add an overload w/ those?
	fn gemm(&self, C: cuda::CUdeviceptr, ldc: usize,
					m: usize, n: usize, k: usize,
					A: cuda::CUdeviceptr, lda: usize,
					B: cuda::CUdeviceptr, ldb: usize) -> Result<(), BlasError> {
		let alpha: f32 = 1.0;
		let beta: f32 = 0.0;
		let compute_type = CUDA_R_32F;
		let rslt = unsafe {
			cublasGemmEx(
				self.hdl, CUBLAS_OP_N, CUBLAS_OP_N, m as i32, n as i32, k as i32,
				&alpha as *const _ as *const raw::c_void,
				A as *const raw::c_void, CUDA_R_32F, lda as i32,
				B as *const raw::c_void, CUDA_R_32F, ldb as i32,
				&beta as *const _ as *const raw::c_void,
				C as *mut raw::c_void, CUDA_R_32F, ldc as i32,
				compute_type, CUBLAS_GEMM_DEFAULT
			)
		};
		if rslt != CUBLAS_STATUS_SUCCESS {
			return Err(cublas_error(rslt));
		}
		return Ok(());
	}
}

// Drop is uncomfortable, because cublasDestroy can technically fail.  Ideally
// one would call 'destroy' first.
impl<'a> Drop for Blas<'a> {
	fn drop(&mut self) {
		self.destroy().unwrap();
	}
}

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

	// simplest GEMM test that uses the wrapper API.
	#[test]
	fn gemm_wrapper() {
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
		let cublas = Blas::new(&ctx).unwrap();
		cublas.gemm(C, ldc, M,N,K, A,lda, B,ldb).unwrap();
	}
}
