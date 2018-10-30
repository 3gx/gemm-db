use std::os::raw;
use cuda_driver_sys::cuda::*;

macro_rules! cu {
	($call:expr) => ({
		let res: CUresult = $call;
		assert_eq!(res, CUresult::CUDA_SUCCESS);
	})
}

// Context-based CUDA API.
//
// We try to hide the implicit nature of the CUDA API.  Thus there is no
// implicit context, and no implicit stream.  Instead, a context must be given
// to allocate any resource.  To enforce this, the CUDA routines are wrapped as
// methods on this Context struct, and there is no other way to access them.
//
// Despite every API taking the context as an argument, we do not actually
// enforce activation of the context for each API call, for performance
// reasons.  It is up to the user, thereore, to explicitly make the context
// active.
/// @todo can we have explicit checks for that in a debug mode?
pub struct Context {
	ctx: CUcontext,
}
impl Context {
	pub fn new(device: usize) -> Self {
		let dev = device as CUdevice;
		let flags: raw::c_uint = CU_CTX_SCHED_AUTO as raw::c_uint;
		unsafe {
			let mut ctxt: CUcontext = 0 as CUcontext;
			cu!(cuInit(0));
			cu!(cuCtxCreate_v2(&mut ctxt, flags, dev));
			Context{ctx: ctxt}
		}
	}

	// allocates device memory of the given size.
	pub fn allocate(&self, sz: usize) -> Option<CUdeviceptr> {
		let mut rv: CUdeviceptr = 0 as CUdeviceptr;
		let res: CUresult = unsafe {
			cuMemAlloc_v2(&mut rv, sz)
		};
		if res != CUresult::CUDA_SUCCESS {
			error!("Error {:?} allocating {} bytes.", res, sz);
			return None;
		}
		return Some(rv);
	}
	// deallocates a chunk of memory previously obtained via 'alloc'.
	pub fn deallocate(&self, ptr: CUdeviceptr) -> Option<()> {
		let res: CUresult = unsafe {
			cuMemFree_v2(ptr)
		};
		if res != CUresult::CUDA_SUCCESS {
			error!("Error {:?} deallocating {:x}", res, ptr);
			return None;
		}
		return Some(());
	}
}
impl Drop for Context {
	fn drop(&mut self) {
		unsafe {
			cu!(cuCtxDestroy_v2(self.ctx));
			self.ctx = 0 as CUcontext;
		}
	}
}
