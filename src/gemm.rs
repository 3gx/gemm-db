use tensor;
use typ;
use cuda_driver_sys::*;

pub struct Tactic {
	m: usize, n: usize, k: usize,
	lda: usize, ldb: usize, ldc: usize,
	compute_type: typ::Native,
	// A more generic implementation would use vectors for inputs and outputs,
	// but this is for a GEMM.  It'll have one input and one output, always.
	inputs: [tensor::Descriptor;2],
	output: [tensor::Descriptor;1],
}

impl Tactic {
	pub fn new() -> Self {
		Tactic{m: 0, n: 0, k: 0, lda: 0, ldb: 0, ldc: 0, compute_type:
		       typ::Native::F32,
		       inputs: [tensor::Descriptor::new(), tensor::Descriptor::new()],
		       output: [tensor::Descriptor::new()]}
	}

	pub fn evaluate(&self, scratch: cuda::CUdeviceptr) -> f32 {
		return 0.0;
	}
}

// Computes the max scratch space we would need to run any tactic.  This
// includes space for all tensors, including any padding needed---both for the
// tensor itself as well as the alignment.
pub fn largest_scratch_needed(tactics: &Vec<Tactic>) -> usize {
	tactics.iter().fold(0, |cur, tac| cur.max(scratch_needed(tac)))
}

// returns the next aligned value beyond 'addr', which might be addr itself if
// addr is perfectly aligned already.
pub fn next_align(addr: usize, align: usize) -> usize {
	assert!(align > 0);
	let align_signed: isize = align as isize;
	let addr_signed: isize = addr as isize;
	let rv = (addr_signed + (align_signed-1)) & -align_signed;
	return rv as usize;
}

pub fn scratch_needed(tactic: &Tactic) -> usize {
	let mut sum: usize = 0;
	for inp in tactic.inputs.iter() {
		sum += inp.size_bytes();
		sum = next_align(sum, 16);
	}
	for outp in tactic.output.iter() {
		sum += outp.size_bytes();
		sum = next_align(sum, 16);
	}
	return sum;
}

#[cfg(test)]
mod test {
	use super::*;
	use tensor;
	use typ;

	#[test]
	fn scratch_gemm_unaligned_packed() {
		let mut a = tensor::Descriptor::new();
		let mut b = tensor::Descriptor::new();
		let mut c = tensor::Descriptor::new();
		a.dims = vec![128,256];
		b.dims = vec![256,512];
		c.dims = vec![b.dims[0], a.dims[1]];
		a.ty = typ::Native::F32;
		b.ty = typ::Native::F32;
		c.ty = typ::Native::F32;
		a.strides = a.packed_strides();
		b.strides = b.packed_strides();
		c.strides = c.packed_strides();
		a.alignment = a.natural_alignment();
		b.alignment = b.natural_alignment();
		c.alignment = c.natural_alignment();
		let mut tact = Tactic::new();
		tact.m = a.dims[0];
		tact.n = a.dims[1];
		tact.k = b.dims[1];
		tact.lda = a.ty.size(); // natural alignment.
		tact.ldb = b.ty.size();
		tact.ldc = c.ty.size();
		tact.compute_type = typ::Native::F32;
		let needed = a.dims[0]*a.dims[1]*a.ty.size() +
		             b.dims[0]*b.dims[1]*b.ty.size() +
		             c.dims[0]*c.dims[1]*c.ty.size();
		tact.inputs = [a,b];
		tact.output = [c];
		assert_eq!(needed, scratch_needed(&tact));
	}
}
