use cu;
use cuda_driver_sys::*;
use tensor;
use typ;

#[derive(Clone, PartialEq, Eq)]
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

	pub fn evaluate(&self, _scratch: cuda::CUdeviceptr) -> f32 {
		// @todo fixme implement.
		return 0.0;
	}
}

pub fn costs(tactics: &Vec<Tactic>, ctx: &cu::Context) -> Vec<f32> {
	let mut rv: Vec<f32> = Vec::with_capacity(tactics.len());
	rv.resize(tactics.len(), 0.0);
	let scratch_size: usize = largest_scratch_needed(tactics);
	let scratch: cuda::CUdeviceptr = ctx.alloc(scratch_size).unwrap();
	for i in 0..tactics.len() {
		rv[i] = tactics[i].evaluate(scratch);
	}
	ctx.dealloc(scratch).unwrap();
	return rv;
}

// Computes the max scratch space we would need to run any tactic.  This
// includes space for all tensors, including any padding needed---both for the
// tensor itself as well as the alignment.
pub fn largest_scratch_needed(tactics: &Vec<Tactic>) -> usize {
	tactics.iter().fold(0, |cur, tac| cur.max(scratch_needed(tac)))
}

// returns the next aligned value beyond 'addr', which might be addr itself if
// addr is perfectly aligned already.
fn next_align(addr: usize, align: usize) -> usize {
	assert!(align > 0);
	let align_signed: isize = align as isize;
	let addr_signed: isize = addr as isize;
	let rv = (addr_signed + (align_signed-1)) & -align_signed;
	return rv as usize;
}

fn scratch_needed(tactic: &Tactic) -> usize {
	let mut sum: usize = 0;
	for inp in tactic.inputs.iter() {
		sum = next_align(sum, inp.alignment);
		sum += inp.size_bytes();
	}
	for outp in tactic.output.iter() {
		sum = next_align(sum, outp.alignment);
		sum += outp.size_bytes();
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
		a.dims = vec![256,128]; // KxM
		b.dims = vec![384,256]; // NxK
		c.dims = vec![b.dims[0], a.dims[1]]; // NxM matrix
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
		assert_eq!(a.dims[0], b.dims[1]);
		assert_eq!(a.dims[1], c.dims[1]);
		assert_eq!(b.dims[0], c.dims[0]);
		tact.m = a.dims[0];
		tact.n = b.dims[1];
		tact.k = a.dims[1];
		tact.lda = a.strides[1];
		tact.ldb = b.strides[1];
		tact.ldc = c.strides[1];
		assert_eq!(a.strides[1], a.dims[0]); // because packed.
		assert_eq!(b.strides[1], b.dims[0]); // because packed.
		assert_eq!(c.strides[1], c.dims[0]); // because packed.
		tact.compute_type = typ::Native::F32;
		let needed = a.dims[0]*a.dims[1]*a.ty.size() +
		             b.dims[0]*b.dims[1]*b.ty.size() +
		             c.dims[0]*c.dims[1]*c.ty.size();
		tact.inputs = [a,b];
		tact.output = [c];
		assert_eq!(needed, scratch_needed(&tact));
	}


	fn packed_tensor(dims: &Vec<usize>, ty: &typ::Native) -> tensor::Descriptor {
		let mut rv = tensor::Descriptor::new();
		rv.dims = dims.clone();
		rv.ty = ty.clone();
		rv.strides = rv.packed_strides();
		rv.alignment = rv.natural_alignment();
		rv
	}
	fn tactic(inputs: &Vec<tensor::Descriptor>,
	          outputs: &Vec<tensor::Descriptor>) -> Tactic {
		for inp in inputs { // make sure all tensors are the same type.
			for outp in outputs {
				assert_eq!(inp.ty, outp.ty);
			}
		}
		let mut tac = Tactic::new();
		tac.m = inputs[0].dims[0];
		tac.n = inputs[1].dims[1];
		tac.k = inputs[0].dims[1];
		assert_eq!(inputs[0].dims[0], outputs[0].dims[0]); // M
		assert_eq!(inputs[1].dims[1], outputs[0].dims[1]); // N
		assert_eq!(inputs[0].dims[1], inputs[1].dims[0]); // K
		tac.lda = inputs[0].strides[1];
		tac.ldb = inputs[1].strides[1];
		tac.ldc = outputs[0].strides[1];
		tac.compute_type = inputs[0].ty;
		tac.inputs = [inputs[0].clone(),inputs[1].clone()];
		tac.output = [outputs[0].clone()];
		tac
	}

	#[test]
	fn scratch_needed_unaligned_packed_f16() {
		let a = packed_tensor(&vec![16,24], &typ::Native::F16);
		let b = packed_tensor(&vec![24,48], &typ::Native::F16);
		let c = packed_tensor(&vec![16,48], &typ::Native::F16);
		let tact: Tactic = tactic(&vec![a.clone(),b.clone()], &vec![c.clone()]);
		let needed = a.dims[0]*a.dims[1]*a.ty.size() +
		             b.dims[0]*b.dims[1]*b.ty.size() +
		             c.dims[0]*c.dims[1]*c.ty.size();
		assert_eq!(needed, scratch_needed(&tact));
	}

	#[test]
	fn scratch_needed_unpacked_i32() {
		let mut a = packed_tensor(&vec![21,37], &typ::Native::I32);
		let mut b = packed_tensor(&vec![37,94], &typ::Native::I32);
		let mut c = packed_tensor(&vec![21,94], &typ::Native::I32);
		// ... now unpack it.
		a.strides = vec![1, 32];
		b.strides = vec![1, 48];
		c.strides = vec![1, 32];
		let tact: Tactic = tactic(&vec![a.clone(),b.clone()], &vec![c.clone()]);
		let needed = a.size_bytes() + b.size_bytes() + c.size_bytes();
		assert_eq!(needed, scratch_needed(&tact));
	}

	#[test]
	fn scratch_needed_unaligned_unpacked_f16() {
		let mut a = packed_tensor(&vec![129,145], &typ::Native::F16);
		let mut b = packed_tensor(&vec![145,47], &typ::Native::F16);
		let mut c = packed_tensor(&vec![129,47], &typ::Native::F16);
		a.strides = vec![1, 132];
		b.strides = vec![1, 150];
		c.strides = vec![1, 140];
		a.alignment = 16;
		b.alignment = 16;
		c.alignment = 16;
		let mut needed = a.size_bytes();
		needed = next_align(needed, b.alignment);
		needed += b.size_bytes();
		needed = next_align(needed, c.alignment);
		needed += c.size_bytes();
		let tact = tactic(&vec![a.clone(),b.clone()], &vec![c.clone()]);
		assert_eq!(needed, scratch_needed(&tact));
	}
}
