use typ;

pub struct Descriptor {
	pub dims: Vec<usize>,
	pub strides: Vec<usize>,
	pub alignment: usize,
	pub ty: typ::Native,
}

impl Descriptor {
	pub fn new() -> Self {
		Descriptor{dims: vec![], strides: vec![], alignment: 0,
		           ty: typ::Native::Void}
	}

	// Returns the row-major packed strides for the current tensor.
	pub fn packed_strides(&self) -> Vec<usize> {
		let mut rv: Vec<usize> = vec![0; self.dims.len()];
		let mut prod = 1;
		for (idx, dim) in self.dims.iter().enumerate() {
			rv[idx] = prod;
			prod = prod*dim;
		}
		return rv;
	}

	// Returns the natural alignment for the given tensor.
	pub fn natural_alignment(&self) -> usize {
		assert_ne!(self.ty, typ::Native::Void);
		return self.ty.size();
	}

	// Returns the number of elements in this tensor.  Note that bytes for
	// padding are not included in this computation.
	pub fn len(&self) -> usize {
		self.dims.iter().product()
	}

	// Computes the number of bytes needed to represent this tensor.  Note this
	// includes both 1) any padding needed and 2) multiplies the type
	pub fn size_bytes(&self) -> usize {
		assert_ne!(self.ty, typ::Native::Void); // untyped data can't have a size.
		assert_eq!(self.dims.len(), self.strides.len());
		// need to find the index of the dimension with the max stride.
		let mut d = 0 as usize;
		let mut strd = 0;
		for (i, s) in self.strides.iter().enumerate() {
			if *s > strd {
				d = i;
				strd = *s;
			}
		}
		return self.dims[d]*self.strides[d]*self.ty.size();
	}
}
