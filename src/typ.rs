// This file is crazily named because "type" is a keyword in rust: can't have a
// module with the same name.

// A Native type is a type that is builtin to the language.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Native {
	U8, U16, U32, U64,
	I8, I16, I32, I64,
	F32, F64,
	Boolean,
	Character,
	Void,
}

impl Native {
	// True if this type is "wider" than the given Native type.  Wider means that
	// it is always safe to assign a narrower-type to the wider-type, and almost
	// always unsafe to assign the other way.
	// These do not follow C rules, and are intended to be more restrictive and
	// force the user to cast.  Notable exceptions from C:
	//   - Characters are not integer types at all, and thus never "wider".
	//   - Floating point values are not integer types and cannot be wider or
	//     narrower than their integer counterparts.
	pub fn wider(&self, other: Native) -> bool {
		match self {
			&Native::U8 => false,
			&Native::U16 if other == Native::U8 => true,
			&Native::U32 if other == Native::U8 || other == Native::U16 => true,
			&Native::U64 if other == Native::U8 || other == Native::U16 ||
			                other == Native::U32 => true,
			&Native::I8 => false,
			&Native::I16 if other == Native::I8 => true,
			&Native::I32 if other == Native::I8 || other == Native::I16 => true,
			&Native::I64 if other == Native::I8 || other == Native::I16 ||
			                other == Native::I32 => true,
			&Native::F64 if other == Native::F32 => true,
			_ => false,
		}
	}

	/// Returns the number of bytes needed to represent an element of the type.
	pub fn size(&self) -> usize {
		match *self {
			Native::I8 | Native::U8 => 1,
			Native::I16 | Native::U16 => 2,
			Native::I32 | Native::U32 => 4,
			Native::I64 | Native::U64 => 8,
			Native::F32 => 4,
			Native::F64 => 8,
			Native::Boolean | Native::Character => 1,
			Native::Void => 0
		}
	}
}

pub trait RTTI {
	fn type_name(&self) -> String;
}
impl RTTI for i8 { fn type_name(&self) -> String { "i8".to_string() } }
impl RTTI for i16 { fn type_name(&self) -> String { "i16".to_string() } }
impl RTTI for i32 { fn type_name(&self) -> String { "i32".to_string() } }
impl RTTI for i64 { fn type_name(&self) -> String { "i64".to_string() } }
impl RTTI for u8 { fn type_name(&self) -> String { "u8".to_string() } }
impl RTTI for u16 { fn type_name(&self) -> String { "u16".to_string() } }
impl RTTI for u32 { fn type_name(&self) -> String { "u32".to_string() } }
impl RTTI for u64 { fn type_name(&self) -> String { "u64".to_string() } }
impl RTTI for usize { fn type_name(&self) -> String { "usize".to_string() } }
impl RTTI for f32 { fn type_name(&self) -> String { "f32".to_string() } }
impl RTTI for f64 { fn type_name(&self) -> String { "f64".to_string() } }
impl RTTI for bool { fn type_name(&self) -> String { "bool".to_string() } }
impl RTTI for char { fn type_name(&self) -> String { "char".to_string() } }
impl RTTI for Native {
	fn type_name(&self) -> String {
		match *self {
			Native::U8 => "u8".to_string(), Native::U16 => "u16".to_string(),
			Native::U32 => "u32".to_string(), Native::U64 => "u64".to_string(),
			Native::I8 => "u8".to_string(), Native::I16 => "u16".to_string(),
			Native::I32 => "u32".to_string(), Native::I64 => "u64".to_string(),
			Native::F32 => "f32".to_string(), Native::F64 => "f64".to_string(),
			Native::Boolean => "bool".to_string(),
			Native::Character => "char".to_string(),
			Native::Void => "void".to_string(),
		}
	}
}

pub trait Name {
	fn name(&self) -> String;
}

impl Name for Native {
	fn name(&self) -> String {
		match self {
			&Native::U8 => "uint8_t",
			&Native::I8 => "int8_t",
			&Native::U16 => "uint16_t", &Native::I16 => "int16_t",
			&Native::U32 => "uint32_t", &Native::I32 => "int32_t",
			&Native::U64 => "uint64_t", &Native::I64 => "int64_t",
			&Native::F32 => "float", &Native::F64 => "double",
			&Native::Boolean => "bool",
			&Native::Character => "char",
			&Native::Void => "void",
		}.to_string()
	}
}
