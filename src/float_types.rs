// Re-export parry and rapier for the appropriate float size
#[cfg(feature = "f64")]
pub use parry3d_f64 as parry3d;
#[cfg(feature = "f64")]
pub use rapier3d_f64 as rapier3d;

#[cfg(feature = "f32")]
pub use parry3d;
#[cfg(feature = "f32")]
pub use rapier3d;

// Our Real scalar type:
#[cfg(feature = "f32")]
pub type Real = f32;
#[cfg(feature = "f64")]
pub type Real = f64;

/// A small epsilon for geometric comparisons, adjusted per precision.
#[cfg(feature = "f32")]
pub const EPSILON: Real = 1e-5;
#[cfg(feature = "f64")]
pub const EPSILON: Real = 1e-12;

// Pi
#[cfg(feature = "f32")]
pub const PI: Real = core::f32::consts::PI;
#[cfg(feature = "f64")]
pub const PI: Real = core::f64::consts::PI;

// Tau
#[cfg(feature = "f32")]
pub const TAU: Real = core::f32::consts::TAU;
#[cfg(feature = "f64")]
pub const TAU: Real = core::f64::consts::TAU;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Unit conversion
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// ### Inch(`in`)
pub const INCH: Real = 25.4;
/// ### Foot(`ft`)
pub const FOOT: Real = 25.4 * 12.0;
/// ### Yard(`yd`)
pub const YARD: Real = 25.4 * 36.0;
/// ### Millimeter(`mm`)
pub const MM: Real = 1.0;
/// ### Centimeter(`cm`)
pub const CM: Real = 10.0;
/// ### Meter(`m`)
pub const METER: Real = 1000.0;
