//#![allow(dead_code)]
#![forbid(unsafe_code, unused)]

pub mod bsp;
pub mod csg;
pub mod errors;
pub mod float_types;
pub mod plane;
pub mod polygon;
pub mod vertex;

#[cfg(test)]
mod tests;
