//! Data types

use std::{f64::consts::PI, ops::Index};

// ---- Enums ------------
#[derive(Debug, Clone, Copy, Default)]
pub enum BiquadFilterMode {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Bell,
    Lowshelf,
    Highshelf,
    #[default]
    Custom,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum FIRFilterMode {
    Lowpass {cutoff: f64, sample_rate: f64},
    Highpass { cutoff: f64, sample_rate: f64 },
    Bandpass { low_cutoff: f64, high_cutoff: f64, sample_rate: f64 },
    Notch { cutoff: f64, sample_rate: f64 },
    Hilbert,
    Differentiator,
    #[default]
    Custom,
}

pub enum WindowFunction {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    FlatTop,
}

impl WindowFunction {
    pub fn apply(&self, length: usize, index: usize) -> f64 {
        match self {
            WindowFunction::Rectangular => 1.0,
            WindowFunction::Hann => Self::hann(length, index),
            WindowFunction::Hamming => Self::hamming(length, index),
            WindowFunction::Blackman => Self::blackman(length, index),
            WindowFunction::BlackmanHarris => Self::blackman_harris(length, index),
            WindowFunction::FlatTop => Self::flat_top(length, index),
        }
    }

    fn hann(length: usize, index: usize) -> f64 {
        let denominator = length as f64 - 1.0;
        let numerator = 2.0 * PI * index as f64;

        0.5 * (1.0 - (numerator / denominator).cos())
    }

    fn hamming(length: usize, index: usize) -> f64 {
        let denominator = length as f64 - 1.0;
        let numerator = 2.0 * PI * index as f64;
        let cosine = (numerator / denominator).cos();

        0.54 - 0.46 * cosine
    }

    fn blackman(length: usize, index: usize) -> f64 {
        let denominator = length as f64 - 1.0;
        let t = 2.0 * PI * index as f64 / denominator;

        0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
    }

    fn blackman_harris(length: usize, index: usize) -> f64 {
        let denominator = length as f64 - 1.0;
        let t = 2.0 * PI * index as f64 / denominator;

        0.35875 - 0.48829 * t.cos() + 0.14128 * (2.0 * t).cos() - 0.01168 * (3.0 * t).cos()
    }

    fn flat_top(length: usize, index: usize) -> f64 {
        let denominator = length as f64 - 1.0;
        let t = 2.0 * PI * index as f64 / denominator;

        0.21557895 - 0.41663158 * t.cos() + 0.27726316 * (2.0 * t).cos()
            - 0.08357895 * (3.0 * t).cos() + 0.00694737 * (4.0 * t).cos()
    }
}

// ---- Structs ------------
#[derive(Debug, Clone, Copy)]
/// Output from a single `StateVariableFilter` pass — all three responses at once
pub struct SvfOutput<T> {
    pub low: T,
    pub high: T,
    pub band: T,
}

#[derive(Debug, Clone)]
struct SlidingHead {
    size: usize,
    position: usize,
}

impl SlidingHead {
    // Init
    fn with_capacity(capacity: usize) -> Self {
        Self {
            size: capacity,
            position: 0,
        }
    }

    fn increment_head(&mut self, amount: usize) {
        if self.size > 0 {
            self.position = (self.position + amount) % self.size;
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlidingWindow<T> {
    head: SlidingHead,
    buffer: Vec<T>,
}

impl<T> Index<usize> for SlidingWindow<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let len = self.buffer.len();
        assert!(index < len, "index out of bounds");

        let is_full = len == self.head.size;
        if is_full {
            &self.buffer[(self.head.position + index) % self.head.size]
        } else {
            &self.buffer[index]
        }
    }
}

impl<T> SlidingWindow<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = if capacity == 0 { 1 } else { capacity };

        Self {
            head: SlidingHead::with_capacity(capacity),
            buffer: Vec::with_capacity(capacity),
        }
    }

    pub fn slide(&mut self, value: T) {
        if self.buffer.len() < self.head.size {
            self.buffer.push(value);
        } else {
            self.buffer[self.head.position] = value;
        }
        self.head.increment_head(1);
    }


    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn as_slices(&self) -> (&[T], &[T]) {
        let (left, right) = self.buffer.split_at(self.head.position);
        (left, right)
    }
}

impl<T> SlidingWindow<T>
where
    T: Clone,
{
    pub fn resize(&mut self, new_capacity: usize, value: T) {
        self.buffer.rotate_left(self.head.position);
        self.head.position = 0;

        if new_capacity < self.buffer.len() {
            let drop = self.buffer.len() - new_capacity;
            self.buffer.drain(..drop);
        } else {
            self.buffer.resize(new_capacity, value);
        }

        self.head.size = new_capacity;
    }
}
