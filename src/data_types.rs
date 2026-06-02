//! Data types

use std::ops::{Index};

// ---- Enums ------------
pub enum FIRFilterMode {
    LowPass {cutoff: f64, sample_rate: f64},
    Highpass { cutoff: f64, sample_rate: f64 },
    Bandpass { low_cutoff: f64, high_cutoff: f64, sample_rate: f64 },
    Notch { cutoff: f64, sample_rate: f64 },
    Hilbert,
    Differentiator,
    Raw,
}

// ---- Structs ------------
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
