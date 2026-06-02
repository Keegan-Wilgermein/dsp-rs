//! Filters

// ---- Imports ------------
use std::{f64::consts::PI, ops::{Add, Div, Mul, Sub}};
use crate::data_types::{BiquadFilterMode, FIRFilterMode, SlidingWindow, WindowFunction};

// ---- Structs -------------
#[derive(Debug, Clone)]
/// # BiquadFilter
/// A second order Infinite Impulse Response (IIR) filter
/// 
/// Call differente update functions to change the filters effect
pub struct BiquadFilter<T> {
    // Input coefficients
    /// Weight multiplyed with the current input sample
    b0: T,
    /// Weight multiplied with the previous input sample - `x1`
    b1: T,
    /// Weight multiplied with the second previous input sample - `x2`
    b2: T,

    // Output coefficients
    /// Weight multiplied with the previous output sample - `y1`
    a1: T,
    /// Weight multipled with the second previous output sample - `y2`
    a2: T,
    
    // Previous input samples
    /// First previous input sample
    x1: T,
    /// Second previous input sample
    x2: T,
    
    // Previous output samples
    /// Previous output sample
    y1: T,
    /// Second previous output sample
    y2: T,

    // Other
    /// The current mode
    mode: BiquadFilterMode,
}

impl<T> Default for BiquadFilter<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            a1: T::default(),
            a2: T::default(),
            b0: T::default(),
            b1: T::default(),
            b2: T::default(),
            x1: T::default(),
            x2: T::default(),
            y1: T::default(),
            y2: T::default(),
            mode: BiquadFilterMode::default(),
        }
    }
}

impl<T> BiquadFilter<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    f64: Into<T>,
{
    // Getters
    /// Gets `self.a1`
    pub fn get_a1(&self) -> T {
        self.a1
    }

    /// Gets `self.a2`
    pub fn get_a2(&self) -> T {
        self.a2
    }

    /// Gets `self.b0`
    pub fn get_b0(&self) -> T {
        self.b0
    }

    /// Gets `self.b1`
    pub fn get_b1(&self) -> T {
        self.b1
    }

    /// Gets `self.b2`
    pub fn get_b2(&self) -> T {
        self.b2
    }

    /// Gets `self.x1`
    pub fn get_x1(&self) -> T {
        self.x1
    }

    /// Gets `self.x2`
    pub fn get_x2(&self) -> T {
        self.x2
    }

    /// Gets `self.y1`
    pub fn get_y1(&self) -> T {
        self.y1
    }

    /// Gets `self.y2`
    pub fn get_y2(&self) -> T {
        self.y2
    }

    /// Gets the current mode of the `BiquadFilter`
    pub fn get_mode(&self) -> BiquadFilterMode {
        self.mode
    }

    // Setters
    /// Updates coefficients with provided values
    pub fn update_raw(
        &mut self,
        a1: T,
        a2: T,
        b0: T,
        b1: T,
        b2: T,
    ) {
        self.a1 = a1;
        self.a2 = a2;
        self.b0 = b0;
        self.b1 = b1;
        self.b2 = b2;

        self.mode = BiquadFilterMode::Custom;
    }

    /// Updates coefficients to represent a lowpass filter
    pub fn update_lowpass<V>(
        &mut self,
        cutoff: V,
        q: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let (alpha, omega) = Self::alpha_omega(
            sample_rate.into(),
            cutoff.into(),
            q.into(),
        );
        let omega_cos = omega.cos();
        let a0 = (1.0 + alpha).into();

        // Calculate a coefficients
        let (a1, a2) = Self::pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1.into() / a0;
        self.a2 = a2.into() / a0;

        // Calculate b coefficients
        self.b0 = ((1.0 - omega_cos) / 2.0).into() / a0;
        self.b1 = (1.0 - omega_cos).into() / a0;
        self.b2 = self.b0;

        self.mode = BiquadFilterMode::Lowpass;
    }

    /// Updates coefficients to represent a highpass filter
    pub fn update_highpass<V>(
        &mut self,
        cutoff: V,
        q: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let (alpha, omega) = Self::alpha_omega(
            sample_rate.into(),
            cutoff.into(),
            q.into(),
        );
        let omega_cos = omega.cos();
        let a0 = (1.0 + alpha).into();

        // Calculate a coefficients
        let (a1, a2) = Self::pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1.into() / a0;
        self.a2 = a2.into() / a0;

        // Calculate b coefficients
        self.b0 = ((1.0 + omega_cos) / 2.0).into() / a0;
        self.b1 = (-(1.0 + omega_cos)).into() / a0;
        self.b2 = self.b0;

        self.mode = BiquadFilterMode::Highpass;
    }

    /// Updates coefficients to represent a bandpass filter
    pub fn update_bandpass<V>(
        &mut self,
        cutoff: V,
        q: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let (alpha, omega) = Self::alpha_omega(
            sample_rate.into(),
            cutoff.into(),
            q.into(),
        );
        let a0 = (1.0 + alpha).into();

        // Calculate a coefficients
        let (a1, a2) = Self::pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1.into() / a0;
        self.a2 = a2.into() / a0;

        // Calculate b coefficients
        self.b0 = alpha.into() / a0;
        self.b1 = 0.0_f64.into();
        self.b2 = (-alpha).into() / a0;

        self.mode = BiquadFilterMode::Bandpass;
    }

    /// Updates coefficients to represent a notch filter
    pub fn update_notch<V>(
        &mut self,
        cutoff: V,
        q: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let (alpha, omega) = Self::alpha_omega(
            sample_rate.into(),
            cutoff.into(),
            q.into(),
        );
        let a0 = (1.0 + alpha).into();

        // Calculate a coefficients
        let (a1, a2) = Self::pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1.into() / a0;
        self.a2 = a2.into() / a0;

        // Calculate b coefficients
        self.b0 = 1.0_f64.into() / a0;
        self.b1 = (-(2.0 * omega.cos())).into() / a0;
        self.b2 = 1.0_f64.into() / a0;

        self.mode = BiquadFilterMode::Notch;
    }

    /// Updates coefficients to represent a bell filter
    pub fn update_bell<V>(
        &mut self,
        cutoff: V,
        q: V,
        gain: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let amplitude = decibel_to_amplitude(gain.into());
        let (alpha, omega) = Self::alpha_omega(
            sample_rate.into(),
            cutoff.into(),
            q.into(),
        );
        let omega_cos = omega.cos();
        let a0 = (1.0 + alpha / amplitude).into();

        // Calculate a coefficients
        self.a1 = (-(2.0 * omega_cos)).into() / a0;
        self.a2 = ((1.0 - alpha / amplitude)).into() / a0;

        // Calculate b coefficients
        self.b0 = ((1.0 + alpha * amplitude)).into() / a0;
        self.b1 = (-(2.0 * omega_cos)).into() / a0;
        self.b2 = ((1.0 - alpha * amplitude)).into() / a0;

        self.mode = BiquadFilterMode::Bell;
    }

    /// Updates coefficients to represent a lowshelf filter
    pub fn update_lowshelf<V>(
        &mut self,
        cutoff: V,
        gain: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let gain: f64 = gain.into();
        let sample_rate: f64 = sample_rate.into();

        let amplitude = decibel_to_amplitude(gain);
        let amp_sqrt = amplitude.sqrt();

        let omega = (2.0 * PI * cutoff) / sample_rate;
        let omega_cos = omega.cos();
        let alpha = omega.sin() / 2.0_f64.sqrt();

        let a0 = ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha).into();

        // Calculate a coefficients
        self.a1 = (-(2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * omega_cos))).into() / a0;
        self.a2 = ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha).into() / a0;

        // Calculate b coefficients
        self.b0 = (amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha)).into() / a0;
        self.b1 = (2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * omega_cos)).into() / a0;
        self.b2 = (amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha)).into() / a0;

        self.mode = BiquadFilterMode::Lowshelf;
    }

    /// Updates coefficients to represent a highshelf filter
    pub fn update_highshelf<V>(
        &mut self,
        cutoff: V,
        gain: V,
        sample_rate: V,
    )
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let gain: f64 = gain.into();
        let sample_rate: f64 = sample_rate.into();

        let amplitude = decibel_to_amplitude(gain);
        let amp_sqrt = amplitude.sqrt();

        let omega = (2.0 * PI * cutoff) / sample_rate;
        let omega_cos = omega.cos();
        let alpha = omega.sin() / 2.0_f64.sqrt();

        let a0 = ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha).into();

        // Calculate a coefficients
        self.a1 = (2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * omega_cos)).into() / a0;
        self.a2 = ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha).into() / a0;

        // Calculate b coefficients
        self.b0 = (amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha)).into() / a0;
        self.b1 = (-(2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * omega_cos))).into() / a0;
        self.b2 = (amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha)).into() / a0;

        self.mode = BiquadFilterMode::Highshelf;
    }

    // Application
    /// Processes an input sample and updates previous samples
    pub fn process(&mut self, sample: &mut T) {
        // Temporary storage of current x2
        let temp_x2 = self.x2;

        // Update previous input samples
        self.x2 = self.x1;
        self.x1 = *sample;

        // Process
        let in_current = self.b0 * self.x1;
        let in_previous = self.b1 * self.x2;
        let in_previous_2 = self.b2 * temp_x2;
        let sum_in = in_current + in_previous + in_previous_2;

        let out_previous = self.a1 * self.y1;
        let out_previous_2 = self.a2 * self.y2;
        let sub_out = out_previous + out_previous_2;

        *sample = (sum_in - sub_out).into();

        // Update previous output samples
        self.y2 = self.y1;
        self.y1 = *sample;
    }

    /// Processes a slice of samples
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }

    // Helper functions
    /// Returns as (alpha, omega)
    /// 
    /// Used for all biquad filters except
    /// - Lowshelf
    /// - Highshelf
    fn alpha_omega(
        sample_rate: f64,
        cutoff: f64,
        q: f64,
    ) -> (f64, f64) {
        let omega = (2.0 * PI * cutoff) / sample_rate;
        let alpha = omega.sin() / (2.0 * q);

        (alpha, omega)
    }

    /// Calculates a1 and a2 for
    /// - Lowpass
    /// - Highpass
    /// - Bandpass
    /// - Notch
    /// 
    /// Biquad filters as (a1, a2)
    fn pass_biquad_a_coefficients(alpha: f64, omega: f64) -> (f64, f64) {
        let a1 = -2.0 * omega.cos();
        let a2 = 1.0 - alpha;

        (a1, a2)
    }
}

/// Finite Impulse Response (FIR) Filter
/// 
/// Buffer is FIFO so index 0 is the newest and index `self.len() - 1` is the oldest
/// 
/// Buffer is rotating so the oldest value gets overridden with the newest value
pub struct FIRFilter<T> {
    taps: Vec<T>,
    buffer: SlidingWindow<T>,
    mode: FIRFilterMode,
}

impl<T> FIRFilter<T>
where
    T: Copy + Default + Mul<Output = T> + Add<Output = T>,
{
    /// Creates a new `FIRFilter with tap capacity and a specified filter mode`
    pub fn new(capacity: usize, mode: FIRFilterMode) -> Self {
        Self {
            taps: Vec::with_capacity(capacity),
            buffer: SlidingWindow::with_capacity(capacity),
            mode,
        }
    }

    // Getters
    /// Gets the list of tap coefficients
    pub fn get_taps(&self) -> Vec<T> {
        self.taps.clone()
    }

    /// Gets a specific tap at index
    /// 
    /// Returns `None` if index out of bounds
    pub fn get_tap_index(&self, index: usize) -> Option<T> {
        if self.taps.len() > index {
            return Some(self.taps[index]);
        }

        None
    }

    /// Gets the amount of taps
    pub fn get_tap_length(&self) -> usize {
        self.taps.len()
    }

    /// Gets the current buffer of past input samples
    pub fn get_buffer(&self) -> SlidingWindow<T> {
        self.buffer.clone()
    }

    /// Gets a specific input sample from the internal buffer
    /// 
    /// Returns `None` if index out of bounds
    pub fn get_buffer_index(&self, index: usize) -> Option<T> {
        if self.buffer.len() > index {
            return Some(self.buffer[index]);
        }

        None
    }

    /// Gets the amount of input samples in the buffer, or the size of the buffer if it is full
    pub fn get_buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Gets the current mode of the `FIRFilter`
    pub fn get_mode(&self) -> FIRFilterMode {
        self.mode
    }

    // Setters
    /// Resizes the previous input sample buffer and the taps buffer
    pub fn resize(&mut self, new_capacity: usize) {
        self.buffer.resize(new_capacity, T::default());
        self.taps.resize(new_capacity, T::default());
    }

    // Application
    pub fn process(&mut self, sample: &mut T) {
        self.buffer.slide(*sample);

        let (left, right) = self.buffer.as_slices();
        let mid_point = left.len();

        let mut sum = T::default();

        for (&value, &tap) in left.iter().zip(self.taps[..mid_point].iter()) {
            sum = sum + (value * tap)
        }

        for (&value, &tap) in right.iter().zip(self.taps[mid_point..].iter()) {
            sum = sum + (value * tap)
        }

        *sample = sum;
    }

    pub fn batch_apply(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

impl<T> FIRFilter<T>
where
    T: Copy + Default + Mul<Output = T> + Add<Output = T>,
    f64: Into<T>,
{
    /// Updates taps to represent a lowpass filter
    pub fn update_lowpass<V>(
        &mut self,
        cutoff: V,
        sample_rate: V,
        window: WindowFunction,
    )
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let sample_rate: f64 = sample_rate.into();
        let num_taps = self.taps.capacity();

        self.taps.clear();
        for tap_index in 0..num_taps {
            self.taps.push(Self::sinc_lowpass(tap_index, num_taps, cutoff, sample_rate, &window).into());
        }

        self.mode = FIRFilterMode::Lowpass { cutoff, sample_rate };
    }

    /// Updates taps to represent a highpass filter
    ///
    /// Best results with an odd number of taps (Type I FIR)
    pub fn update_highpass<V>(
        &mut self,
        cutoff: V,
        sample_rate: V,
        window: WindowFunction,
    )
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let sample_rate: f64 = sample_rate.into();
        let num_taps = self.taps.capacity();
        let center_tap = (num_taps - 1) / 2;

        self.taps.clear();
        for tap_index in 0..num_taps {
            let lowpass_tap = Self::sinc_lowpass(tap_index, num_taps, cutoff, sample_rate, &window);
            let identity_tap = if tap_index == center_tap { 1.0 } else { 0.0 };
            self.taps.push((identity_tap - lowpass_tap).into());
        }

        self.mode = FIRFilterMode::Highpass { cutoff, sample_rate };
    }

    /// Updates taps to represent a bandpass filter
    pub fn update_bandpass<V>(
        &mut self,
        low_cutoff: V,
        high_cutoff: V,
        sample_rate: V,
        window: WindowFunction,
    )
    where
        V: Into<f64>,
    {
        let low_cutoff: f64 = low_cutoff.into();
        let high_cutoff: f64 = high_cutoff.into();
        let sample_rate: f64 = sample_rate.into();
        let num_taps = self.taps.capacity();

        self.taps.clear();
        for tap_index in 0..num_taps {
            let lowpass_at_high_cutoff = Self::sinc_lowpass(tap_index, num_taps, high_cutoff, sample_rate, &window);
            let lowpass_at_low_cutoff  = Self::sinc_lowpass(tap_index, num_taps, low_cutoff,  sample_rate, &window);
            self.taps.push((lowpass_at_high_cutoff - lowpass_at_low_cutoff).into());
        }

        self.mode = FIRFilterMode::Bandpass { low_cutoff, high_cutoff, sample_rate };
    }

    /// Updates taps to represent a notch filter
    ///
    /// Bandwidth defaults to two frequency bins (`2 * sample_rate / num_taps`).
    /// Best results with an odd number of taps (Type I FIR)
    pub fn update_notch<V>(
        &mut self,
        cutoff: V,
        sample_rate: V,
        window: WindowFunction,
    )
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let sample_rate: f64 = sample_rate.into();
        let num_taps = self.taps.capacity();
        let center_tap = (num_taps - 1) / 2;

        let bandwidth = 2.0 * sample_rate / num_taps as f64;
        let notch_low_cutoff  = (cutoff - bandwidth / 2.0).max(0.0);
        let notch_high_cutoff = (cutoff + bandwidth / 2.0).min(sample_rate / 2.0);

        self.taps.clear();
        for tap_index in 0..num_taps {
            let lowpass_at_high_cutoff = Self::sinc_lowpass(tap_index, num_taps, notch_high_cutoff, sample_rate, &window);
            let lowpass_at_low_cutoff  = Self::sinc_lowpass(tap_index, num_taps, notch_low_cutoff,  sample_rate, &window);
            let bandpass_tap = lowpass_at_high_cutoff - lowpass_at_low_cutoff;
            let identity_tap = if tap_index == center_tap { 1.0 } else { 0.0 };
            self.taps.push((identity_tap - bandpass_tap).into());
        }

        self.mode = FIRFilterMode::Notch { cutoff, sample_rate };
    }

    /// Updates taps to represent a Hilbert transformer
    ///
    /// Best results with an odd number of taps (Type III FIR)
    pub fn update_hilbert(&mut self, window: WindowFunction) {
        let num_taps = self.taps.capacity();
        let center = (num_taps as f64 - 1.0) / 2.0;

        self.taps.clear();
        for tap_index in 0..num_taps {
            let offset_from_center = tap_index as f64 - center;
            let tap_value = if offset_from_center.abs() < 1e-10 {
                0.0
            } else {
                2.0 * (PI * offset_from_center / 2.0).sin().powi(2) / (PI * offset_from_center)
            };
            self.taps.push((tap_value * window.apply(num_taps, tap_index)).into());
        }

        self.mode = FIRFilterMode::Hilbert;
    }

    /// Updates taps to represent a differentiator
    pub fn update_differentiator(&mut self, window: WindowFunction) {
        let num_taps = self.taps.capacity();
        let center = (num_taps as f64 - 1.0) / 2.0;

        self.taps.clear();
        for tap_index in 0..num_taps {
            let offset_from_center = tap_index as f64 - center;
            let tap_value = if offset_from_center.abs() < 1e-10 {
                0.0
            } else {
                (PI * offset_from_center).cos() / offset_from_center
            };
            self.taps.push((tap_value * window.apply(num_taps, tap_index)).into());
        }

        self.mode = FIRFilterMode::Differentiator;
    }

    // Helper functions
    fn sinc_lowpass(
        tap_index: usize,
        num_taps: usize,
        cutoff: f64,
        sample_rate: f64,
        window: &WindowFunction,
    ) -> f64 {
        let center = (num_taps as f64 - 1.0) / 2.0;
        let normalized_cutoff = cutoff / sample_rate;
        let offset_from_center = tap_index as f64 - center;
        let tap_value = if offset_from_center.abs() < 1e-10 {
            2.0 * normalized_cutoff
        } else {
            (2.0 * PI * normalized_cutoff * offset_from_center).sin() / (PI * offset_from_center)
        };
        tap_value * window.apply(num_taps, tap_index)
    }
}

// ---- Functions ------------
/// Calculates decibel gain from linear gain
fn decibel_to_amplitude(gain: f64) -> f64 {
    let decibel = 20.0 * gain.log10();
    10.0_f64.powf(decibel / 40.0)
}
