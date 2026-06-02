//! Filters

// ---- Imports ------------
use std::{f64::consts::PI, ops::{Add, Div, Mul, Sub}, time::Duration};
use num_traits::Float;
use crate::data_types::{BiquadFilterMode, ChebyshevType, CombMode, FIRFilterMode, SlidingWindow, SvfOutput, WindowFunction};

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

// ---- Allpass Filter ------
#[derive(Debug, Clone)]
/// # AllpassFilter
/// A first-order allpass filter
///
/// Passes all frequencies at equal volume but shifts their phase by varying amounts
/// depending on frequency. Used as a building block for reverb diffusion and phasers.
pub struct AllpassFilter<T> {
    /// Phase-shift coefficient, should stay between -1.0 and 1.0
    coefficient: T,
    /// Internal delay state
    z1: T,
}

impl<T> Default for AllpassFilter<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            coefficient: T::default(),
            z1: T::default(),
        }
    }
}

impl<T> AllpassFilter<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `AllpassFilter` with the given coefficient
    pub fn new<V>(coefficient: V) -> Self
    where
        V: Into<f64>,
    {
        let coefficient: f64 = coefficient.into();
        Self {
            coefficient: coefficient.into(),
            z1: T::default(),
        }
    }

    // Getters
    /// Gets the current phase-shift coefficient
    pub fn get_coefficient(&self) -> T {
        self.coefficient
    }

    /// Gets the current internal delay state
    pub fn get_z1(&self) -> T {
        self.z1
    }

    // Setters
    /// Updates the phase-shift coefficient
    pub fn update<V>(&mut self, coefficient: V)
    where
        V: Into<f64>,
    {
        let coefficient: f64 = coefficient.into();
        self.coefficient = coefficient.into();
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        let w = *sample + self.coefficient * self.z1;
        *sample = self.z1 - self.coefficient * w;
        self.z1 = w;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples {
            self.process(sample);
        }
    }
}

// ---- State Variable Filter ------
#[derive(Debug, Clone)]
/// # StateVariableFilter
/// A single-pass filter that outputs lowpass, highpass, and bandpass simultaneously
///
/// More efficient than running three separate filters when multiple responses are needed
/// from the same signal.
pub struct StateVariableFilter<T> {
    // State variables
    low: T,
    high: T,
    band: T,
    // Precomputed coefficients
    /// `2 * sin(π * cutoff / sample_rate)` — recomputed on `update`
    frequency_coefficient: T,
    /// `1.0 / q` — recomputed on `update`
    damping: T,
}

impl<T> Default for StateVariableFilter<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            low: T::default(),
            high: T::default(),
            band: T::default(),
            frequency_coefficient: T::default(),
            damping: T::default(),
        }
    }
}

impl<T> StateVariableFilter<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `StateVariableFilter` with the given cutoff, Q, and sample rate
    pub fn new<V>(cutoff: V, q: V, sample_rate: V) -> Self
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let q: f64 = q.into();
        let sample_rate: f64 = sample_rate.into();

        Self {
            low: T::default(),
            high: T::default(),
            band: T::default(),
            frequency_coefficient: Self::compute_frequency_coefficient(cutoff, sample_rate).into(),
            damping: (1.0 / q).into(),
        }
    }

    // Getters
    /// Gets the last computed lowpass output
    pub fn get_low(&self) -> T {
        self.low
    }

    /// Gets the last computed highpass output
    pub fn get_high(&self) -> T {
        self.high
    }

    /// Gets the last computed bandpass output
    pub fn get_band(&self) -> T {
        self.band
    }

    // Setters
    /// Updates the cutoff frequency and Q, recomputing internal coefficients
    pub fn update<V>(&mut self, cutoff: V, q: V, sample_rate: V)
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let q: f64 = q.into();
        let sample_rate: f64 = sample_rate.into();

        self.frequency_coefficient = Self::compute_frequency_coefficient(cutoff, sample_rate).into();
        self.damping = (1.0 / q).into();
    }

    // Application
    /// Processes a single sample, returning all three filter outputs simultaneously
    pub fn process(&mut self, sample: T) -> SvfOutput<T> {
        self.low  = self.low + self.frequency_coefficient * self.band;
        self.high = sample - self.low - self.damping * self.band;
        self.band = self.frequency_coefficient * self.high + self.band;

        SvfOutput {
            low: self.low,
            high: self.high,
            band: self.band,
        }
    }

    /// Processes a slice of samples in place, writing back the lowpass output
    pub fn batch_process_low(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample).low;
        }
    }

    /// Processes a slice of samples in place, writing back the highpass output
    pub fn batch_process_high(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample).high;
        }
    }

    /// Processes a slice of samples in place, writing back the bandpass output
    pub fn batch_process_band(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            *sample = self.process(*sample).band;
        }
    }

    // Helper functions
    fn compute_frequency_coefficient(cutoff: f64, sample_rate: f64) -> f64 {
        2.0 * (PI * cutoff / sample_rate).sin()
    }
}

// ---- Comb Filter ------
#[derive(Debug, Clone)]
/// # CombFilter
/// Mixes a signal with a delayed copy of itself, creating evenly spaced peaks and notches
///
/// - `Feedforward` — input mixed with delayed input (sharper notches)
/// - `Feedback` — input mixed with delayed output (resonant, ringing peaks)
pub struct CombFilter<T> {
    buffer: SlidingWindow<T>,
    gain: T,
    mode: CombMode,
}

impl<T> CombFilter<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `CombFilter` with the given delay in samples and gain
    pub fn new<V>(delay_samples: usize, gain: V, mode: CombMode) -> Self
    where
        V: Into<f64>,
    {
        let delay_samples = delay_samples.max(1);
        let mut buffer = SlidingWindow::with_capacity(delay_samples);
        for _ in 0..delay_samples {
            buffer.slide(T::default());
        }
        Self {
            buffer,
            gain: gain.into().into(),
            mode,
        }
    }

    /// Creates a new `CombFilter`, converting a `Duration` delay to samples internally
    pub fn new_from_duration<V>(delay: Duration, sample_rate: V, gain: V, mode: CombMode) -> Self
    where
        V: Into<f64>,
    {
        let delay_samples = (delay.as_secs_f64() * sample_rate.into()) as usize;
        Self::new(delay_samples, gain, mode)
    }

    // Getters
    /// Gets the current delay length in samples
    pub fn get_delay_samples(&self) -> usize {
        self.buffer.capacity()
    }

    /// Gets the current comb mode
    pub fn get_mode(&self) -> CombMode {
        self.mode
    }

    /// Gets a reference to the internal delay buffer
    pub fn get_buffer(&self) -> &SlidingWindow<T> {
        &self.buffer
    }

    // Setters
    /// Sets a new delay length in samples, clearing the buffer
    pub fn update(&mut self, delay_samples: usize) {
        let delay_samples = delay_samples.max(1);
        let mut buffer = SlidingWindow::with_capacity(delay_samples);
        for _ in 0..delay_samples {
            buffer.slide(T::default());
        }
        self.buffer = buffer;
    }

    /// Sets a new delay length from a `Duration`, clearing the buffer
    pub fn update_from_duration<V>(&mut self, delay: Duration, sample_rate: V)
    where
        V: Into<f64>,
    {
        let delay_samples = (delay.as_secs_f64() * sample_rate.into()) as usize;
        self.update(delay_samples);
    }

    /// Updates the gain
    pub fn update_gain<V>(&mut self, gain: V)
    where
        V: Into<f64>,
    {
        self.gain = gain.into().into();
    }

    // Application
    /// Processes a single sample in place
    ///
    /// `buffer[0]` is always the oldest (most-delayed) sample because `SlidingWindow`
    /// pre-fills with zeros on construction and overwrites the oldest slot on each `slide`.
    pub fn process(&mut self, sample: &mut T) {
        let delayed = self.buffer[0];
        match self.mode {
            CombMode::Feedforward => {
                self.buffer.slide(*sample);
                *sample = *sample + self.gain * delayed;
            }
            CombMode::Feedback => {
                let output = *sample + self.gain * delayed;
                self.buffer.slide(output);
                *sample = output;
            }
        }
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- DC Blocker ------
#[derive(Debug, Clone)]
/// # DcBlocker
/// Removes DC offset (0 Hz bias) from a signal using a one-pole highpass filter
///
/// The cutoff is fixed at ~10 Hz, which is low enough to leave audio untouched
/// while eliminating any constant voltage bias.
pub struct DcBlocker<T> {
    x1: T,
    y1: T,
    /// `1 - 2π * 10 / sample_rate` — recomputed on `update`
    coefficient: T,
}

impl<T> Default for DcBlocker<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            x1: T::default(),
            y1: T::default(),
            coefficient: T::default(),
        }
    }
}

impl<T> DcBlocker<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `DcBlocker` tuned to the given sample rate
    pub fn new<V: Into<f64>>(sample_rate: V) -> Self {
        Self {
            x1: T::default(),
            y1: T::default(),
            coefficient: (1.0 - (2.0 * PI * 10.0 / sample_rate.into())).into(),
        }
    }

    // Getters
    /// Gets the internal pole coefficient
    pub fn get_coefficient(&self) -> T { self.coefficient }
    /// Gets the previous input sample
    pub fn get_x1(&self) -> T { self.x1 }
    /// Gets the previous output sample
    pub fn get_y1(&self) -> T { self.y1 }

    // Setters
    /// Recomputes the coefficient for a new sample rate
    pub fn update<V: Into<f64>>(&mut self, sample_rate: V) {
        self.coefficient = (1.0 - (2.0 * PI * 10.0 / sample_rate.into())).into();
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        let output = *sample - self.x1 + self.coefficient * self.y1;
        self.x1 = *sample;
        self.y1 = output;
        *sample = output;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Leaky Integrator ------
#[derive(Debug, Clone)]
/// # LeakyIntegrator
/// Smooths a signal by blending input with the previous output
///
/// `coefficient` is between 0 and 1 — closer to 1 means more smoothing (slower response),
/// closer to 0 means the output tracks the input closely.
///
/// Formula: `y[n] = (1 - α) * x[n] + α * y[n-1]`
pub struct LeakyIntegrator<T> {
    coefficient: T,
    one_minus_coefficient: T,
    state: T,
}

impl<T> Default for LeakyIntegrator<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            coefficient: T::default(),
            one_minus_coefficient: T::default(),
            state: T::default(),
        }
    }
}

impl<T> LeakyIntegrator<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `LeakyIntegrator` with the given smoothing coefficient (0–1)
    pub fn new<V: Into<f64>>(coefficient: V) -> Self {
        let alpha: f64 = coefficient.into();
        Self {
            coefficient: alpha.into(),
            one_minus_coefficient: (1.0 - alpha).into(),
            state: T::default(),
        }
    }

    // Getters
    /// Gets the current smoothing coefficient
    pub fn get_coefficient(&self) -> T { self.coefficient }
    /// Gets the current internal state
    pub fn get_state(&self) -> T { self.state }

    // Setters
    /// Updates the smoothing coefficient
    pub fn update<V: Into<f64>>(&mut self, coefficient: V) {
        let alpha: f64 = coefficient.into();
        self.coefficient = alpha.into();
        self.one_minus_coefficient = (1.0 - alpha).into();
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        self.state = self.one_minus_coefficient * *sample + self.coefficient * self.state;
        *sample = self.state;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Exponential Moving Average ------
#[derive(Debug, Clone)]
/// # ExponentialMovingAverage
/// Weights recent samples more heavily via a single smoothing coefficient
///
/// `coefficient` is between 0 and 1 — closer to 1 means faster response (less smoothing),
/// closer to 0 means more smoothing.
///
/// Formula: `y[n] = α * x[n] + (1 - α) * y[n-1]`
pub struct ExponentialMovingAverage<T> {
    coefficient: T,
    one_minus_coefficient: T,
    state: T,
}

impl<T> Default for ExponentialMovingAverage<T>
where
    T: Default,
{
    fn default() -> Self {
        Self {
            coefficient: T::default(),
            one_minus_coefficient: T::default(),
            state: T::default(),
        }
    }
}

impl<T> ExponentialMovingAverage<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `ExponentialMovingAverage` with the given smoothing coefficient (0–1)
    pub fn new<V: Into<f64>>(coefficient: V) -> Self {
        let alpha: f64 = coefficient.into();
        Self {
            coefficient: alpha.into(),
            one_minus_coefficient: (1.0 - alpha).into(),
            state: T::default(),
        }
    }

    // Getters
    /// Gets the current smoothing coefficient
    pub fn get_coefficient(&self) -> T { self.coefficient }
    /// Gets the current internal state
    pub fn get_state(&self) -> T { self.state }

    // Setters
    /// Updates the smoothing coefficient
    pub fn update<V: Into<f64>>(&mut self, coefficient: V) {
        let alpha: f64 = coefficient.into();
        self.coefficient = alpha.into();
        self.one_minus_coefficient = (1.0 - alpha).into();
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        self.state = self.coefficient * *sample + self.one_minus_coefficient * self.state;
        *sample = self.state;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Moving Average ------
#[derive(Debug, Clone)]
/// # MovingAverage
/// Averages the last N samples using a circular buffer and a running sum
///
/// Cheap to run per-sample — the sum is maintained incrementally rather than
/// recomputed each call. Buffer is pre-filled with zeros so output is valid
/// from the very first sample (zero-padded warmup).
pub struct MovingAverage<T> {
    buffer: Vec<T>,
    write_head: usize,
    sum: T,
    /// Precomputed `1.0 / window` to avoid a division in `process`
    inv_window: T,
}

impl<T> MovingAverage<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `MovingAverage` with the given window size
    pub fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            buffer: vec![T::default(); window],
            write_head: 0,
            sum: T::default(),
            inv_window: (1.0 / window as f64).into(),
        }
    }

    // Getters
    /// Gets the current window size
    pub fn get_window(&self) -> usize { self.buffer.len() }
    /// Gets a reference to the internal sample buffer
    pub fn get_buffer(&self) -> &[T] { &self.buffer }

    // Setters
    /// Resizes the window, clearing all internal state
    pub fn update(&mut self, window: usize) {
        let window = window.max(1);
        self.buffer = vec![T::default(); window];
        self.write_head = 0;
        self.sum = T::default();
        self.inv_window = (1.0 / window as f64).into();
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        let oldest = self.buffer[self.write_head];
        self.sum = self.sum + *sample - oldest;
        self.buffer[self.write_head] = *sample;
        self.write_head = (self.write_head + 1) % self.buffer.len();
        *sample = self.sum * self.inv_window;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Butterworth Filter ------
#[derive(Debug, Clone)]
/// # ButterworthFilter
/// A maximally flat IIR filter built from cascaded biquad stages
///
/// No passband ripple at any stage count. Each additional pair of stages steepens
/// the rolloff by 12 dB/octave. `stages` is always even (odd values round down,
/// 0 defaults to 2, max 20).
pub struct ButterworthFilter<T> {
    biquads: Vec<BiquadFilter<T>>,
    stages: usize,
    cutoff: f64,
    sample_rate: f64,
}

impl<T> ButterworthFilter<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `ButterworthFilter` with the given stage count, cutoff, and sample rate
    pub fn new<V>(stages: usize, cutoff: V, sample_rate: V) -> Self
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let sample_rate: f64 = sample_rate.into();
        let stages = Self::normalize_stages(stages);
        let mut biquads = Vec::with_capacity(stages);
        Self::configure_biquads(&mut biquads, stages, cutoff, sample_rate);
        Self { biquads, stages, cutoff, sample_rate }
    }

    // Getters
    /// Gets the current stage count
    pub fn get_stages(&self) -> usize { self.stages }
    /// Gets the current cutoff frequency
    pub fn get_cutoff(&self) -> f64 { self.cutoff }
    /// Gets the current sample rate
    pub fn get_sample_rate(&self) -> f64 { self.sample_rate }
    /// Gets a reference to the internal biquad stage array
    pub fn get_biquads(&self) -> &[BiquadFilter<T>] { &self.biquads }

    // Setters
    /// Recomputes coefficients for all stages at a new cutoff frequency and sample rate
    pub fn update<V>(&mut self, cutoff: V, sample_rate: V)
    where
        V: Into<f64>,
    {
        self.cutoff = cutoff.into();
        self.sample_rate = sample_rate.into();
        Self::configure_biquads(&mut self.biquads, self.stages, self.cutoff, self.sample_rate);
    }

    /// Rebuilds the biquad chain with a new stage count, reusing the current cutoff and sample rate
    pub fn update_stages(&mut self, stages: usize) {
        self.stages = Self::normalize_stages(stages);
        self.biquads = Vec::with_capacity(self.stages);
        Self::configure_biquads(&mut self.biquads, self.stages, self.cutoff, self.sample_rate);
    }

    // Application
    /// Processes a single sample in place through all stages
    pub fn process(&mut self, sample: &mut T) {
        for biquad in self.biquads.iter_mut() {
            biquad.process(sample);
        }
    }

    /// Processes a slice of samples in place through all stages
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }

    // Helper functions
    fn normalize_stages(stages: usize) -> usize {
        if stages == 0 { 2 } else { ((stages / 2).max(1) * 2).min(20) }
    }

    /// Q for stage `k` (1-indexed) of an N-th order Butterworth
    ///
    /// Poles sit on the unit circle at angles `θ_k = π*(2k+N-1)/(2N)`, all in the left
    /// half-plane. Q = -1/(2*cos(θ_k)) — the cosine is negative so Q is always positive.
    fn butterworth_q(k: usize, n: usize) -> f64 {
        let theta = PI * (2 * k + n - 1) as f64 / (2 * n) as f64;
        -1.0 / (2.0 * theta.cos())
    }

    fn configure_biquads(
        biquads: &mut Vec<BiquadFilter<T>>,
        stages: usize,
        cutoff: f64,
        sample_rate: f64,
    ) {
        let n = 2 * stages;
        biquads.clear();
        for k in 1..=stages {
            let q = Self::butterworth_q(k, n);
            let mut biquad = BiquadFilter::default();
            biquad.update_lowpass(cutoff, q, sample_rate);
            biquads.push(biquad);
        }
    }
}

// ---- Chebyshev Filter ------
#[derive(Debug, Clone)]
/// # ChebyshevFilter
/// A sharper-rolloff IIR filter built from cascaded biquad stages, at the cost of ripple
///
/// - `TypeI` — equiripple in the passband, monotone stopband
/// - `TypeII` — equiripple in the stopband, monotone passband
///
/// `ripple` is the allowed ripple in dB. `stages` follows the same rules as `ButterworthFilter`.
pub struct ChebyshevFilter<T> {
    biquads: Vec<BiquadFilter<T>>,
    stages: usize,
    cutoff: f64,
    ripple: f64,
    sample_rate: f64,
    kind: ChebyshevType,
}

impl<T> ChebyshevFilter<T>
where
    T: Copy + Default + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `ChebyshevFilter` with the given parameters
    pub fn new<V>(stages: usize, cutoff: V, ripple: V, kind: ChebyshevType, sample_rate: V) -> Self
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let ripple: f64 = ripple.into();
        let sample_rate: f64 = sample_rate.into();
        let stages = ButterworthFilter::<T>::normalize_stages(stages);
        let mut biquads = Vec::with_capacity(stages);
        Self::configure_biquads(&mut biquads, stages, cutoff, ripple, sample_rate, kind);
        Self { biquads, stages, cutoff, ripple, sample_rate, kind }
    }

    // Getters
    /// Gets the current stage count
    pub fn get_stages(&self) -> usize { self.stages }
    /// Gets the current cutoff frequency
    pub fn get_cutoff(&self) -> f64 { self.cutoff }
    /// Gets the current ripple in dB
    pub fn get_ripple(&self) -> f64 { self.ripple }
    /// Gets the current Chebyshev type
    pub fn get_kind(&self) -> ChebyshevType { self.kind }
    /// Gets the current sample rate
    pub fn get_sample_rate(&self) -> f64 { self.sample_rate }
    /// Gets a reference to the internal biquad stage array
    pub fn get_biquads(&self) -> &[BiquadFilter<T>] { &self.biquads }

    // Setters
    /// Recomputes coefficients for all stages with new parameters
    pub fn update<V>(&mut self, cutoff: V, ripple: V, sample_rate: V)
    where
        V: Into<f64>,
    {
        self.cutoff = cutoff.into();
        self.ripple = ripple.into();
        self.sample_rate = sample_rate.into();
        Self::configure_biquads(&mut self.biquads, self.stages, self.cutoff, self.ripple, self.sample_rate, self.kind);
    }

    /// Rebuilds the biquad chain with a new stage count
    pub fn update_stages(&mut self, stages: usize) {
        self.stages = ButterworthFilter::<T>::normalize_stages(stages);
        self.biquads = Vec::with_capacity(self.stages);
        Self::configure_biquads(&mut self.biquads, self.stages, self.cutoff, self.ripple, self.sample_rate, self.kind);
    }

    /// Switches between Type I and Type II, recomputing all stages
    pub fn update_kind(&mut self, kind: ChebyshevType) {
        self.kind = kind;
        Self::configure_biquads(&mut self.biquads, self.stages, self.cutoff, self.ripple, self.sample_rate, self.kind);
    }

    // Application
    /// Processes a single sample in place through all stages
    pub fn process(&mut self, sample: &mut T) {
        for biquad in self.biquads.iter_mut() {
            biquad.process(sample);
        }
    }

    /// Processes a slice of samples in place through all stages
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }

    // Helper functions
    fn configure_biquads(
        biquads: &mut Vec<BiquadFilter<T>>,
        stages: usize,
        cutoff: f64,
        ripple: f64,
        sample_rate: f64,
        kind: ChebyshevType,
    ) {
        let n = 2 * stages;
        biquads.clear();
        for k in 1..=stages {
            let (b0, b1, b2, a1, a2) = match kind {
                ChebyshevType::TypeI  => Self::type_i_stage(k, n, ripple, cutoff, sample_rate),
                ChebyshevType::TypeII => Self::type_ii_stage(k, n, ripple, cutoff, sample_rate),
            };
            let mut biquad = BiquadFilter::default();
            biquad.update_raw(a1.into(), a2.into(), b0.into(), b1.into(), b2.into());
            biquads.push(biquad);
        }
    }

    /// Bilinear-transform coefficients for Type I stage `k` of an N-th order filter
    ///
    /// Chebyshev poles have unequal magnitudes, so each stage needs its own pole
    /// frequency — `update_lowpass` (which hardcodes the pole at `cutoff`) would
    /// be wrong here. The BLT is applied explicitly with prewarping.
    fn type_i_stage(k: usize, n: usize, ripple_db: f64, cutoff: f64, sample_rate: f64) -> (f64, f64, f64, f64, f64) {
        let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
        let mu = (1.0 / epsilon).asinh() / n as f64;
        let phi = PI * (2 * k - 1) as f64 / (2 * n) as f64;

        let sigma = mu.sinh() * phi.sin();
        let omega = mu.cosh() * phi.cos();
        let pole_magnitude = (sigma * sigma + omega * omega).sqrt();
        let q = pole_magnitude / (2.0 * sigma);

        let f_d = (PI * cutoff / sample_rate).tan();
        let w_p = pole_magnitude * f_d;
        let b_p = w_p / q;
        let a0 = 4.0 + 2.0 * b_p + w_p * w_p;

        (
            w_p * w_p / a0,
            2.0 * w_p * w_p / a0,
            w_p * w_p / a0,
            (-8.0 + 2.0 * w_p * w_p) / a0,
            (4.0 - 2.0 * b_p + w_p * w_p) / a0,
        )
    }

    /// Bilinear-transform coefficients for Type II stage `k` of an N-th order filter
    ///
    /// Type II poles are the reciprocals of the Type I poles (`s → 1/s`).
    /// Each stage also has a pair of imaginary zeros at `±j/sin(φ_k)`, which
    /// produce the stopband notches — these require b1 ≠ 0, unlike a pure lowpass.
    fn type_ii_stage(k: usize, n: usize, ripple_db: f64, cutoff: f64, sample_rate: f64) -> (f64, f64, f64, f64, f64) {
        let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
        let mu = (1.0 / epsilon).asinh() / n as f64;
        let phi = PI * (2 * k - 1) as f64 / (2 * n) as f64;

        // Type I pole → Type II pole via s → 1/s
        let sigma = mu.sinh() * phi.sin();
        let omega = mu.cosh() * phi.cos();
        let s_i_mag_sq = sigma * sigma + omega * omega;
        let pole_magnitude = 1.0 / s_i_mag_sq.sqrt();
        let q_p = s_i_mag_sq.sqrt() / (2.0 * sigma);

        // Imaginary zeros at ±j/sin(φ_k)
        let zero_freq = 1.0 / phi.sin();

        let f_d = (PI * cutoff / sample_rate).tan();
        let w_p = pole_magnitude * f_d;
        let w_z = zero_freq * f_d;
        let b_p = w_p / q_p;
        let a0 = 4.0 + 2.0 * b_p + w_p * w_p;

        (
            (4.0 + w_z * w_z) / a0,
            (-8.0 + 2.0 * w_z * w_z) / a0,
            (4.0 + w_z * w_z) / a0,
            (-8.0 + 2.0 * w_p * w_p) / a0,
            (4.0 - 2.0 * b_p + w_p * w_p) / a0,
        )
    }
}

// ---- Ladder Filter ------
#[derive(Debug, Clone)]
/// # LadderFilter
/// Classic analog-style lowpass modelled on the Moog synthesizer
///
/// Four one-pole lowpass stages with feedback from output to input. `tanh` saturation
/// at each stage produces the warm, nonlinear character. At `resonance` above ~3.5
/// out of 4 the filter self-oscillates, producing a pure sine at the cutoff frequency.
pub struct LadderFilter<T: Float> {
    poles: [T; 4],
    feedback: T,
    frequency_coefficient: T,
    resonance: T,
    cutoff: f64,
    sample_rate: f64,
}

impl<T: Float> LadderFilter<T> {
    /// Creates a new `LadderFilter` with the given cutoff frequency, resonance (0–4), and sample rate
    pub fn new<V>(cutoff: V, resonance: V, sample_rate: V) -> Self
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let resonance: f64 = resonance.into();
        let sample_rate: f64 = sample_rate.into();
        let zero = T::from(0.0).unwrap();
        Self {
            poles: [zero; 4],
            feedback: zero,
            frequency_coefficient: T::from(2.0 * (PI * cutoff / sample_rate).sin()).unwrap(),
            resonance: T::from(resonance).unwrap(),
            cutoff,
            sample_rate,
        }
    }

    // Getters
    /// Gets the current cutoff frequency
    pub fn get_cutoff(&self) -> f64 { self.cutoff }
    /// Gets the current resonance value
    pub fn get_resonance(&self) -> T { self.resonance }
    /// Gets the current sample rate
    pub fn get_sample_rate(&self) -> f64 { self.sample_rate }
    /// Gets the four internal pole states
    pub fn get_poles(&self) -> [T; 4] { self.poles }
    /// Gets the feedback signal from the last processed sample
    pub fn get_feedback(&self) -> T { self.feedback }

    // Setters
    /// Recomputes coefficients for a new cutoff, resonance, and sample rate
    pub fn update<V>(&mut self, cutoff: V, resonance: V, sample_rate: V)
    where
        V: Into<f64>,
    {
        let cutoff: f64 = cutoff.into();
        let resonance: f64 = resonance.into();
        let sample_rate: f64 = sample_rate.into();
        self.cutoff = cutoff;
        self.sample_rate = sample_rate;
        self.frequency_coefficient = T::from(2.0 * (PI * cutoff / sample_rate).sin()).unwrap();
        self.resonance = T::from(resonance).unwrap();
    }

    // Application
    /// Processes a single sample in place
    ///
    /// Each of the four poles applies a one-pole lowpass with `tanh` saturation.
    /// Feedback subtracts a scaled version of the last pole output from the input,
    /// producing resonance.
    pub fn process(&mut self, sample: &mut T) {
        let input = *sample - self.resonance * self.poles[3];
        let p0 = self.poles[0] + self.frequency_coefficient * (input.tanh() - self.poles[0].tanh());
        let p1 = self.poles[1] + self.frequency_coefficient * (p0.tanh() - self.poles[1].tanh());
        let p2 = self.poles[2] + self.frequency_coefficient * (p1.tanh() - self.poles[2].tanh());
        let p3 = self.poles[3] + self.frequency_coefficient * (p2.tanh() - self.poles[3].tanh());
        self.poles = [p0, p1, p2, p3];
        self.feedback = self.resonance * p3;
        *sample = p3;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Savitzky-Golay Filter ------
#[derive(Debug, Clone)]
/// # SavitzkyGolay
/// Smooths a signal by fitting a polynomial to each window of samples
///
/// Preserves peaks and edges far better than a moving average — the polynomial fit
/// tracks the local shape of the signal rather than averaging it away.
/// `window` must be odd and > `degree`. The coefficients are recomputed only in
/// `new` and `update`, not per sample.
pub struct SavitzkyGolay<T> {
    window: usize,
    degree: usize,
    coefficients: Vec<T>,
    buffer: SlidingWindow<T>,
}

impl<T> SavitzkyGolay<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `SavitzkyGolay` filter. `window` is clamped to the nearest larger
    /// odd number if even, and `degree` is clamped to `window - 1` if too large.
    pub fn new(window: usize, degree: usize) -> Self {
        let (window, degree) = Self::normalize_params(window, degree);
        let coefficients = Self::compute_coefficients(window, degree);
        let mut buffer = SlidingWindow::with_capacity(window);
        
        for _ in 0..window {
            buffer.slide(T::default());
        }

        Self {
            window,
            degree,
            coefficients: coefficients.into_iter().map(|c| c.into()).collect(),
            buffer,
        }
    }

    // Getters
    /// Gets the current window size
    pub fn get_window(&self) -> usize { self.window }

    /// Gets the current polynomial degree
    pub fn get_degree(&self) -> usize { self.degree }

    /// Gets a reference to the computed convolution coefficients
    pub fn get_coefficients(&self) -> &[T] { &self.coefficients }

    /// Gets a reference to the internal sample buffer
    pub fn get_buffer(&self) -> &SlidingWindow<T> { &self.buffer }

    // Setters
    /// Recomputes filter coefficients with new window and degree
    pub fn update(&mut self, window: usize, degree: usize) {
        let (window, degree) = Self::normalize_params(window, degree);
        self.window = window;
        self.degree = degree;

        let raw = Self::compute_coefficients(window, degree);
        self.coefficients = raw.into_iter().map(|c| c.into()).collect();
        self.buffer = SlidingWindow::with_capacity(window);

        for _ in 0..window {
            self.buffer.slide(T::default());
        }
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        self.buffer.slide(*sample);
        let (left, right) = self.buffer.as_slices();
        let mid = left.len();
        let mut sum = T::default();

        for (&value, &coeff) in left.iter().zip(self.coefficients[..mid].iter()) {
            sum = sum + value * coeff;
        }

        for (&value, &coeff) in right.iter().zip(self.coefficients[mid..].iter()) {
            sum = sum + value * coeff;
        }

        *sample = sum;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }

    // Helper functions
    fn normalize_params(window: usize, degree: usize) -> (usize, usize) {
        let window = window.max(3) | 1; // at least 3, force odd
        let degree = degree.min(window - 1).max(1);
        (window, degree)
    }

    /// Computes causal Savitzky-Golay coefficients via least-squares pseudoinverse
    ///
    /// Sets up the Vandermonde design matrix for the last `window` samples, fits a
    /// polynomial of the given degree, and extracts the row that evaluates the fit
    /// at the most recent sample (time = 0). The result is a causal FIR filter.
    fn compute_coefficients(window: usize, degree: usize) -> Vec<f64> {
        let d = degree + 1;
        // Design matrix: A[i][j] = (i - (window-1))^j  (time: -(window-1)..0)
        let mut a = vec![vec![0.0_f64; d]; window];
        let h = (window - 1) as f64;
        for i in 0..window {
            for j in 0..d {
                a[i][j] = (i as f64 - h).powi(j as i32);
            }
        }
        // A^T A  (d x d)
        let mut ata = vec![vec![0.0_f64; d]; d];
        for row in 0..d {
            for col in 0..d {
                for i in 0..window {
                    ata[row][col] += a[i][row] * a[i][col];
                }
            }
        }

        // Invert A^T A via Gauss-Jordan elimination
        let inv = Self::invert_matrix(ata);
        // Coefficients = first row of (A^T A)^{-1} A^T
        // c[i] = sum_k inv[0][k] * A[i][k]
        (0..window)
        .map(|i| (0..d).map(|k| inv[0][k] * a[i][k]).sum())
        .collect()
    }

    fn invert_matrix(mut m: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = m.len();
        // Augment with identity
        let mut aug: Vec<Vec<f64>> = m
        .iter_mut()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { 1.0 } else { 0.0 });
            }
            r
        })
        .collect();

        // Forward elimination
        for col in 0..n {
            // Partial pivot
            let pivot = (col..n)
            .max_by(|&a, &b| aug[a][col].abs()
            .partial_cmp(&aug[b][col].abs())
            .unwrap());

            if let Some(pivot) = pivot {
                aug.swap(col, pivot);
            }

            let diag = aug[col][col];
            if diag.abs() < 1e-12 { continue; }

            for j in 0..2 * n {
                aug[col][j] /= diag;
            }

            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        aug.into_iter().map(|row| row[n..].to_vec()).collect()
    }
}

// ---- Wiener Filter ------
#[derive(Debug, Clone)]
/// # WienerFilter
/// Reduces stationary noise using a precomputed noise profile
///
/// Given a per-tap noise power estimate (typically measured from a silent section),
/// the filter computes coefficients `h[k] = 1 / (1 + noise[k])` — close to 1 where
/// noise is low (passes signal), close to 0 where noise is high (attenuates it).
/// Call `update_noise_profile` any time the noise estimate changes.
pub struct WienerFilter<T> {
    noise_profile: Vec<f64>,
    coefficients: Vec<T>,
    buffer: SlidingWindow<T>,
}

impl<T> WienerFilter<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T>,
    f64: Into<T>,
{
    /// Creates a new `WienerFilter` from the given noise profile
    pub fn new(noise_profile: &[f64]) -> Self {
        let len = noise_profile.len().max(1);
        let coefficients = Self::compute_coefficients(noise_profile);
        let mut buffer = SlidingWindow::with_capacity(len);
        for _ in 0..len {
            buffer.slide(T::default());
        }
        Self {
            noise_profile: noise_profile.to_vec(),
            coefficients: coefficients.into_iter().map(|c| c.into()).collect(),
            buffer,
        }
    }

    // Getters
    /// Gets a reference to the current noise profile
    pub fn get_noise_profile(&self) -> &[f64] { &self.noise_profile }
    /// Gets a reference to the internal sample buffer
    pub fn get_buffer(&self) -> &SlidingWindow<T> { &self.buffer }

    // Setters
    /// Replaces the noise profile and recomputes filter coefficients
    pub fn update_noise_profile(&mut self, noise_profile: &[f64]) {
        let new_len = noise_profile.len().max(1);
        let old_len = self.noise_profile.len();
        self.noise_profile = noise_profile.to_vec();
        self.coefficients = Self::compute_coefficients(noise_profile)
            .into_iter()
            .map(|c| c.into())
            .collect();
        if new_len != old_len {
            self.buffer = SlidingWindow::with_capacity(new_len);
            for _ in 0..new_len {
                self.buffer.slide(T::default());
            }
        }
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        self.buffer.slide(*sample);
        let (left, right) = self.buffer.as_slices();
        let mid = left.len();
        let mut sum = T::default();
        for (&value, &coeff) in left.iter().zip(self.coefficients[..mid].iter()) {
            sum = sum + value * coeff;
        }
        for (&value, &coeff) in right.iter().zip(self.coefficients[mid..].iter()) {
            sum = sum + value * coeff;
        }
        *sample = sum;
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }

    // Helper functions
    fn compute_coefficients(noise_profile: &[f64]) -> Vec<f64> {
        noise_profile.iter().map(|&n| 1.0 / (1.0 + n)).collect()
    }
}

// ---- Median Filter ------
#[derive(Debug, Clone)]
/// # MedianFilter
/// Replaces each sample with the median of its surrounding window
///
/// Excellent at removing impulse noise (clicks, spikes) without blurring transients.
/// Has no frequency response in the traditional sense — purely order-based.
/// Odd window sizes are recommended so the median is unambiguous.
pub struct MedianFilter<T> {
    buffer: SlidingWindow<T>,
}

impl<T> MedianFilter<T>
where
    T: Copy + Default + PartialOrd,
{
    /// Creates a new `MedianFilter` with the given window size
    pub fn new(window: usize) -> Self {
        let window = window.max(1);
        let mut buffer = SlidingWindow::with_capacity(window);
        for _ in 0..window {
            buffer.slide(T::default());
        }
        Self { buffer }
    }

    // Getters
    /// Gets the current window size
    pub fn get_window(&self) -> usize { self.buffer.capacity() }
    /// Gets a reference to the internal sample buffer
    pub fn get_buffer(&self) -> &SlidingWindow<T> { &self.buffer }

    // Setters
    /// Resizes the window, clearing internal state
    pub fn update(&mut self, window: usize) {
        let window = window.max(1);
        self.buffer = SlidingWindow::with_capacity(window);
        for _ in 0..window {
            self.buffer.slide(T::default());
        }
    }

    // Application
    /// Processes a single sample in place
    pub fn process(&mut self, sample: &mut T) {
        self.buffer.slide(*sample);
        let window = self.buffer.capacity();
        let mut sorted: Vec<T> = (0..window).map(|i| self.buffer[i]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *sample = sorted[window / 2];
    }

    /// Processes a slice of samples in place
    pub fn batch_process(&mut self, samples: &mut [T]) {
        for sample in samples.iter_mut() {
            self.process(sample);
        }
    }
}

// ---- Functions ------------
/// Calculates decibel gain from linear gain
fn decibel_to_amplitude(gain: f64) -> f64 {
    let decibel = 20.0 * gain.log10();
    10.0_f64.powf(decibel / 40.0)
}
