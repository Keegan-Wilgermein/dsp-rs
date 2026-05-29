//! Filters

// ---- Imports ------------
use std::f64::consts::PI;
use crate::data_types::{self, FilterMode};

// ---- Structs -------------
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    // Input coefficients
    b0: f64,
    b1: f64,
    b2: f64,

    // Output coefficients
    a1: f64,
    a2: f64,
    
    // Previous input samples
    x1: f64,
    x2: f64,
    
    // Previous output samples
    y1: f64,
    y2: f64,
}

impl Default for BiquadFilter {
    fn default() -> Self {
        Self {
            a1: 0.0,
            a2: 0.0,
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}

impl BiquadFilter {
    // Getters
    pub fn get_a1(&self) -> f64 {
        self.a1
    }

    pub fn get_a2(&self) -> f64 {
        self.a2
    }

    pub fn get_b0(&self) -> f64 {
        self.b0
    }

    pub fn get_b1(&self) -> f64 {
        self.b1
    }

    pub fn get_b2(&self) -> f64 {
        self.b2
    }

    pub fn get_x1(&self) -> f64 {
        self.x1
    }

    pub fn get_x2(&self) -> f64 {
        self.x2
    }

    pub fn get_y1(&self) -> f64 {
        self.y1
    }

    pub fn get_y2(&self) -> f64 {
        self.y2
    }

    // Setters
    /// Updates coefficients with provided values
    pub fn update_raw<T>(
        &mut self,
        a1: T,
        a2: T,
        b0: T,
        b1: T,
        b2: T,
    )
    where
        T: Into<f64>,
    {
        self.a1 = a1.into();
        self.a2 = a2.into();
        self.b0 = b0.into();
        self.b1 = b1.into();
        self.b2 = b2.into();
    }

    pub fn update_lowpass(
        &mut self,
        cutoff: f64,
        q: f64,
        sample_rate: f64,
    ) {
        let (alpha, omega) = alpha_omega(sample_rate, cutoff, q);
        let omega_cos = omega.cos();
        let a0 = 1.0 + alpha;

        // Calculate a coefficients
        let (a1, a2) = pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;

        // Calculate b coefficients
        self.b0 = ((1.0 - omega_cos) / 2.0) / a0;
        self.b1 = (1.0 - omega_cos) / a0;
        self.b2 = self.b0;
    }

    pub fn update_highpass(
        &mut self,
        cutoff: f64,
        q: f64,
        sample_rate: f64,
    ) {
        let (alpha, omega) = alpha_omega(sample_rate, cutoff, q);
        let omega_cos = omega.cos();
        let a0 = 1.0 + alpha;

        // Calculate a coefficients
        let (a1, a2) = pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;

        // Calculate b coefficients
        self.b0 = ((1.0 + omega_cos) / 2.0) / a0;
        self.b1 = -((1.0 + omega_cos)) / a0;
        self.b2 = self.b0;
    }

    pub fn update_bandpass(
        &mut self,
        cutoff: f64,
        q: f64,
        sample_rate: f64,
    ) {
        let (alpha, omega) = alpha_omega(sample_rate, cutoff, q);
        let a0 = 1.0 + alpha;

        // Calculate a coefficients
        let (a1, a2) = pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;

        // Calculate b coefficients
        self.b0 = alpha / a0;
        self.b1 = 0.0;
        self.b2 = -alpha / a0;
    }

    pub fn update_notch(
        &mut self,
        cutoff: f64,
        q: f64,
        sample_rate: f64,
    ) {
        let (alpha, omega) = alpha_omega(sample_rate, cutoff, q);
        let a0 = 1.0 + alpha;

        // Calculate a coefficients
        let (a1, a2) = pass_biquad_a_coefficients(alpha, omega);
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;

        // Calculate b coefficients
        self.b0 = 1.0 / a0;
        self.b1 = -(2.0 * omega.cos()) / a0;
        self.b2 = 1.0 / a0;
    }

    pub fn update_bell(
        &mut self,
        cutoff: f64,
        q: f64,
        gain: f64,
        sample_rate: f64,
    ) {
        let amplitude = decibel_to_amplitude(gain);
        let (alpha, omega) = alpha_omega(sample_rate, cutoff, q);
        let omega_cos = omega.cos();
        let a0 = 1.0 + alpha / amplitude;

        // Calculate a coefficients
        self.a1 = -(2.0 * omega_cos) / a0;
        self.a2 = (1.0 - alpha / amplitude) / a0;

        // Calculate b coefficients
        self.b0 = (1.0 + alpha * amplitude) / a0;
        self.b1 = -(2.0 * omega_cos) / a0;
        self.b2 = (1.0 - alpha * amplitude) / a0;
    }

    pub fn update_lowshelf(
        &mut self,
        cutoff: f64,
        gain: f64,
        sample_rate: f64,
    ) {
        let amplitude = decibel_to_amplitude(gain);
        let amp_sqrt = amplitude.sqrt();
        
        let omega = (2.0 * PI * cutoff) / sample_rate;
        let omega_cos = omega.cos();
        let alpha = omega.sin() / 2.0_f64.sqrt();

        let a0 = (amplitude + 1.0) + (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha;

        // Calculate a coefficients
        self.a1 = -(2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * omega_cos)) / a0;
        self.a2 = ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha) / a0;

        // Calculate b coefficients
        self.b0 = (amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha)) / a0;
        self.b1 = (2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * omega_cos)) / a0;
        self.b2 = (amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha)) / a0;
    }

    pub fn update_highshelf(
        &mut self,
        cutoff: f64,
        gain: f64,
        sample_rate: f64,
    ) {
        let amplitude = decibel_to_amplitude(gain);
        let amp_sqrt = amplitude.sqrt();

        let omega = (2.0 * PI * cutoff) / sample_rate;
        let omega_cos = omega.cos();
        let alpha = omega.sin() / 2.0_f64.sqrt();

        let a0 = (amplitude + 1.0) - (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha;

        // Calculate a coefficients
        self.a1 = (2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * omega_cos)) / a0;
        self.a2 = ((amplitude + 1.0) - (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha) / a0;

        // Calculate b coefficients
        self.b0 = (amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos + 2.0 * amp_sqrt * alpha)) / a0;
        self.b1 = -(2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * omega_cos)) / a0;
        self.b2 = (amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * omega_cos - 2.0 * amp_sqrt * alpha)) / a0;
    }

    // Exectution
    /// Processes an input sample and updates previous samples
    pub fn process<T>(&mut self, sample: &mut T)
    where
        T: Into<f64> + From<f64> + Copy,
    {
        // Temporary storage of current x2
        let temp_x2 = self.x2;

        // Update previous input samples
        self.x2 = self.x1;
        self.x1 = (*sample).into();

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
        self.y1 = (*sample).into();
    }

    /// Processes a slice of samples
    pub fn batch_process<T>(&mut self, samples: &mut [T])
    where
        T: Into<f64> + From<f64> + Copy,
    {
        for sample in samples {
            self.process(sample);
        }
    }
}

// ---- Functions ------------
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

/// Calculates decibel gain from linear gain
fn decibel_to_amplitude(gain: f64) -> f64 {
    let decibel = 20.0 * gain.log10();
    10.0_f64.powf(decibel / 40.0)
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
