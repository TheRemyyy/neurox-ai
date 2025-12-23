//! MNIST dataset loader for spiking neural networks
//!
//! Converts MNIST images to spike trains using rate coding

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

/// MNIST image (28×28 grayscale)
#[derive(Debug, Clone)]
pub struct MNISTImage {
    /// Pixel values (0-255)
    pub pixels: Vec<u8>,

    /// Label (0-9)
    pub label: u8,

    /// Image dimensions
    pub width: usize,
    pub height: usize,
}

impl MNISTImage {
    /// Convert to spike train using rate coding
    /// Higher pixel intensity = higher firing rate
    pub fn to_spike_train(&self, duration_ms: f32, dt: f32, max_rate_hz: f32) -> Vec<Vec<bool>> {
        let n_timesteps = (duration_ms / dt) as usize;
        let n_pixels = self.pixels.len();

        let mut spike_train = vec![vec![false; n_pixels]; n_timesteps];

        for (pixel_idx, &intensity) in self.pixels.iter().enumerate() {
            // Convert intensity to firing rate (0-255 → 0-max_rate_hz)
            let rate = (intensity as f32 / 255.0) * max_rate_hz;
            let spike_prob = rate * dt / 1000.0; // Probability per timestep

            // Generate Poisson spike train
            let mut rng = rand::thread_rng();
            for t in 0..n_timesteps {
                if rand::Rng::gen::<f32>(&mut rng) < spike_prob {
                    spike_train[t][pixel_idx] = true;
                }
            }
        }

        spike_train
    }

    /// Convert to input currents for direct neuron stimulation
    pub fn to_input_currents(&self, scale: f32) -> Vec<f32> {
        self.pixels
            .iter()
            .map(|&intensity| (intensity as f32 / 255.0) * scale)
            .collect()
    }
}

/// MNIST dataset
pub struct MNISTDataset {
    /// Training images
    pub train_images: Vec<MNISTImage>,

    /// Test images
    pub test_images: Vec<MNISTImage>,
}

impl MNISTDataset {
    /// Load MNIST from IDX format files
    pub fn load<P: AsRef<Path>>(data_dir: P) -> IoResult<Self> {
        let data_dir = data_dir.as_ref();

        log::info!("Loading MNIST dataset from {:?}", data_dir);

        let train_images = Self::load_images(
            data_dir.join("train-images-idx3-ubyte"),
            data_dir.join("train-labels-idx1-ubyte"),
        )?;

        let test_images = Self::load_images(
            data_dir.join("t10k-images-idx3-ubyte"),
            data_dir.join("t10k-labels-idx1-ubyte"),
        )?;

        log::info!("Loaded {} training images, {} test images",
                   train_images.len(), test_images.len());

        Ok(Self {
            train_images,
            test_images,
        })
    }

    /// Load images from IDX format
    fn load_images<P: AsRef<Path>>(image_path: P, label_path: P) -> IoResult<Vec<MNISTImage>> {
        // Load labels
        let mut label_file = File::open(label_path)?;
        let mut label_header = [0u8; 8];
        label_file.read_exact(&mut label_header)?;

        let n_labels = u32::from_be_bytes([label_header[4], label_header[5], label_header[6], label_header[7]]) as usize;

        let mut labels = vec![0u8; n_labels];
        label_file.read_exact(&mut labels)?;

        // Load images
        let mut image_file = File::open(image_path)?;
        let mut image_header = [0u8; 16];
        image_file.read_exact(&mut image_header)?;

        let n_images = u32::from_be_bytes([image_header[4], image_header[5], image_header[6], image_header[7]]) as usize;
        let height = u32::from_be_bytes([image_header[8], image_header[9], image_header[10], image_header[11]]) as usize;
        let width = u32::from_be_bytes([image_header[12], image_header[13], image_header[14], image_header[15]]) as usize;

        let pixels_per_image = width * height;
        let mut all_pixels = vec![0u8; n_images * pixels_per_image];
        image_file.read_exact(&mut all_pixels)?;

        // Create images
        let mut images = Vec::with_capacity(n_images);
        for i in 0..n_images {
            let start = i * pixels_per_image;
            let end = start + pixels_per_image;
            let pixels = all_pixels[start..end].to_vec();

            images.push(MNISTImage {
                pixels,
                label: labels[i],
                width,
                height,
            });
        }

        Ok(images)
    }

    /// Get training batch
    pub fn get_train_batch(&self, batch_idx: usize, batch_size: usize) -> &[MNISTImage] {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.train_images.len());
        &self.train_images[start..end]
    }

    /// Get test batch
    pub fn get_test_batch(&self, batch_idx: usize, batch_size: usize) -> &[MNISTImage] {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(self.test_images.len());
        &self.test_images[start..end]
    }

    /// Number of training batches
    pub fn n_train_batches(&self, batch_size: usize) -> usize {
        (self.train_images.len() + batch_size - 1) / batch_size
    }

    /// Number of test batches
    pub fn n_test_batches(&self, batch_size: usize) -> usize {
        (self.test_images.len() + batch_size - 1) / batch_size
    }
}

/// Simple in-memory MNIST loader (for testing without files)
pub fn load_mnist_synthetic(n_train: usize, n_test: usize) -> MNISTDataset {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut train_images = Vec::with_capacity(n_train);
    let mut test_images = Vec::with_capacity(n_test);

    // Generate synthetic MNIST-like data
    for _ in 0..n_train {
        let label = rng.gen_range(0..10);
        let pixels = (0..784).map(|_| rng.gen_range(0..255)).collect();
        train_images.push(MNISTImage {
            pixels,
            label,
            width: 28,
            height: 28,
        });
    }

    for _ in 0..n_test {
        let label = rng.gen_range(0..10);
        let pixels = (0..784).map(|_| rng.gen_range(0..255)).collect();
        test_images.push(MNISTImage {
            pixels,
            label,
            width: 28,
            height: 28,
        });
    }

    MNISTDataset {
        train_images,
        test_images,
    }
}
