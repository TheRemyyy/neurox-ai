//! MNIST dataset loader for spiking neural networks
//!
//! Converts MNIST images to spike trains using rate coding
//! Supports automatic download from http://yann.lecun.com/exdb/mnist/

use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{Read, Result as IoResult, Write};
use std::path::Path;

/// MNIST download URLs (using PyTorch mirror - original LeCun server moved)
const MNIST_BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist/";
const MNIST_FILES: [(&str, &str); 4] = [
    ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
    ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
    ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
    ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte"),
];

/// Download MNIST dataset to specified directory
///
/// Downloads all 4 files (train images, train labels, test images, test labels)
/// and decompresses them from .gz format.
///
/// Returns the path to the download directory.
pub fn download_mnist<P: AsRef<Path>>(target_dir: P) -> Result<String, Box<dyn std::error::Error>> {
    let target_dir = target_dir.as_ref();

    // Create directory if it doesn't exist
    fs::create_dir_all(target_dir)?;

    println!("Downloading MNIST dataset to {:?}...", target_dir);

    for (gz_name, raw_name) in MNIST_FILES.iter() {
        let output_path = target_dir.join(raw_name);

        // Skip if already downloaded
        if output_path.exists() {
            println!("  ✓ {} already exists", raw_name);
            continue;
        }

        let url = format!("{}{}", MNIST_BASE_URL, gz_name);
        println!("  Downloading {}...", gz_name);

        // Download file
        let response = ureq::get(&url)
            .call()
            .map_err(|e| format!("Failed to download {}: {}", gz_name, e))?;

        // Read response body
        let mut compressed_data = Vec::new();
        response.into_reader().read_to_end(&mut compressed_data)?;

        // Decompress gzip
        let mut decoder = GzDecoder::new(&compressed_data[..]);
        let mut decompressed_data = Vec::new();
        decoder.read_to_end(&mut decompressed_data)?;

        // Write to file
        let mut file = File::create(&output_path)?;
        file.write_all(&decompressed_data)?;

        println!(
            "  ✓ {} downloaded ({} bytes)",
            raw_name,
            decompressed_data.len()
        );
    }

    println!("MNIST download complete!");

    Ok(target_dir.to_string_lossy().to_string())
}

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
    /// All images (combined view for simpler API)
    pub images: Vec<MNISTImage>,

    /// Training images
    pub train_images: Vec<MNISTImage>,

    /// Test images
    pub test_images: Vec<MNISTImage>,
}

impl MNISTDataset {
    /// Load MNIST from IDX format files
    pub fn load_from_dir<P: AsRef<Path>>(data_dir: P) -> IoResult<Self> {
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

        log::info!(
            "Loaded {} training images, {} test images",
            train_images.len(),
            test_images.len()
        );

        let images = train_images.clone();

        Ok(Self {
            images,
            train_images,
            test_images,
        })
    }

    /// Load MNIST from separate image and label files
    pub fn load<P: AsRef<Path>>(image_path: P, label_path: P) -> IoResult<Self> {
        let images = Self::load_images(image_path, label_path)?;

        Ok(Self {
            images: images.clone(),
            train_images: images.clone(),
            test_images: images,
        })
    }

    /// Generate synthetic MNIST-like data for testing
    pub fn generate_synthetic(n_samples: usize) -> Vec<MNISTImage> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut images = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let label = rng.gen_range(0..10);

            // Generate digit-like patterns (not random noise)
            let mut pixels = vec![0u8; 784];

            // Create a simple pattern based on the digit
            let center_x = 14;
            let center_y = 14;

            match label {
                0 => {
                    // Circle (Center)
                    for y in 0..28 {
                        for x in 0..28 {
                            let dx = (x as i32 - center_x) as f32;
                            let dy = (y as i32 - center_y) as f32;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if (dist - 8.0).abs() < 2.0 {
                                pixels[y * 28 + x] = 255;
                            }
                        }
                    }
                }
                1 => {
                    // Vertical Line (Center)
                    for y in 4..24 {
                        for x in 13..15 {
                            pixels[y * 28 + x] = 255;
                        }
                    }
                }
                2 => {
                    // Horizontal Line (Center)
                    for y in 13..15 {
                        for x in 4..24 {
                            pixels[y * 28 + x] = 255;
                        }
                    }
                }
                3 => {
                    // Diagonal /
                    for i in 4..24 {
                        let x = i;
                        let y = 27 - i;
                        if x < 28 && y < 28 {
                            pixels[y * 28 + x] = 255;
                            pixels[y * 28 + x + 1] = 255;
                        }
                    }
                }
                4 => {
                    // Diagonal \
                    for i in 4..24 {
                        let x = i;
                        let y = i;
                        if x < 28 && y < 28 {
                            pixels[y * 28 + x] = 255;
                            pixels[y * 28 + x + 1] = 255;
                        }
                    }
                }
                5 => {
                    // Cross +
                    for i in 4..24 {
                        // V
                        pixels[i * 28 + 14] = 255;
                        pixels[i * 28 + 15] = 255;
                        // H
                        pixels[14 * 28 + i] = 255;
                        pixels[15 * 28 + i] = 255;
                    }
                }
                6 => {
                    // Square Box
                    for y in 8..20 {
                        for x in 8..20 {
                            if x == 8 || x == 19 || y == 8 || y == 19 {
                                pixels[y * 28 + x] = 255;
                            }
                        }
                    }
                }
                7 => {
                    // Triangle (Up)
                    // (14, 4) to (4, 24) and (24, 24)
                    for y in 4..24 {
                        let target_x1 = 14 - (y - 4) / 2;
                        let target_x2 = 14 + (y - 4) / 2;
                        if target_x1 < 28 {
                            pixels[y * 28 + target_x1] = 255;
                        }
                        if target_x2 < 28 {
                            pixels[y * 28 + target_x2] = 255;
                        }
                        if y == 23 {
                            for x in 4..25 {
                                pixels[y * 28 + x] = 255;
                            }
                        }
                    }
                }
                8 => {
                    // X Shape (Cross) - Distinct from +
                    for i in 4..24 {
                        pixels[i * 28 + i] = 255; // \
                        pixels[(27 - i) * 28 + i] = 255; // /
                    }
                }
                9 => {
                    // Solid Block (Fill)
                    for y in 10..18 {
                        for x in 10..18 {
                            pixels[y * 28 + x] = 255;
                        }
                    }
                }
                _ => {}
            }

            // Add some noise
            for p in pixels.iter_mut() {
                if *p == 0 && rng.gen::<f32>() < 0.05 {
                    *p = rng.gen_range(20..80);
                }
            }

            images.push(MNISTImage {
                pixels,
                label,
                width: 28,
                height: 28,
            });
        }

        images
    }

    /// Load images from IDX format
    fn load_images<P: AsRef<Path>>(image_path: P, label_path: P) -> IoResult<Vec<MNISTImage>> {
        // Load labels
        let mut label_file = File::open(label_path)?;
        let mut label_header = [0u8; 8];
        label_file.read_exact(&mut label_header)?;

        let n_labels = u32::from_be_bytes([
            label_header[4],
            label_header[5],
            label_header[6],
            label_header[7],
        ]) as usize;

        let mut labels = vec![0u8; n_labels];
        label_file.read_exact(&mut labels)?;

        // Load images
        let mut image_file = File::open(image_path)?;
        let mut image_header = [0u8; 16];
        image_file.read_exact(&mut image_header)?;

        let n_images = u32::from_be_bytes([
            image_header[4],
            image_header[5],
            image_header[6],
            image_header[7],
        ]) as usize;
        let height = u32::from_be_bytes([
            image_header[8],
            image_header[9],
            image_header[10],
            image_header[11],
        ]) as usize;
        let width = u32::from_be_bytes([
            image_header[12],
            image_header[13],
            image_header[14],
            image_header[15],
        ]) as usize;

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
    let train_images = MNISTDataset::generate_synthetic(n_train);
    let test_images = MNISTDataset::generate_synthetic(n_test);

    // Clone training images for the 'images' field (legacy compatibility)
    let images = train_images.clone();

    MNISTDataset {
        images,
        train_images,
        test_images,
    }
}
