//! Dataset loaders for training spiking neural networks

pub mod mnist;

pub use mnist::{MNISTDataset, MNISTImage, load_mnist_synthetic};
