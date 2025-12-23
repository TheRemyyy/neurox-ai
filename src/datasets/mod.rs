//! Dataset loaders for training spiking neural networks

pub mod mnist;
pub mod text_dataset;

pub use mnist::{MNISTDataset, MNISTImage, load_mnist_synthetic};
pub use text_dataset::{TextDataset, TextDatasetStats, JsonDataset, JsonDatasetStats, SupervisedPair, VocabWord, SentenceTemplateJson, PragmaticRuleJson, MetacognitionConfigJson, IntentRuleJson};
