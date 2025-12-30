//! Dataset loaders for training spiking neural networks

pub mod mnist;
pub mod text_dataset;

pub use mnist::{download_mnist, load_mnist_synthetic, MNISTDataset, MNISTImage};
pub use text_dataset::{
    EmotionTriggerJson, IntentRuleJson, JsonDataset, JsonDatasetStats, MetacognitionConfigJson,
    PragmaticRuleJson, SentenceTemplateJson, SentimentPatternCategory, SentimentPatterns,
    SupervisedPair, TextDataset, TextDatasetStats, VocabWord,
};
