//! Metacognition - Thinking About Thinking
//!
//! Self-monitoring and regulation of cognitive processes.
//! Enables the AI to evaluate its own reasoning and adjust strategies.
//!
//! # Features
//! - Confidence monitoring
//! - Error detection and correction
//! - Learning strategy selection
//! - Cognitive resource allocation
//!
//! # References
//! - Metacognitive AI architectures (2024-2025)
//! - Self-referential learning in neural networks

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Confidence level for a cognitive process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    VeryLow,  // < 0.2
    Low,      // 0.2 - 0.4
    Moderate, // 0.4 - 0.6
    High,     // 0.6 - 0.8
    VeryHigh, // > 0.8
}

impl ConfidenceLevel {
    pub fn from_value(v: f32) -> Self {
        if v < 0.2 {
            ConfidenceLevel::VeryLow
        } else if v < 0.4 {
            ConfidenceLevel::Low
        } else if v < 0.6 {
            ConfidenceLevel::Moderate
        } else if v < 0.8 {
            ConfidenceLevel::High
        } else {
            ConfidenceLevel::VeryHigh
        }
    }

    pub fn to_value(&self) -> f32 {
        match self {
            ConfidenceLevel::VeryLow => 0.1,
            ConfidenceLevel::Low => 0.3,
            ConfidenceLevel::Moderate => 0.5,
            ConfidenceLevel::High => 0.7,
            ConfidenceLevel::VeryHigh => 0.9,
        }
    }
}

/// Cognitive strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveStrategy {
    /// Quick, intuitive response
    FastIntuitive,
    /// Careful, analytical processing
    SlowAnalytical,
    /// Pattern matching from memory
    MemoryRetrieval,
    /// Step-by-step reasoning
    ChainOfThought,
    /// Seeking external information
    InformationSeeking,
    /// Decomposing into subproblems
    ProblemDecomposition,
    /// Using analogies
    AnalogicalReasoning,
}

impl CognitiveStrategy {
    /// Estimated cognitive cost (0.0 to 1.0)
    pub fn cost(&self) -> f32 {
        match self {
            CognitiveStrategy::FastIntuitive => 0.1,
            CognitiveStrategy::MemoryRetrieval => 0.2,
            CognitiveStrategy::SlowAnalytical => 0.5,
            CognitiveStrategy::ChainOfThought => 0.6,
            CognitiveStrategy::AnalogicalReasoning => 0.4,
            CognitiveStrategy::ProblemDecomposition => 0.7,
            CognitiveStrategy::InformationSeeking => 0.8,
        }
    }

    /// Expected processing time (relative)
    pub fn time_estimate(&self) -> f32 {
        match self {
            CognitiveStrategy::FastIntuitive => 1.0,
            CognitiveStrategy::MemoryRetrieval => 2.0,
            CognitiveStrategy::AnalogicalReasoning => 3.0,
            CognitiveStrategy::SlowAnalytical => 4.0,
            CognitiveStrategy::ChainOfThought => 5.0,
            CognitiveStrategy::ProblemDecomposition => 6.0,
            CognitiveStrategy::InformationSeeking => 8.0,
        }
    }
}

/// Cognitive process being monitored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveProcess {
    /// Process name
    pub name: String,
    /// Current confidence
    pub confidence: f32,
    /// Uncertainty (inverse of confidence)
    pub uncertainty: f32,
    /// Currently active strategy
    pub strategy: CognitiveStrategy,
    /// Processing progress (0.0 to 1.0)
    pub progress: f32,
    /// Detected errors/issues
    pub errors: Vec<String>,
    /// Whether this process succeeded
    pub completed: Option<bool>,
}

impl CognitiveProcess {
    pub fn new(name: String, strategy: CognitiveStrategy) -> Self {
        Self {
            name,
            confidence: 0.5,
            uncertainty: 0.5,
            strategy,
            progress: 0.0,
            errors: Vec::new(),
            completed: None,
        }
    }

    /// Update confidence
    pub fn update_confidence(&mut self, delta: f32) {
        self.confidence = (self.confidence + delta).clamp(0.0, 1.0);
        self.uncertainty = 1.0 - self.confidence;
    }

    /// Report an error
    pub fn report_error(&mut self, error: String) {
        self.errors.push(error);
        self.confidence *= 0.8; // Reduce confidence on error
        self.uncertainty = 1.0 - self.confidence;
    }

    /// Mark as complete
    pub fn complete(&mut self, success: bool) {
        self.completed = Some(success);
        self.progress = 1.0;
    }
}

/// Strategy performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRecord {
    pub strategy: CognitiveStrategy,
    pub success_count: u32,
    pub failure_count: u32,
    pub avg_confidence: f32,
    pub avg_time: f32,
}

impl StrategyRecord {
    pub fn new(strategy: CognitiveStrategy) -> Self {
        Self {
            strategy,
            success_count: 0,
            failure_count: 0,
            avg_confidence: 0.5,
            avg_time: strategy.time_estimate(),
        }
    }

    /// Record an outcome
    pub fn record(&mut self, success: bool, confidence: f32, time: f32) {
        if success {
            self.success_count += 1;
        } else {
            self.failure_count += 1;
        }

        let n = (self.success_count + self.failure_count) as f32;
        self.avg_confidence = (self.avg_confidence * (n - 1.0) + confidence) / n;
        self.avg_time = (self.avg_time * (n - 1.0) + time) / n;
    }

    /// Success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.5
        } else {
            self.success_count as f32 / total as f32
        }
    }
}

/// Metacognition system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metacognition {
    /// Current cognitive processes
    pub active_processes: Vec<CognitiveProcess>,
    /// Strategy performance history
    pub strategy_records: Vec<StrategyRecord>,
    /// Confidence history
    pub confidence_history: VecDeque<f32>,
    /// Error history
    pub error_history: VecDeque<String>,
    /// Global cognitive load (0.0 to 1.0)
    pub cognitive_load: f32,
    /// Maximum history size
    pub max_history: usize,
    /// Know-don't-know threshold
    pub know_threshold: f32,
}

impl Default for Metacognition {
    fn default() -> Self {
        Self::new()
    }
}

impl Metacognition {
    pub fn new() -> Self {
        let strategies = vec![
            CognitiveStrategy::FastIntuitive,
            CognitiveStrategy::SlowAnalytical,
            CognitiveStrategy::MemoryRetrieval,
            CognitiveStrategy::ChainOfThought,
            CognitiveStrategy::InformationSeeking,
            CognitiveStrategy::ProblemDecomposition,
            CognitiveStrategy::AnalogicalReasoning,
        ];

        Self {
            active_processes: Vec::new(),
            strategy_records: strategies.into_iter().map(StrategyRecord::new).collect(),
            confidence_history: VecDeque::new(),
            error_history: VecDeque::new(),
            cognitive_load: 0.0,
            max_history: 100,
            know_threshold: 0.6,
        }
    }

    /// Start monitoring a new cognitive process
    pub fn start_process(&mut self, name: String, strategy: CognitiveStrategy) -> usize {
        let process = CognitiveProcess::new(name, strategy);
        self.cognitive_load += strategy.cost();
        self.active_processes.push(process);
        self.active_processes.len() - 1
    }

    /// Update a process
    pub fn update_process(&mut self, idx: usize, confidence_delta: f32, progress: f32) {
        if let Some(process) = self.active_processes.get_mut(idx) {
            process.update_confidence(confidence_delta);
            process.progress = progress.clamp(0.0, 1.0);

            // Track confidence
            self.confidence_history.push_back(process.confidence);
            if self.confidence_history.len() > self.max_history {
                self.confidence_history.pop_front();
            }
        }
    }

    /// Report error in a process
    pub fn report_error(&mut self, idx: usize, error: String) {
        if let Some(process) = self.active_processes.get_mut(idx) {
            process.report_error(error.clone());

            self.error_history.push_back(error);
            if self.error_history.len() > self.max_history {
                self.error_history.pop_front();
            }
        }
    }

    /// Complete a process
    pub fn complete_process(&mut self, idx: usize, success: bool) {
        if let Some(process) = self.active_processes.get_mut(idx) {
            process.complete(success);
            self.cognitive_load -= process.strategy.cost();
            self.cognitive_load = self.cognitive_load.max(0.0);

            // Update strategy record
            if let Some(record) = self
                .strategy_records
                .iter_mut()
                .find(|r| r.strategy == process.strategy)
            {
                // Estimate actual time from strategy base time and progress
                let estimated_time = process.strategy.time_estimate() * process.progress.max(0.1);
                record.record(success, process.confidence, estimated_time);
            }
        }

        // Remove completed processes
        self.active_processes.retain(|p| p.completed.is_none());
    }

    /// Check if we "know" something (have sufficient confidence)
    pub fn knows(&self, confidence: f32) -> bool {
        confidence >= self.know_threshold
    }

    /// Check if we're uncertain about something
    pub fn uncertain(&self, confidence: f32) -> bool {
        confidence < self.know_threshold && confidence >= 0.3
    }

    /// Check if we "don't know" (very low confidence)
    pub fn dont_know(&self, confidence: f32) -> bool {
        confidence < 0.3
    }

    /// Select best strategy for a task
    pub fn select_strategy(&self, task_complexity: f32, time_available: f32) -> CognitiveStrategy {
        // Filter strategies by time constraint
        let viable: Vec<&StrategyRecord> = self
            .strategy_records
            .iter()
            .filter(|r| r.avg_time <= time_available)
            .collect();

        if viable.is_empty() {
            return CognitiveStrategy::FastIntuitive; // Fallback
        }

        // Score strategies
        let mut best = CognitiveStrategy::FastIntuitive;
        let mut best_score = 0.0;

        for record in viable {
            // Score = success_rate * complexity_match - cost
            let complexity_match = if task_complexity > 0.5 {
                // Complex task: prefer analytical strategies
                match record.strategy {
                    CognitiveStrategy::SlowAnalytical
                    | CognitiveStrategy::ChainOfThought
                    | CognitiveStrategy::ProblemDecomposition => 1.0,
                    _ => 0.5,
                }
            } else {
                // Simple task: prefer fast strategies
                match record.strategy {
                    CognitiveStrategy::FastIntuitive | CognitiveStrategy::MemoryRetrieval => 1.0,
                    _ => 0.5,
                }
            };

            let score = record.success_rate() * complexity_match - record.strategy.cost() * 0.3;

            if score > best_score {
                best_score = score;
                best = record.strategy;
            }
        }

        best
    }

    /// Should we switch strategies?
    pub fn should_switch_strategy(&self, current_confidence: f32, progress: f32) -> bool {
        // Switch if making poor progress with low confidence
        if progress > 0.3 && current_confidence < 0.3 {
            return true;
        }

        // Switch if stalled
        if progress > 0.5 && progress < 0.6 && current_confidence < 0.5 {
            return true;
        }

        false
    }

    /// Get metacognitive assessment
    pub fn assess(&self) -> MetacognitiveAssessment {
        let avg_confidence = if self.confidence_history.is_empty() {
            0.5
        } else {
            self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32
        };

        let error_rate = self.error_history.len() as f32 / self.max_history as f32;

        MetacognitiveAssessment {
            overall_confidence: avg_confidence,
            confidence_level: ConfidenceLevel::from_value(avg_confidence),
            cognitive_load: self.cognitive_load,
            error_rate,
            active_processes: self.active_processes.len(),
            recommended_action: if self.cognitive_load > 0.8 {
                RecommendedAction::ReduceLoad
            } else if error_rate > 0.3 {
                RecommendedAction::SlowDown
            } else if avg_confidence < 0.3 {
                RecommendedAction::SeekInformation
            } else {
                RecommendedAction::Continue
            },
        }
    }

    /// Get statistics
    pub fn stats(&self) -> MetacognitionStats {
        MetacognitionStats {
            active_processes: self.active_processes.len(),
            cognitive_load: self.cognitive_load,
            avg_confidence: self.confidence_history.iter().sum::<f32>()
                / self.confidence_history.len().max(1) as f32,
            total_errors: self.error_history.len(),
        }
    }
}

/// Recommended action based on metacognitive assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    Continue,
    SlowDown,
    SpeedUp,
    SeekInformation,
    ReduceLoad,
    SwitchStrategy,
}

/// Metacognitive assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveAssessment {
    pub overall_confidence: f32,
    pub confidence_level: ConfidenceLevel,
    pub cognitive_load: f32,
    pub error_rate: f32,
    pub active_processes: usize,
    pub recommended_action: RecommendedAction,
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionStats {
    pub active_processes: usize,
    pub cognitive_load: f32,
    pub avg_confidence: f32,
    pub total_errors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_level() {
        assert_eq!(ConfidenceLevel::from_value(0.1), ConfidenceLevel::VeryLow);
        assert_eq!(ConfidenceLevel::from_value(0.5), ConfidenceLevel::Moderate);
        assert_eq!(ConfidenceLevel::from_value(0.9), ConfidenceLevel::VeryHigh);
    }

    #[test]
    fn test_process_lifecycle() {
        let mut meta = Metacognition::new();

        let idx = meta.start_process("Test".into(), CognitiveStrategy::SlowAnalytical);
        assert_eq!(meta.active_processes.len(), 1);

        meta.update_process(idx, 0.1, 0.5);
        assert!(meta.active_processes[idx].confidence > 0.5);

        meta.complete_process(idx, true);
        assert_eq!(meta.active_processes.len(), 0);
    }

    #[test]
    fn test_know_dont_know() {
        let meta = Metacognition::new();

        assert!(meta.knows(0.7));
        assert!(!meta.knows(0.3));
        assert!(meta.dont_know(0.2));
        assert!(meta.uncertain(0.4));
    }

    #[test]
    fn test_strategy_selection() {
        let meta = Metacognition::new();

        // Complex task with lots of time
        let strategy = meta.select_strategy(0.8, 10.0);
        assert!(matches!(
            strategy,
            CognitiveStrategy::SlowAnalytical
                | CognitiveStrategy::ChainOfThought
                | CognitiveStrategy::ProblemDecomposition
        ));

        // Simple task with little time
        let strategy = meta.select_strategy(0.2, 2.0);
        assert!(matches!(
            strategy,
            CognitiveStrategy::FastIntuitive | CognitiveStrategy::MemoryRetrieval
        ));
    }

    #[test]
    fn test_error_tracking() {
        let mut meta = Metacognition::new();

        let idx = meta.start_process("Test".into(), CognitiveStrategy::FastIntuitive);
        meta.report_error(idx, "Something went wrong".into());

        assert_eq!(meta.error_history.len(), 1);
        assert!(meta.active_processes[idx].confidence < 0.5);
    }

    #[test]
    fn test_assessment() {
        let meta = Metacognition::new();
        let assessment = meta.assess();

        assert!(assessment.overall_confidence >= 0.0 && assessment.overall_confidence <= 1.0);
        assert!(assessment.cognitive_load >= 0.0);
    }
}
