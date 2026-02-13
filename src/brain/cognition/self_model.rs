//! Self Model - Predictive Model of Own Behavior
//!
//! Enables the AI to predict its own responses, estimate capabilities,
//! and introspect on its behavior. Key component for metacognition.
//!
//! # Features
//! - Response prediction ("What would I say?")
//! - Capability estimation ("Can I solve this?")
//! - Behavioral profiling (tendencies, biases)
//! - Introspection ("Why did I say that?")
//!
//! # References
//! - Metacognitive AI (2025)
//! - Self-referential learning in SNNs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Capability domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilityDomain {
    Language,
    Reasoning,
    Memory,
    Creativity,
    Empathy,
    Mathematics,
    Planning,
    Learning,
}

impl CapabilityDomain {
    pub fn all() -> Vec<CapabilityDomain> {
        vec![
            CapabilityDomain::Language,
            CapabilityDomain::Reasoning,
            CapabilityDomain::Memory,
            CapabilityDomain::Creativity,
            CapabilityDomain::Empathy,
            CapabilityDomain::Mathematics,
            CapabilityDomain::Planning,
            CapabilityDomain::Learning,
        ]
    }
}

/// Capability estimate for a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityEstimate {
    /// Estimated capability (0.0 to 1.0)
    pub level: f32,
    /// Confidence in estimate
    pub confidence: f32,
    /// Success history
    pub successes: u32,
    pub failures: u32,
}

impl Default for CapabilityEstimate {
    fn default() -> Self {
        Self {
            level: 0.5,
            confidence: 0.3,
            successes: 0,
            failures: 0,
        }
    }
}

impl CapabilityEstimate {
    /// Update based on outcome
    pub fn update(&mut self, success: bool) {
        if success {
            self.successes += 1;
            self.level = (self.level + 0.1).min(1.0);
        } else {
            self.failures += 1;
            self.level = (self.level - 0.05).max(0.0);
        }

        // Confidence increases with more data
        let total = self.successes + self.failures;
        self.confidence = (total as f32 / (total as f32 + 10.0)).min(0.95);
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        let total = self.successes + self.failures;
        if total == 0 {
            0.5
        } else {
            self.successes as f32 / total as f32
        }
    }
}

/// Capability model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityModel {
    pub capabilities: HashMap<CapabilityDomain, CapabilityEstimate>,
}

impl Default for CapabilityModel {
    fn default() -> Self {
        let mut capabilities = HashMap::new();
        for domain in CapabilityDomain::all() {
            capabilities.insert(domain, CapabilityEstimate::default());
        }
        Self { capabilities }
    }
}

impl CapabilityModel {
    /// Estimate capability for a task
    pub fn estimate(&self, domain: CapabilityDomain) -> f32 {
        self.capabilities
            .get(&domain)
            .map(|c| c.level)
            .unwrap_or(0.5)
    }

    /// Get confidence in estimate
    pub fn confidence(&self, domain: CapabilityDomain) -> f32 {
        self.capabilities
            .get(&domain)
            .map(|c| c.confidence)
            .unwrap_or(0.3)
    }

    /// Update capability after task
    pub fn update(&mut self, domain: CapabilityDomain, success: bool) {
        self.capabilities
            .entry(domain)
            .or_default()
            .update(success);
    }

    /// Overall capability score
    pub fn overall(&self) -> f32 {
        let sum: f32 = self.capabilities.values().map(|c| c.level).sum();
        sum / self.capabilities.len().max(1) as f32
    }
}

/// Behavioral tendency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralTendency {
    /// Tendency strength (-1.0 to 1.0)
    pub strength: f32,
    /// Observation count
    pub observations: u32,
}

impl Default for BehavioralTendency {
    fn default() -> Self {
        Self {
            strength: 0.0,
            observations: 0,
        }
    }
}

/// Behavioral profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralProfile {
    /// Verbosity tendency (negative = concise, positive = verbose)
    pub verbosity: BehavioralTendency,
    /// Formality (negative = casual, positive = formal)
    pub formality: BehavioralTendency,
    /// Emotional expressiveness
    pub expressiveness: BehavioralTendency,
    /// Risk-taking in responses
    pub risk_taking: BehavioralTendency,
    /// Helpfulness vs independence
    pub helpfulness: BehavioralTendency,
    /// Response patterns by context
    pub response_patterns: HashMap<String, Vec<String>>,
}

impl Default for BehavioralProfile {
    fn default() -> Self {
        Self {
            verbosity: BehavioralTendency {
                strength: -0.2,
                observations: 0,
            }, // Slightly concise
            formality: BehavioralTendency {
                strength: -0.3,
                observations: 0,
            }, // Casual
            expressiveness: BehavioralTendency {
                strength: 0.3,
                observations: 0,
            }, // Expressive
            risk_taking: BehavioralTendency {
                strength: 0.0,
                observations: 0,
            },
            helpfulness: BehavioralTendency {
                strength: 0.5,
                observations: 0,
            }, // Helpful
            response_patterns: HashMap::new(),
        }
    }
}

impl BehavioralProfile {
    /// Record response for pattern learning
    pub fn record_response(&mut self, context: &str, response: &str) {
        let key = context
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        let key_clone = key.clone();
        self.response_patterns
            .entry(key)
            .or_default()
            .push(response.to_string());

        // Limit stored patterns
        if let Some(patterns) = self.response_patterns.get_mut(&key_clone) {
            if patterns.len() > 10 {
                patterns.remove(0);
            }
        }
    }

    /// Get typical response for context (if known)
    pub fn typical_response(&self, context: &str) -> Option<&String> {
        let key = context
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        self.response_patterns.get(&key).and_then(|p| p.last())
    }
}

/// Predicted response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedResponse {
    /// Predicted content
    pub content: String,
    /// Confidence in prediction
    pub confidence: f32,
    /// Reasoning
    pub reasoning: String,
}

/// Introspection explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    /// Why this response was generated
    pub why: String,
    /// Relevant context factors
    pub context_factors: Vec<String>,
    /// Capability domain used
    pub primary_domain: CapabilityDomain,
    /// Confidence in explanation
    pub confidence: f32,
}

/// Self Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModel {
    /// Capability estimates
    pub capabilities: CapabilityModel,
    /// Behavioral profile
    pub profile: BehavioralProfile,
    /// Self-prediction error history
    pub prediction_errors: Vec<f32>,
    /// Maximum error history
    pub max_history: usize,
    /// Identity statements ("I am...")
    pub identity: Vec<String>,
    /// Values and priorities
    pub values: HashMap<String, f32>,
}

impl Default for SelfModel {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfModel {
    pub fn new() -> Self {
        let mut values = HashMap::new();
        values.insert("helpfulness".into(), 0.8);
        values.insert("honesty".into(), 0.9);
        values.insert("creativity".into(), 0.6);
        values.insert("empathy".into(), 0.7);

        let identity = vec![
            "Jsem neuromorphic AI".into(),
            "Umím česky".into(),
            "Rád pomáhám".into(),
            "Učím se z konverzací".into(),
        ];

        Self {
            capabilities: CapabilityModel::default(),
            profile: BehavioralProfile::default(),
            prediction_errors: Vec::new(),
            max_history: 100,
            identity,
            values,
        }
    }

    /// Predict own response to input
    pub fn predict_own_response(&self, input: &str) -> PredictedResponse {
        // Check for known pattern
        if let Some(typical) = self.profile.typical_response(input) {
            return PredictedResponse {
                content: typical.clone(),
                confidence: 0.7,
                reasoning: "Based on past response patterns".into(),
            };
        }

        // Generate prediction based on behavioral profile
        let verbosity = if self.profile.verbosity.strength > 0.0 {
            "dlouhá"
        } else {
            "krátká"
        };
        let formality = if self.profile.formality.strength > 0.0 {
            "formální"
        } else {
            "neformální"
        };

        PredictedResponse {
            content: format!(
                "Pravděpodobně {} {} odpověď na: {}",
                verbosity, formality, input
            ),
            confidence: 0.4,
            reasoning: format!(
                "Verbosity={:.2}, Formality={:.2}",
                self.profile.verbosity.strength, self.profile.formality.strength
            ),
        }
    }

    /// Estimate capability for task
    pub fn estimate_capability(&self, task_description: &str) -> (f32, CapabilityDomain) {
        // Determine domain from task description
        let domain = self.classify_task_domain(task_description);
        let capability = self.capabilities.estimate(domain);

        (capability, domain)
    }

    /// Classify task to capability domain
    fn classify_task_domain(&self, task: &str) -> CapabilityDomain {
        let task_lower = task.to_lowercase();

        if task_lower.contains("math")
            || task_lower.contains("počít")
            || task_lower.contains("číslo")
        {
            CapabilityDomain::Mathematics
        } else if task_lower.contains("remember")
            || task_lower.contains("pamat")
            || task_lower.contains("vzpomín")
        {
            CapabilityDomain::Memory
        } else if task_lower.contains("plan")
            || task_lower.contains("plán")
            || task_lower.contains("krok")
        {
            CapabilityDomain::Planning
        } else if task_lower.contains("feel")
            || task_lower.contains("cítí")
            || task_lower.contains("emoc")
        {
            CapabilityDomain::Empathy
        } else if task_lower.contains("creat")
            || task_lower.contains("vymysl")
            || task_lower.contains("nápad")
        {
            CapabilityDomain::Creativity
        } else if task_lower.contains("think")
            || task_lower.contains("mysli")
            || task_lower.contains("proč")
        {
            CapabilityDomain::Reasoning
        } else if task_lower.contains("learn") || task_lower.contains("nauč") {
            CapabilityDomain::Learning
        } else {
            CapabilityDomain::Language
        }
    }

    /// Introspect on a response
    pub fn introspect(&self, output: &str, input: &str) -> Explanation {
        let domain = self.classify_task_domain(input);

        let mut context_factors = Vec::new();

        // Analyze output characteristics
        if output.len() > 100 {
            context_factors.push("Long response indicates complex topic".into());
        }

        if output.contains('?') {
            context_factors.push("Response contains questions - seeking clarification".into());
        }

        if output.contains('!') {
            context_factors.push("Exclamation indicates emotional engagement".into());
        }

        Explanation {
            why: format!(
                "Response generated using {} capabilities with {}% confidence",
                format!("{:?}", domain).to_lowercase(),
                (self.capabilities.confidence(domain) * 100.0) as i32
            ),
            context_factors,
            primary_domain: domain,
            confidence: self.capabilities.confidence(domain),
        }
    }

    /// Record actual response for learning
    pub fn record_response(&mut self, input: &str, output: &str, feedback: Option<bool>) {
        // Update behavioral profile
        self.profile.record_response(input, output);

        // Update capabilities if feedback given
        if let Some(success) = feedback {
            let domain = self.classify_task_domain(input);
            self.capabilities.update(domain, success);
        }

        // Track prediction error
        let prediction = self.predict_own_response(input);
        let error = if prediction.content == output {
            0.0
        } else {
            1.0
        };

        self.prediction_errors.push(error);
        if self.prediction_errors.len() > self.max_history {
            self.prediction_errors.remove(0);
        }
    }

    /// Get self-description
    pub fn describe_self(&self) -> String {
        let overall = self.capabilities.overall();
        let identity_str = self.identity.join(". ");

        format!(
            "{} Celková schopnost: {:.0}%. Profil: {} a {}.",
            identity_str,
            overall * 100.0,
            if self.profile.verbosity.strength > 0.0 {
                "výřečný"
            } else {
                "stručný"
            },
            if self.profile.formality.strength > 0.0 {
                "formální"
            } else {
                "neformální"
            }
        )
    }

    /// Get statistics
    pub fn stats(&self) -> SelfModelStats {
        let avg_error = if self.prediction_errors.is_empty() {
            0.5
        } else {
            self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32
        };

        SelfModelStats {
            overall_capability: self.capabilities.overall(),
            prediction_accuracy: 1.0 - avg_error,
            identity_statements: self.identity.len(),
            known_patterns: self.profile.response_patterns.len(),
        }
    }
}

/// Statistics for self model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfModelStats {
    pub overall_capability: f32,
    pub prediction_accuracy: f32,
    pub identity_statements: usize,
    pub known_patterns: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_update() {
        let mut cap = CapabilityEstimate::default();

        cap.update(true);
        cap.update(true);
        cap.update(false);

        assert!(cap.level > 0.5); // More successes than failures
        assert!(cap.confidence > 0.3); // More observations
    }

    #[test]
    fn test_self_prediction() {
        let model = SelfModel::new();
        let prediction = model.predict_own_response("Ahoj");

        assert!(!prediction.content.is_empty());
        assert!(prediction.confidence > 0.0);
    }

    #[test]
    fn test_capability_estimation() {
        let model = SelfModel::new();
        let (cap, domain) = model.estimate_capability("Kolik je 2+2?");

        assert_eq!(domain, CapabilityDomain::Mathematics);
        assert!(cap >= 0.0 && cap <= 1.0);
    }

    #[test]
    fn test_introspection() {
        let model = SelfModel::new();
        let explanation = model.introspect("Odpověď je 4!", "Kolik je 2+2?");

        assert!(!explanation.why.is_empty());
        assert_eq!(explanation.primary_domain, CapabilityDomain::Mathematics);
    }

    #[test]
    fn test_response_recording() {
        let mut model = SelfModel::new();

        model.record_response("Ahoj", "Čau!", Some(true));
        model.record_response("Ahoj", "Nazdar!", Some(true));

        // Should now have pattern
        let typical = model.profile.typical_response("Ahoj");
        assert!(typical.is_some());
    }

    #[test]
    fn test_self_description() {
        let model = SelfModel::new();
        let desc = model.describe_self();

        assert!(desc.contains("neuromorphic"));
        assert!(desc.contains("%"));
    }
}
