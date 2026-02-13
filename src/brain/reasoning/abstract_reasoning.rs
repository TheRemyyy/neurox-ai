//! Abstract Reasoning - Neuro-Symbolic Integration
//!
//! Implements abstract reasoning capabilities combining neural and symbolic approaches:
//! - Analogical reasoning (A:B :: C:?)
//! - Rule-based inference
//! - Compositional reasoning
//!
//! # References
//! - Neuro-Symbolic AI (2024)
//! - ARC Challenge research
//! - Analogical reasoning in neural networks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Relation type for compositional reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Relation {
    IsA,      // X is a Y
    HasA,     // X has a Y
    PartOf,   // X is part of Y
    Causes,   // X causes Y
    Before,   // X happens before Y
    After,    // X happens after Y
    Opposite, // X is opposite of Y
    Similar,  // X is similar to Y
    Contains, // X contains Y
    Uses,     // X uses Y
}

impl Relation {
    /// Get inverse relation
    pub fn inverse(&self) -> Relation {
        match self {
            Relation::IsA => Relation::Contains,
            Relation::HasA => Relation::PartOf,
            Relation::PartOf => Relation::HasA,
            Relation::Causes => Relation::Causes, // Symmetric for simplicity
            Relation::Before => Relation::After,
            Relation::After => Relation::Before,
            Relation::Opposite => Relation::Opposite,
            Relation::Similar => Relation::Similar,
            Relation::Contains => Relation::IsA,
            Relation::Uses => Relation::Uses,
        }
    }
}

/// Logical rule for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalRule {
    /// Rule name
    pub name: String,
    /// Antecedent patterns (premises)
    pub antecedents: Vec<RulePattern>,
    /// Consequent (conclusion)
    pub consequent: RulePattern,
    /// Confidence/strength
    pub confidence: f32,
}

/// Pattern in a rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RulePattern {
    /// Subject (can be variable like "?X")
    pub subject: String,
    /// Relation
    pub relation: Relation,
    /// Object (can be variable)
    pub object: String,
}

impl RulePattern {
    pub fn new(subject: &str, relation: Relation, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            relation,
            object: object.to_string(),
        }
    }

    /// Check if this pattern is a variable
    pub fn is_variable(s: &str) -> bool {
        s.starts_with('?')
    }
}

/// Fact in the knowledge base
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub subject: String,
    pub relation: Relation,
    pub object: String,
    pub confidence: f32,
}

impl Fact {
    pub fn new(subject: &str, relation: Relation, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            relation,
            object: object.to_string(),
            confidence: 1.0,
        }
    }
}

/// Reasoning step in a chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// What was inferred
    pub conclusion: Fact,
    /// Which rule was applied
    pub rule_used: String,
    /// What facts supported this step
    pub supporting_facts: Vec<Fact>,
}

/// Reasoning chain (proof)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningChain {
    pub steps: Vec<ReasoningStep>,
    pub final_conclusion: Option<Fact>,
    pub overall_confidence: f32,
}

impl Default for ReasoningChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningChain {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            final_conclusion: None,
            overall_confidence: 1.0,
        }
    }

    pub fn add_step(&mut self, step: ReasoningStep) {
        self.overall_confidence *= step.conclusion.confidence;
        self.final_conclusion = Some(step.conclusion.clone());
        self.steps.push(step);
    }
}

/// Analogy result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyResult {
    /// The inferred D in A:B :: C:D
    pub result: Vec<f32>,
    /// Confidence
    pub confidence: f32,
    /// Explanation
    pub explanation: String,
}

/// Analogy Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyEngine {
    /// Stored analogy pairs for reference
    pub known_pairs: Vec<(Vec<f32>, Vec<f32>)>,
    /// Maximum stored pairs
    pub max_pairs: usize,
}

impl Default for AnalogyEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalogyEngine {
    pub fn new() -> Self {
        Self {
            known_pairs: Vec::new(),
            max_pairs: 100,
        }
    }

    /// Solve analogy: A is to B as C is to ?
    ///
    /// Uses vector arithmetic: D = C + (B - A)
    pub fn solve(&self, a: &[f32], b: &[f32], c: &[f32]) -> AnalogyResult {
        if a.len() != b.len() || b.len() != c.len() || a.is_empty() {
            return AnalogyResult {
                result: Vec::new(),
                confidence: 0.0,
                explanation: "Dimension mismatch".into(),
            };
        }

        // Compute transformation: B - A
        let transformation: Vec<f32> = b.iter().zip(a.iter()).map(|(bi, ai)| bi - ai).collect();

        // Apply to C: D = C + transformation
        let d: Vec<f32> = c
            .iter()
            .zip(transformation.iter())
            .map(|(ci, ti)| ci + ti)
            .collect();

        // Estimate confidence based on transformation magnitude
        let transform_mag: f32 = transformation.iter().map(|x| x.abs()).sum();
        let confidence = if transform_mag > 0.0 {
            (1.0 / (1.0 + transform_mag * 0.1)).min(0.9)
        } else {
            0.5 // No transformation = uncertain
        };

        AnalogyResult {
            result: d,
            confidence,
            explanation: format!("Applied transformation with magnitude {:.2}", transform_mag),
        }
    }

    /// Learn from analogy pair
    pub fn learn_pair(&mut self, a: Vec<f32>, b: Vec<f32>) {
        if self.known_pairs.len() >= self.max_pairs {
            self.known_pairs.remove(0);
        }
        self.known_pairs.push((a, b));
    }

    /// Find similar pair
    pub fn find_similar(&self, query: &[f32]) -> Option<&(Vec<f32>, Vec<f32>)> {
        self.known_pairs.iter().max_by(|a, b| {
            let sim_a = Self::cosine_similarity(&a.0, query);
            let sim_b = Self::cosine_similarity(&b.0, query);
            sim_a
                .partial_cmp(&sim_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a * norm_b > 1e-10 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Abstract Reasoning System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractReasoning {
    /// Logical rules
    pub rules: Vec<LogicalRule>,
    /// Known facts
    pub facts: Vec<Fact>,
    /// Analogy engine
    pub analogy: AnalogyEngine,
    /// Reasoning buffer
    pub reasoning_buffer: Vec<ReasoningChain>,
    /// Maximum chains to store
    pub max_chains: usize,
}

impl Default for AbstractReasoning {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractReasoning {
    pub fn new() -> Self {
        Self {
            rules: Self::default_rules(),
            facts: Vec::new(),
            analogy: AnalogyEngine::new(),
            reasoning_buffer: Vec::new(),
            max_chains: 10,
        }
    }

    /// Default logical rules
    fn default_rules() -> Vec<LogicalRule> {
        vec![
            // Transitivity of IsA
            LogicalRule {
                name: "IsA-Transitivity".into(),
                antecedents: vec![
                    RulePattern::new("?X", Relation::IsA, "?Y"),
                    RulePattern::new("?Y", Relation::IsA, "?Z"),
                ],
                consequent: RulePattern::new("?X", Relation::IsA, "?Z"),
                confidence: 0.9,
            },
            // HasA inheritance
            LogicalRule {
                name: "HasA-Inheritance".into(),
                antecedents: vec![
                    RulePattern::new("?X", Relation::IsA, "?Y"),
                    RulePattern::new("?Y", Relation::HasA, "?Z"),
                ],
                consequent: RulePattern::new("?X", Relation::HasA, "?Z"),
                confidence: 0.8,
            },
            // Causation chain
            LogicalRule {
                name: "Cause-Chain".into(),
                antecedents: vec![
                    RulePattern::new("?X", Relation::Causes, "?Y"),
                    RulePattern::new("?Y", Relation::Causes, "?Z"),
                ],
                consequent: RulePattern::new("?X", Relation::Causes, "?Z"),
                confidence: 0.7,
            },
            // Opposite symmetry
            LogicalRule {
                name: "Opposite-Symmetry".into(),
                antecedents: vec![RulePattern::new("?X", Relation::Opposite, "?Y")],
                consequent: RulePattern::new("?Y", Relation::Opposite, "?X"),
                confidence: 1.0,
            },
        ]
    }

    /// Add a fact
    pub fn add_fact(&mut self, fact: Fact) {
        // Check for duplicates
        let exists = self.facts.iter().any(|f| {
            f.subject == fact.subject && f.relation == fact.relation && f.object == fact.object
        });

        if !exists {
            self.facts.push(fact);
        }
    }

    /// Query for facts
    pub fn query(
        &self,
        subject: Option<&str>,
        relation: Option<Relation>,
        object: Option<&str>,
    ) -> Vec<&Fact> {
        self.facts
            .iter()
            .filter(|f| {
                let s_match = subject.is_none_or(|s| f.subject == s);
                let r_match = relation.is_none_or(|r| f.relation == r);
                let o_match = object.is_none_or(|o| f.object == o);
                s_match && r_match && o_match
            })
            .collect()
    }

    /// Infer new facts using rules
    pub fn infer(&mut self) -> Vec<Fact> {
        let mut new_facts = Vec::new();

        for rule in &self.rules {
            // Try to match rule antecedents
            let bindings = self.match_rule(&rule.antecedents);

            for binding in bindings {
                // Apply binding to consequent
                let subject = self.apply_binding(&rule.consequent.subject, &binding);
                let object = self.apply_binding(&rule.consequent.object, &binding);

                let new_fact = Fact {
                    subject,
                    relation: rule.consequent.relation,
                    object,
                    confidence: rule.confidence,
                };

                // Check if already known
                let exists = self.facts.iter().any(|f| {
                    f.subject == new_fact.subject
                        && f.relation == new_fact.relation
                        && f.object == new_fact.object
                });

                if !exists {
                    new_facts.push(new_fact);
                }
            }
        }

        // Add new facts to knowledge base
        for fact in &new_facts {
            self.facts.push(fact.clone());
        }

        new_facts
    }

    /// Match rule antecedents against facts
    fn match_rule(&self, antecedents: &[RulePattern]) -> Vec<HashMap<String, String>> {
        if antecedents.is_empty() {
            return vec![HashMap::new()];
        }

        let mut bindings = vec![HashMap::new()];

        for pattern in antecedents {
            let mut new_bindings = Vec::new();

            for binding in &bindings {
                // Find facts that match this pattern
                for fact in &self.facts {
                    if fact.relation != pattern.relation {
                        continue;
                    }

                    let mut new_binding = binding.clone();

                    // Try to unify subject
                    if RulePattern::is_variable(&pattern.subject) {
                        if let Some(existing) = binding.get(&pattern.subject) {
                            if *existing != fact.subject {
                                continue; // Binding conflict
                            }
                        } else {
                            new_binding.insert(pattern.subject.clone(), fact.subject.clone());
                        }
                    } else if pattern.subject != fact.subject {
                        continue; // Literal mismatch
                    }

                    // Try to unify object
                    if RulePattern::is_variable(&pattern.object) {
                        if let Some(existing) = binding.get(&pattern.object) {
                            if *existing != fact.object {
                                continue; // Binding conflict
                            }
                        } else {
                            new_binding.insert(pattern.object.clone(), fact.object.clone());
                        }
                    } else if pattern.object != fact.object {
                        continue; // Literal mismatch
                    }

                    new_bindings.push(new_binding);
                }
            }

            bindings = new_bindings;
        }

        bindings
    }

    /// Apply variable binding to a term
    fn apply_binding(&self, term: &str, binding: &HashMap<String, String>) -> String {
        if RulePattern::is_variable(term) {
            binding
                .get(term)
                .cloned()
                .unwrap_or_else(|| term.to_string())
        } else {
            term.to_string()
        }
    }

    /// Solve analogy using analogy engine
    pub fn solve_analogy(&self, a: &[f32], b: &[f32], c: &[f32]) -> AnalogyResult {
        self.analogy.solve(a, b, c)
    }

    /// Compose two concepts using a relation
    pub fn compose(&self, concept_a: &[f32], relation: Relation, concept_b: &[f32]) -> Vec<f32> {
        // Simple composition: weighted combination based on relation
        let weight_a = match relation {
            Relation::IsA | Relation::Similar => 0.3,
            Relation::HasA | Relation::Contains => 0.5,
            Relation::Causes | Relation::Before | Relation::After => 0.4,
            _ => 0.5,
        };
        let weight_b = 1.0 - weight_a;

        let len = concept_a.len().max(concept_b.len());
        let mut result = Vec::with_capacity(len);

        for i in 0..len {
            let a = concept_a.get(i).copied().unwrap_or(0.0);
            let b = concept_b.get(i).copied().unwrap_or(0.0);
            result.push(a * weight_a + b * weight_b);
        }

        result
    }

    /// Get statistics
    pub fn stats(&self) -> AbstractReasoningStats {
        AbstractReasoningStats {
            num_rules: self.rules.len(),
            num_facts: self.facts.len(),
            num_analogy_pairs: self.analogy.known_pairs.len(),
            num_reasoning_chains: self.reasoning_buffer.len(),
        }
    }
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractReasoningStats {
    pub num_rules: usize,
    pub num_facts: usize,
    pub num_analogy_pairs: usize,
    pub num_reasoning_chains: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analogy() {
        let engine = AnalogyEngine::new();

        // king - man + woman = queen (simplified)
        let king = vec![1.0, 0.0, 0.5];
        let man = vec![0.0, 0.0, 0.5];
        let woman = vec![0.0, 1.0, 0.5];

        let result = engine.solve(&king, &man, &woman);

        // Should transform woman similarly to how man transforms to king
        assert!(!result.result.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_fact_query() {
        let mut reasoning = AbstractReasoning::new();

        reasoning.add_fact(Fact::new("dog", Relation::IsA, "animal"));
        reasoning.add_fact(Fact::new("cat", Relation::IsA, "animal"));
        reasoning.add_fact(Fact::new("animal", Relation::IsA, "living_thing"));

        let animals = reasoning.query(None, Some(Relation::IsA), Some("animal"));
        assert_eq!(animals.len(), 2);
    }

    #[test]
    fn test_inference() {
        let mut reasoning = AbstractReasoning::new();

        // Dog is Animal, Animal is Living Thing
        reasoning.add_fact(Fact::new("dog", Relation::IsA, "animal"));
        reasoning.add_fact(Fact::new("animal", Relation::IsA, "living_thing"));

        // Infer: Dog is Living Thing
        let new_facts = reasoning.infer();

        // Should infer transitivity
        let dog_living = new_facts
            .iter()
            .any(|f| f.subject == "dog" && f.object == "living_thing");
        assert!(dog_living, "Should infer dog is living_thing");
    }

    #[test]
    fn test_composition() {
        let reasoning = AbstractReasoning::new();

        let dog = vec![1.0, 0.0, 0.0];
        let big = vec![0.0, 1.0, 0.0];

        let big_dog = reasoning.compose(&dog, Relation::HasA, &big);

        // Should be combination of both
        assert_eq!(big_dog.len(), 3);
        assert!(big_dog[0] > 0.0); // Some dog
        assert!(big_dog[1] > 0.0); // Some big
    }
}
