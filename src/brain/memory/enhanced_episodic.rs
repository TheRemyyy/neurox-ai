//! Enhanced Episodic Memory - Experience Replay and Consolidation
//!
//! Advanced episodic memory system with:
//! - Multi-index retrieval (temporal, semantic, emotional)
//! - Sleep-based consolidation
//! - Experience replay for learning
//! - Schema extraction
//!
//! # References
//! - Hippocampal replay mechanisms (2024-2025)
//! - Memory consolidation during sleep

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

/// Episode ID
pub type EpisodeId = u64;

/// Emotional valence tags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmotionalTag {
    Positive,
    Negative,
    Neutral,
    Surprising,
    Important,
}

/// Episode - A single experience/event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier
    pub id: EpisodeId,
    /// When this occurred
    pub timestamp: f32,
    /// Context embedding
    pub context: Vec<f32>,
    /// Content/event embedding
    pub content: Vec<f32>,
    /// Outcome/result embedding
    pub outcome: Vec<f32>,
    /// Emotional tags
    pub emotional_tags: Vec<EmotionalTag>,
    /// Importance score (for prioritized replay)
    pub importance: f32,
    /// Access count
    pub access_count: u32,
    /// Last access time
    pub last_access: f32,
    /// Consolidation level (0 = fresh, 1 = fully consolidated)
    pub consolidation: f32,
    /// Associated schema IDs
    pub schema_ids: Vec<u64>,
}

impl Episode {
    pub fn new(
        id: EpisodeId,
        timestamp: f32,
        context: Vec<f32>,
        content: Vec<f32>,
        outcome: Vec<f32>,
    ) -> Self {
        Self {
            id,
            timestamp,
            context,
            content,
            outcome,
            emotional_tags: vec![EmotionalTag::Neutral],
            importance: 0.5,
            access_count: 0,
            last_access: timestamp,
            consolidation: 0.0,
            schema_ids: Vec::new(),
        }
    }

    /// Touch episode (for retrieval statistics)
    pub fn touch(&mut self, time: f32) {
        self.access_count += 1;
        self.last_access = time;
    }

    /// Combined embedding for similarity search
    pub fn combined_embedding(&self) -> Vec<f32> {
        let len = self.context.len().max(self.content.len());
        let mut combined = Vec::with_capacity(len);

        for i in 0..len {
            let ctx = self.context.get(i).copied().unwrap_or(0.0);
            let cnt = self.content.get(i).copied().unwrap_or(0.0);
            combined.push((ctx + cnt) / 2.0);
        }

        combined
    }
}

/// Entry for priority queue replay
#[derive(Debug, Clone)]
struct ReplayEntry {
    episode_id: EpisodeId,
    priority: f32,
}

impl PartialEq for ReplayEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ReplayEntry {}

impl PartialOrd for ReplayEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReplayEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
    }
}

/// Schema - Abstracted pattern from multiple episodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub id: u64,
    /// Name/label
    pub name: String,
    /// Prototype embedding
    pub prototype: Vec<f32>,
    /// Contributing episodes
    pub episode_count: u32,
    /// Confidence in schema
    pub confidence: f32,
}

/// Retrieval query
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    /// Semantic query
    pub semantic: Option<Vec<f32>>,
    /// Time range
    pub time_range: Option<(f32, f32)>,
    /// Emotional filter
    pub emotion: Option<EmotionalTag>,
    /// Minimum importance
    pub min_importance: Option<f32>,
    /// Maximum results
    pub limit: usize,
}

impl Default for RetrievalQuery {
    fn default() -> Self {
        Self {
            semantic: None,
            time_range: None,
            emotion: None,
            min_importance: None,
            limit: 10,
        }
    }
}

/// Retrieval result
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub episode: Episode,
    pub similarity: f32,
    pub recency: f32,
}

/// Enhanced Episodic Memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedEpisodicMemory {
    /// All episodes
    episodes: HashMap<EpisodeId, Episode>,
    /// Temporal index (ordered by timestamp)
    temporal_index: VecDeque<EpisodeId>,
    /// Emotional indices
    emotional_index: HashMap<EmotionalTag, Vec<EpisodeId>>,
    /// Learned schemas
    schemas: HashMap<u64, Schema>,
    /// Next episode ID
    next_episode_id: EpisodeId,
    /// Next schema ID
    next_schema_id: u64,
    /// Current time
    current_time: f32,
    /// Maximum episodes to store
    pub max_episodes: usize,
    /// Consolidation rate during replay
    pub consolidation_rate: f32,
}

impl Default for EnhancedEpisodicMemory {
    fn default() -> Self {
        Self::new(10000)
    }
}

impl EnhancedEpisodicMemory {
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: HashMap::new(),
            temporal_index: VecDeque::new(),
            emotional_index: HashMap::new(),
            schemas: HashMap::new(),
            next_episode_id: 1,
            next_schema_id: 1,
            current_time: 0.0,
            max_episodes,
            consolidation_rate: 0.1,
        }
    }

    /// Store a new episode
    pub fn store(
        &mut self,
        context: Vec<f32>,
        content: Vec<f32>,
        outcome: Vec<f32>,
        importance: f32,
    ) -> EpisodeId {
        let id = self.next_episode_id;
        self.next_episode_id += 1;

        let mut episode = Episode::new(id, self.current_time, context, content, outcome);
        episode.importance = importance;

        // Determine emotional tag based on outcome
        let outcome_valence = episode.outcome.iter().sum::<f32>();
        if outcome_valence > 0.5 {
            episode.emotional_tags = vec![EmotionalTag::Positive];
        } else if outcome_valence < -0.5 {
            episode.emotional_tags = vec![EmotionalTag::Negative];
        }

        if importance > 0.8 {
            episode.emotional_tags.push(EmotionalTag::Important);
        }

        // Index by emotion
        for tag in &episode.emotional_tags {
            self.emotional_index.entry(*tag).or_default().push(id);
        }

        // Add to temporal index
        self.temporal_index.push_back(id);

        // Store episode
        self.episodes.insert(id, episode);

        // Prune if necessary
        self.prune_if_needed();

        id
    }

    /// Prune old, low-importance episodes
    fn prune_if_needed(&mut self) {
        while self.episodes.len() > self.max_episodes {
            // Remove oldest, least important episode
            if let Some(&oldest_id) = self.temporal_index.front() {
                if let Some(oldest) = self.episodes.get(&oldest_id) {
                    if oldest.importance < 0.3 && oldest.consolidation < 0.5 {
                        self.remove_episode(oldest_id);
                        continue;
                    }
                }
            }

            // Find least important overall
            let least_important = self
                .episodes
                .iter()
                .min_by(|a, b| {
                    a.1.importance
                        .partial_cmp(&b.1.importance)
                        .unwrap_or(Ordering::Equal)
                })
                .map(|(&id, _)| id);

            if let Some(id) = least_important {
                self.remove_episode(id);
            } else {
                break;
            }
        }
    }

    /// Remove an episode
    fn remove_episode(&mut self, id: EpisodeId) {
        self.episodes.remove(&id);
        self.temporal_index.retain(|&x| x != id);
        for index in self.emotional_index.values_mut() {
            index.retain(|&x| x != id);
        }
    }

    /// Retrieve episodes based on query
    pub fn retrieve(&mut self, query: RetrievalQuery) -> Vec<RetrievalResult> {
        let mut results = Vec::new();

        for episode in self.episodes.values_mut() {
            // Apply filters
            if let Some((start, end)) = query.time_range {
                if episode.timestamp < start || episode.timestamp > end {
                    continue;
                }
            }

            if let Some(emotion) = query.emotion {
                if !episode.emotional_tags.contains(&emotion) {
                    continue;
                }
            }

            if let Some(min_imp) = query.min_importance {
                if episode.importance < min_imp {
                    continue;
                }
            }

            // Compute similarity
            let similarity = if let Some(ref semantic) = query.semantic {
                Self::cosine_similarity(semantic, &episode.combined_embedding())
            } else {
                1.0 // No semantic filter, all pass
            };

            // Compute recency
            let recency = 1.0 / (1.0 + (self.current_time - episode.timestamp).max(0.0) * 0.01);

            // Update access stats
            episode.touch(self.current_time);

            results.push(RetrievalResult {
                episode: episode.clone(),
                similarity,
                recency,
            });
        }

        // Sort by combined score
        results.sort_by(|a, b| {
            let score_a = a.similarity * 0.7 + a.recency * 0.3;
            let score_b = b.similarity * 0.7 + b.recency * 0.3;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        results.truncate(query.limit);
        results
    }

    /// Get recent episodes
    pub fn get_recent(&self, count: usize) -> Vec<&Episode> {
        self.temporal_index
            .iter()
            .rev()
            .take(count)
            .filter_map(|id| self.episodes.get(id))
            .collect()
    }

    /// Priority-based experience replay
    pub fn replay(&mut self, count: usize) -> Vec<Episode> {
        let mut heap = BinaryHeap::new();

        // Build priority queue
        for (&id, episode) in &self.episodes {
            // Priority = importance * (1 - consolidation) + recency bonus
            let recency = 1.0 / (1.0 + (self.current_time - episode.timestamp).max(0.0) * 0.01);
            let priority = episode.importance * (1.0 - episode.consolidation) + recency * 0.2;

            heap.push(ReplayEntry {
                episode_id: id,
                priority,
            });
        }

        // Get top entries and consolidate
        let mut replayed = Vec::new();
        for _ in 0..count {
            if let Some(entry) = heap.pop() {
                if let Some(episode) = self.episodes.get_mut(&entry.episode_id) {
                    // Consolidate during replay
                    episode.consolidation =
                        (episode.consolidation + self.consolidation_rate).min(1.0);
                    episode.touch(self.current_time);
                    replayed.push(episode.clone());
                }
            }
        }

        replayed
    }

    /// Sleep-based consolidation (offline processing)
    pub fn sleep_consolidation(&mut self, duration: f32) {
        let replay_count = (duration * 10.0) as usize; // 10 replays per unit time

        for _ in 0..replay_count {
            // Replay with priority
            let replayed = self.replay(5);

            // Try to extract schemas from replayed episodes
            if replayed.len() >= 3 {
                self.try_extract_schema(&replayed);
            }
        }

        // Decay unconsolidated memories
        for episode in self.episodes.values_mut() {
            if episode.consolidation < 0.3 {
                episode.importance *= 0.95; // Decay importance
            }
        }
    }

    /// Try to extract schema from similar episodes
    fn try_extract_schema(&mut self, episodes: &[Episode]) {
        if episodes.is_empty() {
            return;
        }

        // Compute prototype (average embedding)
        let dim = episodes[0].content.len();
        let mut prototype = vec![0.0; dim];

        for episode in episodes {
            for (i, &val) in episode.content.iter().enumerate() {
                if i < dim {
                    prototype[i] += val;
                }
            }
        }

        let n = episodes.len() as f32;
        for val in &mut prototype {
            *val /= n;
        }

        // Check if this schema already exists
        for schema in self.schemas.values_mut() {
            let similarity = Self::cosine_similarity(&prototype, &schema.prototype);
            if similarity > 0.8 {
                // Merge into existing schema
                schema.episode_count += episodes.len() as u32;
                schema.confidence = (schema.confidence + 0.1).min(1.0);
                return;
            }
        }

        // Create new schema
        let schema_id = self.next_schema_id;
        self.next_schema_id += 1;

        let schema = Schema {
            id: schema_id,
            name: format!("Schema_{}", schema_id),
            prototype,
            episode_count: episodes.len() as u32,
            confidence: 0.3,
        };

        self.schemas.insert(schema_id, schema);

        // Link episodes to schema
        for episode in episodes {
            if let Some(ep) = self.episodes.get_mut(&episode.id) {
                ep.schema_ids.push(schema_id);
            }
        }
    }

    /// Find similar episodes
    pub fn find_similar(&mut self, query: &[f32], top_k: usize) -> Vec<RetrievalResult> {
        let rq = RetrievalQuery {
            semantic: Some(query.to_vec()),
            limit: top_k,
            ..Default::default()
        };
        self.retrieve(rq)
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

    /// Advance time
    pub fn tick(&mut self, dt: f32) {
        self.current_time += dt;
    }

    /// Get statistics
    pub fn stats(&self) -> EnhancedEpisodicStats {
        let avg_consolidation = if self.episodes.is_empty() {
            0.0
        } else {
            self.episodes.values().map(|e| e.consolidation).sum::<f32>()
                / self.episodes.len() as f32
        };

        EnhancedEpisodicStats {
            num_episodes: self.episodes.len(),
            num_schemas: self.schemas.len(),
            avg_consolidation,
            avg_importance: self.episodes.values().map(|e| e.importance).sum::<f32>()
                / self.episodes.len().max(1) as f32,
        }
    }
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedEpisodicStats {
    pub num_episodes: usize,
    pub num_schemas: usize,
    pub avg_consolidation: f32,
    pub avg_importance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_episode() {
        let mut mem = EnhancedEpisodicMemory::new(100);
        let id = mem.store(vec![1.0, 0.0], vec![0.5, 0.5], vec![1.0, 0.0], 0.7);

        assert!(mem.episodes.contains_key(&id));
    }

    #[test]
    fn test_retrieve() {
        let mut mem = EnhancedEpisodicMemory::new(100);

        mem.store(vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], 0.8);
        mem.store(vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0], 0.5);

        let query = RetrievalQuery {
            semantic: Some(vec![1.0, 0.0]),
            limit: 1,
            ..Default::default()
        };

        let results = mem.retrieve(query);
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.5);
    }

    #[test]
    fn test_replay() {
        let mut mem = EnhancedEpisodicMemory::new(100);

        for i in 0..10 {
            mem.store(
                vec![i as f32 * 0.1],
                vec![i as f32 * 0.1],
                vec![1.0],
                (i as f32) * 0.1,
            );
        }

        let replayed = mem.replay(3);
        assert_eq!(replayed.len(), 3);

        // Higher importance episodes should be replayed first
        assert!(replayed[0].importance >= replayed[1].importance);
    }

    #[test]
    fn test_consolidation() {
        let mut mem = EnhancedEpisodicMemory::new(100);
        let id = mem.store(vec![1.0], vec![1.0], vec![1.0], 0.9);

        // Initial consolidation should be 0
        assert_eq!(mem.episodes.get(&id).unwrap().consolidation, 0.0);

        // Replay should increase consolidation
        mem.replay(5);

        assert!(mem.episodes.get(&id).unwrap().consolidation > 0.0);
    }

    #[test]
    fn test_emotional_index() {
        let mut mem = EnhancedEpisodicMemory::new(100);

        // Store positive episode
        mem.store(vec![1.0], vec![1.0], vec![1.0], 0.9); // Positive outcome

        let positives = mem.emotional_index.get(&EmotionalTag::Positive).unwrap();
        assert!(!positives.is_empty());
    }
}
