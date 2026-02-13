//! Knowledge Graph - Long-term Semantic Memory Structure
//!
//! Implements a graph-based knowledge representation for persistent
//! storage of concepts, relationships, and facts.
//!
//! # Features
//! - Entity nodes with embeddings
//! - Typed relationships between entities
//! - Subgraph retrieval for context
//! - Graph reasoning and path finding
//!
//! # References
//! - Neural Knowledge Graphs (2024)
//! - Graph Neural Networks for reasoning

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Entity ID
pub type EntityId = u64;

/// Relationship type between entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    IsA,          // Inheritance
    HasProperty,  // Attribute
    PartOf,       // Composition
    RelatedTo,    // Generic relation
    Causes,       // Causal
    LocatedIn,    // Spatial
    OccursBefore, // Temporal
    OccursAfter,  // Temporal
    OppositeTo,   // Antonym
    SynonymOf,    // Synonym
    UsedFor,      // Functional
    CreatedBy,    // Authorship
    InstanceOf,   // Type instance
    DerivedFrom,  // Origin
    Contains,     // Containment
}

impl RelationType {
    /// Get inverse relation
    pub fn inverse(&self) -> RelationType {
        match self {
            RelationType::IsA => RelationType::Contains,
            RelationType::HasProperty => RelationType::HasProperty,
            RelationType::PartOf => RelationType::Contains,
            RelationType::RelatedTo => RelationType::RelatedTo,
            RelationType::Causes => RelationType::Causes,
            RelationType::LocatedIn => RelationType::Contains,
            RelationType::OccursBefore => RelationType::OccursAfter,
            RelationType::OccursAfter => RelationType::OccursBefore,
            RelationType::OppositeTo => RelationType::OppositeTo,
            RelationType::SynonymOf => RelationType::SynonymOf,
            RelationType::UsedFor => RelationType::UsedFor,
            RelationType::CreatedBy => RelationType::CreatedBy,
            RelationType::InstanceOf => RelationType::Contains,
            RelationType::DerivedFrom => RelationType::DerivedFrom,
            RelationType::Contains => RelationType::PartOf,
        }
    }
}

/// Entity node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: EntityId,
    /// Name/label
    pub name: String,
    /// Semantic embedding
    pub embedding: Vec<f32>,
    /// Entity type
    pub entity_type: String,
    /// Properties (key-value pairs)
    pub properties: HashMap<String, String>,
    /// Salience/importance score
    pub salience: f32,
    /// Access count
    pub access_count: u64,
    /// Last access timestamp
    pub last_access: f32,
}

impl Entity {
    pub fn new(id: EntityId, name: String, embedding: Vec<f32>, entity_type: String) -> Self {
        Self {
            id,
            name,
            embedding,
            entity_type,
            properties: HashMap::new(),
            salience: 0.5,
            access_count: 0,
            last_access: 0.0,
        }
    }

    /// Update access statistics
    pub fn touch(&mut self, timestamp: f32) {
        self.access_count += 1;
        self.last_access = timestamp;
        // Increase salience with access
        self.salience = (self.salience + 0.05).min(1.0);
    }
}

/// Edge (relationship) in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source entity
    pub source: EntityId,
    /// Target entity
    pub target: EntityId,
    /// Relationship type
    pub relation: RelationType,
    /// Confidence/strength
    pub weight: f32,
    /// When this edge was created
    pub timestamp: f32,
}

/// Subgraph query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subgraph {
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub center: EntityId,
}

/// Path in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgePath {
    pub nodes: Vec<EntityId>,
    pub relations: Vec<RelationType>,
    pub total_weight: f32,
}

/// Knowledge Graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All entities
    pub entities: HashMap<EntityId, Entity>,
    /// Outgoing edges (source -> list of edges)
    pub outgoing: HashMap<EntityId, Vec<Edge>>,
    /// Incoming edges (target -> list of edges)
    pub incoming: HashMap<EntityId, Vec<Edge>>,
    /// Name to ID mapping
    pub name_index: HashMap<String, EntityId>,
    /// Next available ID
    next_id: EntityId,
    /// Current timestamp
    current_time: f32,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            name_index: HashMap::new(),
            next_id: 1,
            current_time: 0.0,
        }
    }

    /// Add an entity
    pub fn add_entity(
        &mut self,
        name: String,
        embedding: Vec<f32>,
        entity_type: String,
    ) -> EntityId {
        // Check if already exists
        if let Some(&id) = self.name_index.get(&name) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;

        let entity = Entity::new(id, name.clone(), embedding, entity_type);
        self.entities.insert(id, entity);
        self.name_index.insert(name, id);

        id
    }

    /// Get entity by name
    pub fn get_by_name(&self, name: &str) -> Option<&Entity> {
        self.name_index
            .get(name)
            .and_then(|&id| self.entities.get(&id))
    }

    /// Get entity by ID
    pub fn get(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Get mutable entity by ID
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    /// Add a relationship
    pub fn add_relation(
        &mut self,
        source: EntityId,
        relation: RelationType,
        target: EntityId,
        weight: f32,
    ) {
        if !self.entities.contains_key(&source) || !self.entities.contains_key(&target) {
            return; // Invalid entities
        }

        let edge = Edge {
            source,
            target,
            relation,
            weight,
            timestamp: self.current_time,
        };

        self.outgoing.entry(source).or_default().push(edge.clone());
        self.incoming.entry(target).or_default().push(edge);
    }

    /// Query relationships from an entity
    pub fn get_relations(&self, entity: EntityId) -> Vec<&Edge> {
        self.outgoing
            .get(&entity)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get incoming relationships to an entity
    pub fn get_incoming(&self, entity: EntityId) -> Vec<&Edge> {
        self.incoming
            .get(&entity)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get neighbors of an entity
    pub fn get_neighbors(&self, entity: EntityId) -> Vec<EntityId> {
        let mut neighbors = HashSet::new();

        if let Some(edges) = self.outgoing.get(&entity) {
            for edge in edges {
                neighbors.insert(edge.target);
            }
        }

        if let Some(edges) = self.incoming.get(&entity) {
            for edge in edges {
                neighbors.insert(edge.source);
            }
        }

        neighbors.into_iter().collect()
    }

    /// Get subgraph around an entity (BFS)
    pub fn get_subgraph(&mut self, center: EntityId, depth: usize) -> Subgraph {
        let mut entities = Vec::new();
        let mut edges = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((center, 0));
        visited.insert(center);

        while let Some((node, d)) = queue.pop_front() {
            if let Some(entity) = self.entities.get_mut(&node) {
                entity.touch(self.current_time);
                entities.push(entity.clone());
            }

            if d >= depth {
                continue;
            }

            // Add outgoing edges
            if let Some(out_edges) = self.outgoing.get(&node) {
                for edge in out_edges {
                    edges.push(edge.clone());
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target);
                        queue.push_back((edge.target, d + 1));
                    }
                }
            }

            // Add incoming edges
            if let Some(in_edges) = self.incoming.get(&node) {
                for edge in in_edges {
                    if !edges
                        .iter()
                        .any(|e| e.source == edge.source && e.target == edge.target)
                    {
                        edges.push(edge.clone());
                    }
                    if !visited.contains(&edge.source) {
                        visited.insert(edge.source);
                        queue.push_back((edge.source, d + 1));
                    }
                }
            }
        }

        Subgraph {
            entities,
            edges,
            center,
        }
    }

    /// Find path between two entities (BFS)
    pub fn find_path(
        &self,
        start: EntityId,
        end: EntityId,
        max_depth: usize,
    ) -> Option<KnowledgePath> {
        if start == end {
            return Some(KnowledgePath {
                nodes: vec![start],
                relations: Vec::new(),
                total_weight: 1.0,
            });
        }

        let mut visited = HashMap::new(); // node -> (parent, relation, weight)
        let mut queue = VecDeque::new();

        queue.push_back((start, 0));
        visited.insert(start, (start, RelationType::RelatedTo, 1.0));

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            if let Some(edges) = self.outgoing.get(&node) {
                for edge in edges {
                    if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(edge.target)
                    {
                        e.insert((node, edge.relation, edge.weight));
                        queue.push_back((edge.target, depth + 1));

                        if edge.target == end {
                            // Reconstruct path
                            return Some(self.reconstruct_path(&visited, start, end));
                        }
                    }
                }
            }
        }

        None
    }

    /// Reconstruct path from visited map
    fn reconstruct_path(
        &self,
        visited: &HashMap<EntityId, (EntityId, RelationType, f32)>,
        start: EntityId,
        end: EntityId,
    ) -> KnowledgePath {
        let mut nodes = vec![end];
        let mut relations = Vec::new();
        let mut total_weight = 1.0;

        let mut current = end;
        while current != start {
            if let Some(&(parent, rel, weight)) = visited.get(&current) {
                relations.push(rel);
                total_weight *= weight;
                nodes.push(parent);
                current = parent;
            } else {
                break;
            }
        }

        nodes.reverse();
        relations.reverse();

        KnowledgePath {
            nodes,
            relations,
            total_weight,
        }
    }

    /// Find entities by embedding similarity
    pub fn find_similar(&self, query: &[f32], top_k: usize) -> Vec<(EntityId, f32)> {
        let mut results: Vec<(EntityId, f32)> = self
            .entities
            .iter()
            .map(|(&id, entity)| (id, Self::cosine_similarity(query, &entity.embedding)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
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

    /// Infer new relations using transitivity
    pub fn infer_transitive(&mut self, relation: RelationType) -> Vec<Edge> {
        let mut new_edges = Vec::new();

        // For each edge A -> B
        for (&a, edges_a) in &self.outgoing.clone() {
            for edge_ab in edges_a {
                if edge_ab.relation != relation {
                    continue;
                }
                let b = edge_ab.target;

                // Check for B -> C edges
                if let Some(edges_b) = self.outgoing.get(&b) {
                    for edge_bc in edges_b {
                        if edge_bc.relation != relation {
                            continue;
                        }
                        let c = edge_bc.target;

                        // Infer A -> C if not already present
                        let already_exists = self
                            .outgoing
                            .get(&a)
                            .map(|edges| {
                                edges
                                    .iter()
                                    .any(|e| e.target == c && e.relation == relation)
                            })
                            .unwrap_or(false);

                        if !already_exists && a != c {
                            let weight = edge_ab.weight * edge_bc.weight * 0.9; // Decay
                            new_edges.push(Edge {
                                source: a,
                                target: c,
                                relation,
                                weight,
                                timestamp: self.current_time,
                            });
                        }
                    }
                }
            }
        }

        // Add new edges
        for edge in &new_edges {
            self.outgoing
                .entry(edge.source)
                .or_default()
                .push(edge.clone());
            self.incoming
                .entry(edge.target)
                .or_default()
                .push(edge.clone());
        }

        new_edges
    }

    /// Advance time
    pub fn tick(&mut self, dt: f32) {
        self.current_time += dt;
    }

    /// Decay salience of all entities
    pub fn decay_salience(&mut self, decay_rate: f32) {
        for entity in self.entities.values_mut() {
            entity.salience *= 1.0 - decay_rate;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> KnowledgeGraphStats {
        let total_edges: usize = self.outgoing.values().map(|e| e.len()).sum();

        KnowledgeGraphStats {
            num_entities: self.entities.len(),
            num_edges: total_edges,
            avg_salience: self.entities.values().map(|e| e.salience).sum::<f32>()
                / self.entities.len().max(1) as f32,
        }
    }
}

/// Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphStats {
    pub num_entities: usize,
    pub num_edges: usize,
    pub avg_salience: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entity() {
        let mut kg = KnowledgeGraph::new();
        let id = kg.add_entity("Dog".into(), vec![1.0, 0.0], "Animal".into());

        assert!(kg.entities.contains_key(&id));
        assert_eq!(kg.get_by_name("Dog").unwrap().name, "Dog");
    }

    #[test]
    fn test_add_relation() {
        let mut kg = KnowledgeGraph::new();
        let dog = kg.add_entity("Dog".into(), vec![1.0], "Animal".into());
        let animal = kg.add_entity("Animal".into(), vec![0.8], "Category".into());

        kg.add_relation(dog, RelationType::IsA, animal, 1.0);

        let relations = kg.get_relations(dog);
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].target, animal);
    }

    #[test]
    fn test_find_path() {
        let mut kg = KnowledgeGraph::new();
        let dog = kg.add_entity("Dog".into(), vec![1.0], "Animal".into());
        let mammal = kg.add_entity("Mammal".into(), vec![0.9], "Category".into());
        let animal = kg.add_entity("Animal".into(), vec![0.8], "Category".into());

        kg.add_relation(dog, RelationType::IsA, mammal, 1.0);
        kg.add_relation(mammal, RelationType::IsA, animal, 1.0);

        let path = kg.find_path(dog, animal, 5);
        assert!(path.is_some());

        let path = path.unwrap();
        assert_eq!(path.nodes.len(), 3);
    }

    #[test]
    fn test_subgraph() {
        let mut kg = KnowledgeGraph::new();
        let dog = kg.add_entity("Dog".into(), vec![1.0], "Animal".into());
        let cat = kg.add_entity("Cat".into(), vec![0.9], "Animal".into());
        let mammal = kg.add_entity("Mammal".into(), vec![0.8], "Category".into());

        kg.add_relation(dog, RelationType::IsA, mammal, 1.0);
        kg.add_relation(cat, RelationType::IsA, mammal, 1.0);

        let subgraph = kg.get_subgraph(mammal, 1);
        assert_eq!(subgraph.entities.len(), 3); // dog, cat, mammal
    }

    #[test]
    fn test_transitive_inference() {
        let mut kg = KnowledgeGraph::new();
        let poodle = kg.add_entity("Poodle".into(), vec![1.0], "Breed".into());
        let dog = kg.add_entity("Dog".into(), vec![0.9], "Animal".into());
        let mammal = kg.add_entity("Mammal".into(), vec![0.8], "Category".into());

        kg.add_relation(poodle, RelationType::IsA, dog, 1.0);
        kg.add_relation(dog, RelationType::IsA, mammal, 1.0);

        let new_edges = kg.infer_transitive(RelationType::IsA);

        // Should infer Poodle IsA Mammal
        assert!(!new_edges.is_empty());
        assert!(new_edges
            .iter()
            .any(|e| e.source == poodle && e.target == mammal));
    }
}
