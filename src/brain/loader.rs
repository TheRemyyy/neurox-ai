use crate::brain::NeuromorphicBrain;
use crate::datasets::{JsonDataset, TextDataset};
use crate::language::{IntentType, PartOfSpeech, PragmaticRule, SentenceTemplate};
use std::error::Error;
use std::io::{stdout, Write};

/// Helper to parse POS string to enum
pub fn pos_from_string(s: &str) -> PartOfSpeech {
    match s.to_lowercase().as_str() {
        "noun" => PartOfSpeech::Noun,
        "verb" => PartOfSpeech::Verb,
        "adjective" => PartOfSpeech::Adjective,
        "adverb" => PartOfSpeech::Adverb,
        "pronoun" => PartOfSpeech::Pronoun,
        "preposition" => PartOfSpeech::Preposition,
        "conjunction" => PartOfSpeech::Conjunction,
        "interjection" => PartOfSpeech::Interjection,
        "determiner" => PartOfSpeech::Determiner,
        "particle" => PartOfSpeech::Particle,
        _ => PartOfSpeech::Unknown,
    }
}

pub struct BrainLoader;

impl BrainLoader {
    pub fn load_json_training(
        brain: &mut NeuromorphicBrain,
        file_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        let dataset = JsonDataset::from_file(file_path)?;
        let stats = dataset.stats();
        println!(
            "  Loaded: {} pairs ({} positive, {} negative)",
            stats.total_pairs, stats.positive_pairs, stats.negative_pairs
        );

        // === STEP 0: Load Metacognition Markers ===
        if let Some(markers) = &dataset.metacognition_markers {
            println!(
                "  Loading Metacognition: {} reasoning, {} uncertainty markers",
                markers.reasoning_markers.len(),
                markers.uncertainty_markers.len()
            );
            brain.metacognition.load_markers(
                markers.reasoning_markers.clone(),
                markers.uncertainty_markers.clone(),
            );
        }

        // === STEP 1: Load vocabulary from JSON into Lexicon (with FULL semantics) ===
        println!(
            "  Loading vocabulary: {} words with semantics",
            dataset.vocabulary.len()
        );
        for vocab_word in &dataset.vocabulary {
            let pos = pos_from_string(&vocab_word.pos);
            let mut annotated = crate::language::AnnotatedWord::new(&vocab_word.word, pos)
                .with_valence(vocab_word.valence);

            // Load semantic knowledge
            annotated.definition = vocab_word.definition.clone();
            annotated.synonyms = vocab_word.synonyms.clone();
            annotated.antonyms = vocab_word.antonyms.clone();
            annotated.context_tags = vocab_word.context.clone();
            annotated.responds_to = vocab_word.responds_to.clone();
            annotated.triggers_mood = vocab_word.triggers_mood.clone();
            annotated.requires_bond = vocab_word.requires_bond;
            annotated.neuro_impact = vocab_word.neuro_impact.clone();

            brain.lexicon.add_word(annotated);
            brain
                .language
                .ventral
                .embeddings
                .add_word(vocab_word.word.clone());
        }

        // === STEP 2: Load sentence templates into IFG ===
        println!(
            "  Loading templates: {} sentence structures",
            dataset.sentence_templates.len()
        );
        for template in &dataset.sentence_templates {
            let structure: Vec<PartOfSpeech> = template
                .structure
                .iter()
                .map(|s| pos_from_string(s))
                .collect();
            let intent = match template.intent.as_str() {
                "greeting" => IntentType::Greeting,
                "statement" => IntentType::Statement,
                "question" => IntentType::Question,
                "response" => IntentType::Response,
                "emotional" => IntentType::Emotional,
                "insult" => IntentType::Insult,
                "thanks" => IntentType::Thanks,
                "farewell" => IntentType::Farewell,
                "explanation" => IntentType::Explanation,
                "humor" => IntentType::Humor,
                "philosophy" => IntentType::Philosophy,
                "clarification" => IntentType::Clarification,
                _ => IntentType::Statement,
            };
            brain.ifg_planner.templates.push(SentenceTemplate {
                structure,
                intent_type: intent,
            });
        }

        // === STEP 3: Load pragmatic rules ===
        println!(
            "  Loading pragmatic rules: {} how-to-respond rules",
            dataset.pragmatic_rules.len()
        );
        for rule in &dataset.pragmatic_rules {
            brain.ifg_planner.pragmatic_rules.push(PragmaticRule {
                input_intent: rule.input_intent.clone(),
                response_intent: rule.response_intent.clone(),
                mood_effect: rule.mood_effect,
            });
        }

        // === STEP 4: Load Dynamic Intent Detection Rules ===
        println!(
            "  Loading intent rules: {} classification patterns",
            dataset.intent_rules.len()
        );
        for rule in &dataset.intent_rules {
            let intent = match rule.intent.as_str() {
                "greeting" => IntentType::Greeting,
                "question" => IntentType::Question,
                "emotional" => IntentType::Emotional,
                "statement" => IntentType::Statement,
                "agreement" => IntentType::Statement,
                "insult" => IntentType::Insult,
                "thanks" => IntentType::Thanks,
                "farewell" => IntentType::Farewell,
                "explanation" => IntentType::Explanation,
                "humor" => IntentType::Humor,
                "philosophy" => IntentType::Philosophy,
                "clarification" => IntentType::Clarification,
                _ => IntentType::Statement,
            };
            brain.intent_rules.push((intent, rule.keywords.clone()));
        }

        // === STEP 5: Train supervised pairs for semantic learning ===
        let total = dataset.pairs.len();
        let bar_width = 30;

        if total > 0 {
            println!("  Training supervised pairs...");
        }

        for (i, pair) in dataset.pairs.iter().enumerate() {
            // Supervised learning (semantic vectors, hippocampus, etc.)
            // Optimized: Use batch method and run simulation only periodically
            brain.train_supervised_batch(&pair.input, pair.output.as_deref(), pair.reward);

            // Run full biological simulation only every 50 items to speed up loading
            if (i + 1) % 50 == 0 {
                brain.update(0.1);
            }

            // Apply emotional impact
            brain.apply_emotional_impact(&pair.input);
            if let Some(ref output) = pair.output {
                brain.apply_emotional_impact(output);

                // Store learned response for retrieval
                if pair.reward > 0.0 {
                    let input_intent = brain.detect_intent(&pair.input);
                    // Simplified logic for response intent (could be improved)
                    let response_intent_type = match input_intent {
                        IntentType::Greeting => IntentType::Greeting,
                        IntentType::Question => IntentType::Response,
                        _ => IntentType::Response,
                    };

                    let intent_str = match response_intent_type {
                        IntentType::Greeting => "greeting",
                        IntentType::Response => "response",
                        _ => "response",
                    };

                    brain
                        .ifg_planner
                        .learned_responses
                        .entry(intent_str.to_string())
                        .or_default()
                        .push(output.clone());

                    // Store in Direct Associative Memory
                    let mut triggers = vec![pair.input.clone()];
                    if let Some(alts) = &pair.alternatives {
                        triggers.extend(alts.clone());
                    }

                    for trigger in triggers {
                        let key = trigger
                            .trim()
                            .to_lowercase()
                            .replace("?", "")
                            .replace("!", "")
                            .replace(".", "")
                            .replace(",", "")
                            .trim()
                            .to_string();

                        brain
                            .ifg_planner
                            .direct_memory
                            .entry(key)
                            .or_default()
                            .push((output.clone(), pair.requires_bond));
                    }
                }
            }

            // Progress bar
            if (i + 1) % 10 == 0 || i == total - 1 {
                let progress = (i + 1) as f32 / total as f32;
                let filled = (progress * bar_width as f32) as usize;
                let empty = bar_width - filled;
                print!(
                    "\r  [{}{}] {}/{} ({:.0}%)",
                    "█".repeat(filled),
                    "░".repeat(empty),
                    i + 1,
                    total,
                    progress * 100.0
                );
                stdout().flush().unwrap();
            }
        }

        if total > 0 {
            println!();
        }

        Ok(())
    }

    pub fn load_text_training(
        brain: &mut NeuromorphicBrain,
        file_path: &str,
        window_size: usize,
    ) -> Result<(), Box<dyn Error>> {
        let dataset = TextDataset::from_file(file_path, window_size)?;
        let stats = dataset.stats();
        println!(
            "  Načteno {} vět, {} slov, slovník {} slov",
            stats.sentence_count, stats.total_words, stats.vocab_size
        );

        for word in &dataset.idx_to_word {
            brain.language.ventral.embeddings.add_word(word.clone());
        }

        let pairs = dataset.generate_skipgram_pairs();
        println!("  Generováno {} trénovacích párů", pairs.len());

        let learning_rate = 0.1;
        let batch_size = 100;
        let total_batches = (pairs.len() + batch_size - 1) / batch_size;

        for (batch_idx, chunk) in pairs.chunks(batch_size).enumerate() {
            let brain_pairs: Vec<(usize, usize)> = chunk
                .iter()
                .filter_map(|(c, ctx)| {
                    let center_word = dataset.get_word(*c)?;
                    let context_word = dataset.get_word(*ctx)?;
                    let brain_c = brain.language.get_word_idx(center_word)?;
                    let brain_ctx = brain.language.get_word_idx(context_word)?;
                    Some((brain_c, brain_ctx))
                })
                .collect();

            brain.language.train_on_pairs(&brain_pairs, learning_rate);
            brain.update(0.1);

            if (batch_idx + 1) % 10 == 0 || batch_idx == total_batches - 1 {
                print!(
                    "\r  Trénink: {}/{} batchů ({:.0}%)",
                    batch_idx + 1,
                    total_batches,
                    (batch_idx + 1) as f32 / total_batches as f32 * 100.0
                );
                stdout().flush().unwrap();
            }
        }
        println!();

        Ok(())
    }
}
