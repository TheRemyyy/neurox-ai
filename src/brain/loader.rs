use crate::brain::datasets::{JsonDataset, TextDataset};
use crate::brain::language::{IntentType, PartOfSpeech, PragmaticRule, SentenceTemplate};
use crate::brain::NeuromorphicBrain;
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
            let mut annotated = crate::brain::language::AnnotatedWord::new(&vocab_word.word, pos)
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
            brain.language.ventral.embeddings.add_word(&vocab_word.word);
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

        // === STEP 4.5: Load Sentiment Patterns ===
        if let Some(ref patterns) = dataset.sentiment_patterns {
            let negative_count = patterns
                .negative
                .as_ref()
                .map(|n| n.keywords.len())
                .unwrap_or(0);
            let positive_count = patterns
                .positive
                .as_ref()
                .map(|p| p.keywords.len())
                .unwrap_or(0);
            println!(
                "  Loading sentiment patterns: {} negative, {} positive keywords",
                negative_count, positive_count
            );
            brain.sentiment_patterns = Some(patterns.clone());
        }

        // === STEP 4.6: Load Emotion Triggers ===
        if !dataset.emotion_triggers.is_empty() {
            println!(
                "  Loading emotion triggers: {} rules",
                dataset.emotion_triggers.len()
            );
            for trigger in &dataset.emotion_triggers {
                let emotion = match trigger.emotion.as_str() {
                    "joy" => crate::brain::affect::Emotion::Joy,
                    "sadness" => crate::brain::affect::Emotion::Sadness,
                    "fear" => crate::brain::affect::Emotion::Fear,
                    "anger" => crate::brain::affect::Emotion::Anger,
                    "surprise" => crate::brain::affect::Emotion::Surprise,
                    "disgust" => crate::brain::affect::Emotion::Disgust,
                    "trust" => crate::brain::affect::Emotion::Trust,
                    "anticipation" => crate::brain::affect::Emotion::Anticipation,
                    "love" => crate::brain::affect::Emotion::Love,
                    _ => crate::brain::affect::Emotion::Neutral,
                };
                brain
                    .emotional_state
                    .add_transition(crate::brain::affect::EmotionTransition {
                        trigger_threshold: 0.5,
                        from_emotion: None,
                        to_emotion: emotion,
                        intensity: trigger.intensity,
                        trigger_patterns: trigger.trigger_patterns.clone(),
                    });
            }
        }

        // === STEP 5: Train supervised pairs for semantic learning ===
        let total = dataset.pairs.len();

        if total > 0 {
            println!("  Training supervised pairs...");
        }

        // OPTIMIZED: Process in batches with minimal biological simulation
        let progress_bar = indicatif::ProgressBar::new(total as u64);
        progress_bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("  [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        progress_bar.set_message("Training pairs");

        for (i, pair) in dataset.pairs.iter().enumerate() {
            // Ensure words are in lexicon for word-by-word generation fallback
            let mut ensure_lexicon = |text: &str| {
                for word in text.split_whitespace() {
                    let clean: String = word
                        .to_lowercase()
                        .chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect();
                    if !clean.is_empty() && brain.lexicon.get_by_text(&clean).is_none() {
                        brain
                            .lexicon
                            .add_word(crate::brain::language::AnnotatedWord::new(
                                &clean,
                                PartOfSpeech::Unknown,
                            ));
                    }
                }
            };
            ensure_lexicon(&pair.input);
            if let Some(ref output) = pair.output {
                ensure_lexicon(output);
            }

            // Supervised learning (semantic vectors, hippocampus, assoc, dorsal)
            brain.train_supervised_batch(&pair.input, pair.output.as_deref(), pair.reward);

            // Run biological simulation periodically to keep state consistent
            // 100 is a good balance for quality vs performance
            if (i + 1) % 100 == 0 {
                brain.update(0.1);
            }

            // Apply emotional impact periodically to maintain neurochemical state
            if (i + 1) % 50 == 0 {
                brain.apply_emotional_impact(&pair.input);
                if let Some(ref output) = pair.output {
                    brain.apply_emotional_impact(output);
                }
            }

            if let Some(ref output) = pair.output {
                // Register as potentially learned response
                if pair.reward > 0.0 {
                    // Map category to response intent for generate_sentence lookup
                    let category = pair.category.as_deref().unwrap_or("response");
                    let intent_str = match category {
                        "greeting" => "greeting",
                        "farewell" => "farewell",
                        "thanks" => "thanks",
                        "emotional" => "emotional",
                        "insult" => "insult",
                        // Most categories map to generic "response" for statement/question replies
                        "small_talk" | "identity" | "preferences" | "reaction" | "confirmation"
                        | "uncertainty" | "help" | "opinion" | "clarification" | "world"
                        | "humor" | "personal" | "knowledge" | "future" | "culture" | "meta"
                        | "agreement" | "disagreement" | "pause" | "philosophy" | "tech"
                        | "weather" | "request" | "meta_ai" | "response" => "response",
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

                    // Store input embedding for semantic similarity matching
                    // This enables finding contextually similar responses
                    let input_indices: Vec<usize> = pair
                        .input
                        .split_whitespace()
                        .filter_map(|word| {
                            let clean: String = word
                                .to_lowercase()
                                .chars()
                                .filter(|c| c.is_alphanumeric())
                                .collect();
                            if clean.is_empty() {
                                None
                            } else {
                                brain.language.get_word_idx(&clean)
                            }
                        })
                        .collect();

                    if !input_indices.is_empty() {
                        let embedding = brain.language.comprehend(&input_indices);
                        brain.ifg_planner.semantic_memory.push((
                            embedding,
                            output.clone(),
                            pair.requires_bond,
                        ));
                    }
                }
            }
            progress_bar.inc(1);
        }

        progress_bar.finish_with_message("Training complete");
        brain.update(1.0); // Final deep update to settle state

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
            "  Loaded {} sentences, {} words, vocabulary of {} words",
            stats.sentence_count, stats.total_words, stats.vocab_size
        );

        for word in &dataset.idx_to_word {
            brain.language.ventral.embeddings.add_word(word);
        }

        let pairs = dataset.generate_skipgram_pairs();
        println!("  Generated {} training pairs", pairs.len());

        let learning_rate = 0.1;
        let batch_size = 100;
        let total_batches = pairs.len().div_ceil(batch_size);

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
                    "\r  Training: {}/{} batches ({:.0}%)",
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
