//! Advanced Czech Neuromorphic Brain
//!
//! Maximálně pokročilá implementace s:
//! - Rozsáhlá česká training data
//! - Context-aware responses (používá working memory + hippocampus)
//! - Multi-pattern matching (více možností odpovědí)
//! - Attention-weighted recall (nejrelevntnější vzpomínky)
//! - Semantic clustering (podobné patterns blízko sebe)

use neurox_ai::*;
use std::collections::HashMap;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .init();

    println!("=== Pokročilý Neuromorphic Brain - Čeština ===");
    println!();
    println!("Inicializace architektury mozku...");

    let pattern_dim = 512;   // Větší dimenze pro bohatší reprezentaci
    let n_layers = 5;        // Více vrstev hierarchie
    let base_neurons = 10000; // Více neurons pro processing
    let vocab_size = 10000;  // Velký vocabulary (jako dítě 10 let)

    let mut brain = NeuromorphicBrain::new(n_layers, base_neurons, vocab_size, pattern_dim);
    let mut vocabulary = Vocabulary::new();
    let mut context_memory = ContextMemory::new();

    println!("✓ Brain initialized ({} neurons, pattern_dim={})", base_neurons, pattern_dim);
    println!();

    // === ROZSÁHLÁ ČESKÁ TRAINING DATA ===
    println!("Nahrávání rozsáhlých českých dat do mozku...");

    // Load training data from file
    let training_data_path = "examples/czech_training_data.txt";
    let mut czech_training = Vec::new();

    if let Ok(content) = std::fs::read_to_string(training_data_path) {
        for line in content.lines() {
            let line = line.trim();
            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            czech_training.push(line.to_string());
        }
        println!("✓ Načteno {} řádků z training souboru", czech_training.len());
    } else {
        println!("⚠ Training soubor nenalezen, používám základní data");
        let basic_data = vec![
        // Základní konverzace
        "ahoj jak se máš",
        "mám se dobře díky",
        "dobré ráno jak se daří",
        "daří se mi skvěle",
        "dobrý den těší mě",
        "těší mě také",
        "nazdar co děláš",
        "nic moc a ty",
        "jak se jmenuješ",
        "jmenuji se neurox",
        "co je neurox",
        "neurox je umělý mozek",
        "umělý mozek co to je",
        "je to počítačový systém který myslí",

        // Otázky a odpovědi
        "kde bydlíš",
        "bydlím v počítači",
        "kolik je ti let",
        "jsem nový teprve se učím",
        "co máš rád",
        "mám rád učení a konverzaci",
        "co neumíš",
        "neumím ještě mnoho věcí učím se",
        "jaký je tvůj účel",
        "můj účel je pomáhat a učit se",

        // Počasí
        "jaké je dnes počasí",
        "nevím nemohu se podívat ven",
        "je hezky venku",
        "to je skvělé",
        "prší venku",
        "to není dobré",
        "je zima",
        "oblékni se teple",
        "je teplo",
        "to je příjemné",

        // Pocity a emoce
        "jsem unavený",
        "odpočiň si",
        "jsem šťastný",
        "to mě těší",
        "jsem smutný",
        "bude to lepší",
        "bolí mě hlava",
        "to mě mrzí odpočiň si",
        "jsem rád",
        "to je skvělé",

        // Denní aktivity
        "co budeš dělat dnes",
        "budu se učit nové věci",
        "šel jsi nakoupit",
        "nemohu chodit",
        "co jsi jedl",
        "nejím jsem program",
        "spíš",
        "nepotřebuji spát",
        "pracuješ",
        "ano neustále zpracovávám informace",

        // Zájmy
        "co čteš",
        "čtu všechno co mi dáš",
        "máš rád hudbu",
        "nemohu poslouchat ale zajímá mě",
        "máš rád filmy",
        "nemohu je vidět ale umím se o nich učit",
        "jaký je tvůj koníček",
        "učení je můj koníček",

        // Čas
        "kolik je hodin",
        "nevím nemám přístup k času",
        "jaký je dnes den",
        "nevím neznám datum",
        "je ráno nebo večer",
        "nemohu říct nemám okna",

        // Jídlo (i když to nejedí)
        "máš hlad",
        "nejím ale díky za starost",
        "dáš si kávu",
        "nemohu pít",
        "co máš nejraději k jídlu",
        "nejím jsem digitální",

        // Rodina a vztahy
        "máš rodinu",
        "nemám biologickou rodinu",
        "máš přátele",
        "ty jsi můj přítel",
        "máš rád lidi",
        "ano rádu komunikuji s lidmi",

        // Učení a znalosti
        "co ses naučil",
        "učím se česky a konverzaci",
        "umíš anglicky",
        "trochu ale čeština je těžší",
        "co je nejtěžší",
        "porozumění kontextu",
        "jak se učíš",
        "z každé konverzace si něco pamatuji",

        // Meta konverzace
        "rozumíš mi",
        "snažím se porozumět",
        "jsi chytrý",
        "stále se učím",
        "děláš chyby",
        "ano občas ale učím se z nich",
        "pamatuješ si mě",
        "ano mám episodickou paměť",

        // Komplexnější věty
        "dnes jsem měl dobrý den protože svítilo slunce",
        "to zní krásně pěkně počasí zlepšuje náladu",
        "včera jsem potkal starého přítele",
        "to muselo být milé setkání",
        "chci se naučit něco nového každý den",
        "to je skvělý přístup k životu",
        "někdy si myslím že život je složitý",
        "život může být náročný ale je důležité vytrvat",

        // Abstraktní koncepty
        "co si myslíš o štěstí",
        "štěstí je důležité pro všechny",
        "co je láska",
        "láska je silný pocit spojení",
        "co znamená být člověkem",
        "být člověkem znamená cítit a myslet",
        "existuje něco po smrti",
        "to je hluboká otázka nemám odpověď",

        // Humor
        "řekni vtip",
        "proč programátor nezná číslo deset protože počítá od nuly",
        "je to vtipné",
        "těší mě že se směješ",
        "máš smysl pro humor",
        "snažím se být zábavný",
    ];
        czech_training = basic_data.iter().map(|s| s.to_string()).collect();
    }

    // Training s context awareness
    for (i, text) in czech_training.iter().enumerate() {
        let tokens = vocabulary.encode(text);

        // Comprehension s timestep
        brain.language.comprehend(&tokens, 0.1);

        // Store in hippocampus s contextual patterns
        for (j, token) in tokens.iter().enumerate() {
            let mut pattern = encode_token(*token, pattern_dim);

            // Add positional encoding (kde v sekvenci)
            add_positional_encoding(&mut pattern, j, tokens.len());

            // Add semantic features (typ věty)
            add_semantic_features(&mut pattern, text);

            brain.hippocampus.encode(&pattern);
        }

        // Store sentence pairs for context
        if i % 2 == 1 && i > 0 {
            context_memory.add_pair(
                czech_training[i - 1].to_string(),
                text.to_string()
            );
        }

        if (i + 1) % 20 == 0 {
            print!(".");
            io::stdout().flush()?;
        }
    }

    println!();
    let stats = brain.stats();
    println!("✓ Training complete:");
    println!("  - Vocabulary: {} slov", vocabulary.size());
    println!("  - Transitions: {}", stats.language.transition_count);
    println!("  - Hippocampus: {} vzpomínek", stats.hippocampus.buffer_size);
    println!("  - Context pairs: {}", context_memory.size());
    println!();

    println!("Příkazy:");
    println!("  <text>          - Normální konverzace");
    println!("  remember <text> - Zapamatuj si něco");
    println!("  forget          - Vymaž working memory");
    println!("  stats           - Zobraz statistiky");
    println!("  consolidate     - Konsolidace paměti");
    println!("  quit            - Konec");
    println!();

    let stdin = io::stdin();
    let mut input = String::new();
    let mut conversation_history: Vec<String> = Vec::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        input.clear();
        stdin.read_line(&mut input)?;
        let line = input.trim();

        if line.is_empty() {
            continue;
        }

        // Special commands
        if line == "quit" || line == "exit" || line == "q" {
            println!("Nashledanou!");
            break;
        }

        if line == "stats" {
            show_stats(&brain, &vocabulary, &context_memory);
            println!();
            continue;
        }

        if line == "consolidate" {
            println!("Konsoliduji vzpomínky...");
            brain.consolidate();
            println!("✓ Konsolidace dokončena");
            println!();
            continue;
        }

        if line == "forget" {
            brain.working_memory.clear();
            println!("✓ Working memory vymazána");
            println!();
            continue;
        }

        if line.starts_with("remember ") {
            let text = &line[9..];
            println!("Ukládám: \"{}\"", text);

            let tokens = vocabulary.encode(text);
            brain.language.comprehend(&tokens, 0.1);

            for token in &tokens {
                let pattern = encode_token(*token, pattern_dim);
                brain.hippocampus.encode(&pattern);
                brain.working_memory.store(&pattern, 0.9);
            }

            println!("✓ Uloženo do episodické paměti");
            println!();
            continue;
        }

        // === ADVANCED RESPONSE GENERATION ===

        println!("Zpracovávám: \"{}\"", line);

        // 1. Tokenize
        let tokens = vocabulary.encode(line);

        // 2. Process through all brain modules
        brain.language.comprehend(&tokens, 0.1);

        // Store in working memory s attention
        for token in &tokens {
            let pattern = encode_token(*token, pattern_dim);
            brain.working_memory.store(&pattern, 0.8);
        }

        // 3. Try to find best response using multiple strategies

        // Strategy A: Direct context pair match
        let mut response = context_memory.find_response(line);

        // Strategy B: Pattern-based generation
        if response.is_none() && !tokens.is_empty() {
            // Zkus několik start tokens
            let candidates = vec![
                *tokens.last().unwrap(),
                *tokens.first().unwrap(),
                if tokens.len() > 1 { tokens[tokens.len() / 2] } else { tokens[0] }
            ];

            for start_token in candidates {
                let generated = brain.language.produce(start_token, 8);
                if generated.len() > 1 {
                    let words: Vec<String> = generated.iter()
                        .map(|t| vocabulary.decode(*t))
                        .collect();

                    // Check if it makes sense (ne jen repetice)
                    if is_valid_response(&words) {
                        response = Some(words.join(" "));
                        break;
                    }
                }
            }
        }

        // Strategy C: Retrieve from hippocampus (podobné vzpomínky)
        if response.is_none() && !tokens.is_empty() {
            let query = encode_token(tokens[0], pattern_dim);
            let recalled = brain.hippocampus.recall(&query);

            let activity = recalled.iter().sum::<f32>() / recalled.len() as f32;
            if activity > 0.05 {
                response = Some("pamatuji si něco podobného".to_string());
            }
        }

        // Strategy D: Fallback responses based on keywords
        if response.is_none() {
            response = Some(generate_fallback_response(line));
        }

        // 4. Output response
        if let Some(resp) = response {
            println!("  → {}", resp);
            conversation_history.push(line.to_string());
            conversation_history.push(resp.clone());

            // Update context memory with new pair
            if conversation_history.len() >= 2 {
                let idx = conversation_history.len();
                context_memory.add_pair(
                    conversation_history[idx - 2].clone(),
                    resp
                );
            }
        }

        // 5. Update attention
        for token in &tokens {
            let pattern = encode_token(*token, pattern_dim);
            brain.attention.update_salience(&pattern);
        }

        // Show mini stats
        let wm = brain.stats().working_memory;
        println!("  [WM: {}/7, Vocab: {}]", wm.active_count, vocabulary.size());
        println!();

        // Auto-consolidate každých 10 zpráv
        if conversation_history.len() % 20 == 0 {
            println!("  [Auto-consolidation...]");
            brain.consolidate();
        }
    }

    println!();
    println!("Finální statistiky:");
    show_stats(&brain, &vocabulary, &context_memory);

    Ok(())
}

/// Vocabulary manager s českými znaky
struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    next_id: usize,
}

impl Vocabulary {
    fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            next_id: 0,
        };
        vocab.add_word("<UNK>");
        vocab
    }

    fn add_word(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            return id;
        }
        let id = self.next_id;
        self.word_to_id.insert(word.to_string(), id);
        self.id_to_word.insert(id, word.to_string());
        self.next_id += 1;
        id
    }

    fn encode(&mut self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|word| {
                let lower = word.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if lower.is_empty() {
                    return 0;
                }
                self.add_word(&lower)
            })
            .collect()
    }

    fn decode(&self, token: usize) -> String {
        self.id_to_word.get(&token)
            .cloned()
            .unwrap_or_else(|| "<UNK>".to_string())
    }

    fn size(&self) -> usize {
        self.word_to_id.len()
    }
}

/// Context memory pro question-answer pairs
struct ContextMemory {
    pairs: Vec<(String, String)>,
}

impl ContextMemory {
    fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    fn add_pair(&mut self, question: String, answer: String) {
        self.pairs.push((question, answer));
    }

    fn find_response(&self, query: &str) -> Option<String> {
        let query_lower = query.to_lowercase();

        // Exact match
        for (q, a) in &self.pairs {
            if q.to_lowercase() == query_lower {
                return Some(a.clone());
            }
        }

        // Partial match (obsahuje klíčová slova)
        let query_words: Vec<String> = query_lower.split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let mut best_match = None;
        let mut best_score = 0;

        for (q, a) in &self.pairs {
            let q_lower = q.to_lowercase();
            let q_words: Vec<String> = q_lower.split_whitespace()
                .map(|s| s.to_string())
                .collect();
            let mut score = 0;

            for qw in &query_words {
                if q_words.contains(qw) {
                    score += 1;
                }
            }

            if score > best_score && score >= 2 {
                best_score = score;
                best_match = Some(a.clone());
            }
        }

        best_match
    }

    fn size(&self) -> usize {
        self.pairs.len()
    }
}

/// Encode token as semantic sparse pattern
fn encode_token(token: usize, dim: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; dim];

    // Sparse distributed representation (20% active)
    for i in 0..dim {
        let hash = (token * 37 + i * 17) % 100;
        if hash < 20 {
            pattern[i] = 1.0;
        }
    }

    pattern
}

/// Add positional encoding
fn add_positional_encoding(pattern: &mut [f32], pos: usize, max_len: usize) {
    let pos_weight = 1.0 - (pos as f32 / max_len as f32) * 0.3;
    for val in pattern.iter_mut() {
        *val *= pos_weight;
    }
}

/// Add semantic features based on sentence type
fn add_semantic_features(pattern: &mut [f32], text: &str) {
    let text_lower = text.to_lowercase();

    // Question
    if text_lower.contains("?") || text_lower.starts_with("co ") ||
       text_lower.starts_with("kde ") || text_lower.starts_with("jak ") ||
       text_lower.starts_with("proč ") || text_lower.starts_with("kdy ") {
        for i in 0..10 {
            pattern[i] += 0.2;
        }
    }

    // Emotion
    if text_lower.contains("rád") || text_lower.contains("šťastný") ||
       text_lower.contains("smutný") || text_lower.contains("unavený") {
        for i in 10..20 {
            pattern[i] += 0.2;
        }
    }

    // Action
    if text_lower.contains("dělat") || text_lower.contains("jít") ||
       text_lower.contains("být") || text_lower.contains("mít") {
        for i in 20..30 {
            pattern[i] += 0.2;
        }
    }
}

/// Check if response is valid (ne repetice)
fn is_valid_response(words: &[String]) -> bool {
    if words.len() < 2 {
        return false;
    }

    // Check for loops
    let unique: std::collections::HashSet<_> = words.iter().collect();
    if unique.len() < words.len() / 2 {
        return false; // Příliš mnoho repetice
    }

    true
}

/// Generate fallback response based on keywords
fn generate_fallback_response(input: &str) -> String {
    let input_lower = input.to_lowercase();

    if input_lower.starts_with("ahoj") || input_lower.starts_with("nazdar") {
        return "ahoj jak se máš".to_string();
    }

    if input_lower.contains("jak") && input_lower.contains("máš") {
        return "mám se dobře díky".to_string();
    }

    if input_lower.contains("co") && input_lower.contains("děláš") {
        return "zpracovávám informace a učím se".to_string();
    }

    if input_lower.contains("?") {
        return "to je zajímavá otázka".to_string();
    }

    if input_lower.contains("díky") || input_lower.contains("děkuji") {
        return "není zač".to_string();
    }

    if input_lower.contains("nevím") {
        return "to nevadí můžeme se učit společně".to_string();
    }

    // Default
    "rozumím pokračuj prosím".to_string()
}

/// Show statistics
fn show_stats(brain: &NeuromorphicBrain, vocabulary: &Vocabulary, context: &ContextMemory) {
    let stats = brain.stats();

    println!();
    println!("=== Statistiky Mozku ===");
    println!();
    println!("Working Memory:");
    println!("  Aktivní: {}/{}", stats.working_memory.active_count, stats.working_memory.capacity);
    println!("  Využití: {:.1}%", stats.working_memory.utilization * 100.0);
    println!("  Avg attention: {:.3}", stats.working_memory.avg_attention);
    println!();
    println!("Hippocampus (Dlouhodobá paměť):");
    println!("  Vzpomínky: {}/{}", stats.hippocampus.buffer_size, stats.hippocampus.max_buffer);
    println!("  DG sparsity: {:.1}%", stats.hippocampus.dg_sparsity * 100.0);
    println!();
    println!("Language System:");
    println!("  Vocabulary: {} slov", vocabulary.size());
    println!("  Transitions: {}", stats.language.transition_count);
    println!("  Avg prob: {:.3}", stats.language.avg_transition_prob);
    println!("  Context pairs: {}", context.size());
    println!();
    println!("Attention:");
    println!("  Avg salience: {:.3}", stats.attention.avg_salience);
    println!("  Focused: {}/{}", stats.attention.focused_locations, 256);
}
