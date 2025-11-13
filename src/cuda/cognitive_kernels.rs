//! GPU Kernels for Cognitive Modules
//!
//! All-GPU implementation of attention, working memory, hippocampus, and language.
//! Zero CPU bottlenecks - everything runs on CUDA.

use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync};
use std::sync::Arc;

// ============================================================================
// ATTENTION KERNELS
// ============================================================================

/// Spike correlation kernel for attention (transformer-like)
pub const SPIKE_CORRELATION_KERNEL: &str = r#"
extern "C" __global__ void spike_correlation(
    const float* query,
    const float* keys,
    float* scores,
    const int n_keys,
    const int dim,
    const int query_idx
) {
    int key_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (key_idx >= n_keys) return;

    // Compute dot product
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_k = 0.0f;

    for (int i = 0; i < dim; i++) {
        float q = query[query_idx * dim + i];
        float k = keys[key_idx * dim + i];
        dot += q * k;
        norm_q += q * q;
        norm_k += k * k;
    }

    // Normalized correlation (cosine similarity)
    norm_q = sqrtf(norm_q);
    norm_k = sqrtf(norm_k);

    float score = 0.0f;
    if (norm_q > 1e-8f && norm_k > 1e-8f) {
        score = dot / (norm_q * norm_k);
    }

    scores[key_idx] = score;
}
"#;

/// Softmax kernel for attention weights
pub const SOFTMAX_KERNEL: &str = r#"
extern "C" __global__ void softmax(
    float* scores,
    const int n_scores
) {
    // Each block handles one softmax (for small arrays)
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data to shared memory
    float val = (idx < n_scores) ? scores[idx] : -INFINITY;
    shared_data[tid] = val;
    __syncthreads();

    // Find max for numerical stability (parallel reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    float max_val = shared_data[0];
    __syncthreads();

    // Compute exp(x - max)
    float exp_val = (idx < n_scores) ? expf(scores[idx] - max_val) : 0.0f;
    shared_data[tid] = exp_val;
    __syncthreads();

    // Sum all exp values (parallel reduction)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    float sum = shared_data[0];
    __syncthreads();

    // Normalize
    if (idx < n_scores && sum > 1e-8f) {
        scores[idx] = exp_val / sum;
    }
}
"#;

/// Lateral inhibition kernel for winner-take-all
pub const LATERAL_INHIBITION_KERNEL: &str = r#"
extern "C" __global__ void lateral_inhibition(
    float* scores,
    const int n_scores,
    const float inhibition_strength,
    const float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_scores) return;

    // Compute total activity (sum)
    extern __shared__ float sum_shared[];
    sum_shared[threadIdx.x] = (tid < n_scores) ? scores[tid] : 0.0f;
    __syncthreads();

    // Parallel reduction to get sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_shared[threadIdx.x] += sum_shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total = sum_shared[0];
    __syncthreads();

    // Divisive normalization: output = input / (1 + strength * total)
    float score = scores[tid];
    if (total > 0.0f) {
        score /= (1.0f + inhibition_strength * total);
    }

    // Apply winner-take-all threshold
    if (score < threshold) {
        score = 0.0f;
    }

    scores[tid] = score;
}
"#;

// ============================================================================
// WORKING MEMORY KERNELS
// ============================================================================

/// Cosine similarity kernel for working memory retrieval
pub const COSINE_SIMILARITY_KERNEL: &str = r#"
extern "C" __global__ void cosine_similarity(
    const float* query,
    const float* patterns,
    float* similarities,
    const int n_patterns,
    const int pattern_dim
) {
    int pattern_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pattern_idx >= n_patterns) return;

    // Compute dot product and norms
    float dot = 0.0f;
    float norm_q = 0.0f;
    float norm_p = 0.0f;

    for (int i = 0; i < pattern_dim; i++) {
        float q = query[i];
        float p = patterns[pattern_idx * pattern_dim + i];
        dot += q * p;
        norm_q += q * q;
        norm_p += p * p;
    }

    // Cosine similarity
    norm_q = sqrtf(norm_q);
    norm_p = sqrtf(norm_p);

    float similarity = 0.0f;
    if (norm_q > 1e-8f && norm_p > 1e-8f) {
        similarity = dot / (norm_q * norm_p);
    }

    similarities[pattern_idx] = similarity;
}
"#;

/// Pattern storage kernel for working memory
pub const PATTERN_STORAGE_KERNEL: &str = r#"
extern "C" __global__ void pattern_storage(
    float* persistent_activity,
    const float* pattern,
    const int slot_idx,
    const int pattern_dim,
    const float attention,
    const float attention_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= pattern_dim) return;

    // Only store if attention is high enough
    if (attention < attention_threshold) return;

    // Calculate index in persistent activity array
    int activity_idx = slot_idx * pattern_dim + tid;

    // Store pattern modulated by attention
    persistent_activity[activity_idx] = pattern[tid] * attention;
}
"#;

/// Pattern maintenance kernel for working memory (persistent activity decay)
pub const PATTERN_MAINTENANCE_KERNEL: &str = r#"
extern "C" __global__ void pattern_maintenance(
    float* persistent_activity,
    float* attention_gates,
    const int n_slots,
    const int pattern_dim,
    const float dt,
    const float tau_persistent
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_neurons = n_slots * pattern_dim;

    if (tid >= total_neurons) return;

    // Decay persistent activity
    float decay = expf(-dt / tau_persistent);
    persistent_activity[tid] *= decay;

    // Decay attention gates (very slow)
    if (tid < n_slots) {
        attention_gates[tid] *= 0.999f; // 99.9% retention
    }
}
"#;

// ============================================================================
// LANGUAGE KERNELS
// ============================================================================

/// Token encoding kernel (sparse distributed representation)
pub const TOKEN_ENCODING_KERNEL: &str = r#"
extern "C" __global__ void token_encoding(
    const int* tokens,
    float* patterns,
    const int n_tokens,
    const int pattern_dim,
    const float sparsity_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int token_idx = tid / pattern_dim;
    int dim_idx = tid % pattern_dim;

    if (token_idx >= n_tokens) return;

    int token = tokens[token_idx];

    // Create sparse pattern based on token hash
    // Using simple hash: (token + dim_idx * prime) % 100
    int hash = (token + dim_idx * 37) % 100;
    float val = (float)hash / 100.0f;

    // Apply sparsity (20% active)
    if (val > sparsity_threshold) {
        patterns[tid] = val;
    } else {
        patterns[tid] = 0.0f;
    }
}
"#;

/// Transition matrix update kernel for language learning
pub const TRANSITION_UPDATE_KERNEL: &str = r#"
extern "C" __global__ void transition_update(
    float* transition_matrix,
    const int* tokens,
    const int n_tokens,
    const int vocab_size,
    const float learning_rate
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n_tokens - 1) return;

    int prev_token = tokens[tid];
    int curr_token = tokens[tid + 1];

    if (prev_token >= vocab_size || curr_token >= vocab_size) return;

    // Update transition probability
    int idx = prev_token * vocab_size + curr_token;
    atomicAdd(&transition_matrix[idx], learning_rate);
}
"#;

/// Transition normalization kernel
pub const TRANSITION_NORMALIZE_KERNEL: &str = r#"
extern "C" __global__ void transition_normalize(
    float* transition_matrix,
    const int vocab_size
) {
    int source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source >= vocab_size) return;

    // Compute sum for this source token
    float sum = 0.0f;
    for (int target = 0; target < vocab_size; target++) {
        sum += transition_matrix[source * vocab_size + target];
    }

    // Normalize
    if (sum > 1e-8f) {
        for (int target = 0; target < vocab_size; target++) {
            int idx = source * vocab_size + target;
            transition_matrix[idx] /= sum;
        }
    }
}
"#;

// ============================================================================
// HIPPOCAMPUS KERNELS
// ============================================================================

/// Pattern separation kernel for hippocampus (Dentate Gyrus)
pub const PATTERN_SEPARATION_KERNEL: &str = r#"
extern "C" __global__ void pattern_separation(
    const float* input_pattern,
    float* dg_pattern,
    const int input_dim,
    const int dg_dim,
    const float sparsity,
    const int* random_seeds
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dg_dim) return;

    // Map input to DG with expansion (10×)
    float activation = 0.0f;

    // Random projection with seed-based hashing
    int seed = random_seeds[tid];
    for (int i = 0; i < input_dim; i++) {
        // Simple hash function for connectivity
        int hash = (seed * 31 + i * 17) % 100;
        if (hash < 20) { // 20% connectivity
            float weight = (float)((hash * 13) % 100 - 50) / 50.0f;
            activation += input_pattern[i] * weight;
        }
    }

    // Apply threshold for sparsity (2-5%)
    float threshold = 0.9f; // High threshold for sparse coding
    dg_pattern[tid] = (activation > threshold) ? activation : 0.0f;
}
"#;

/// Pattern completion kernel for hippocampus (CA3)
pub const PATTERN_COMPLETION_KERNEL: &str = r#"
extern "C" __global__ void pattern_completion(
    const float* partial_pattern,
    const float* recurrent_weights,
    float* completed_pattern,
    const int pattern_dim,
    const int n_iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= pattern_dim) return;

    // Iterative pattern completion via recurrent network
    float activation = partial_pattern[tid];

    for (int iter = 0; iter < n_iterations; iter++) {
        float new_activation = 0.0f;

        // Recurrent connections
        for (int j = 0; j < pattern_dim; j++) {
            int weight_idx = tid * pattern_dim + j;
            new_activation += completed_pattern[j] * recurrent_weights[weight_idx];
        }

        // Non-linearity (tanh)
        activation = tanhf(new_activation);

        // Update with momentum
        completed_pattern[tid] = 0.7f * completed_pattern[tid] + 0.3f * activation;

        __syncthreads(); // Synchronize across iterations
    }
}
"#;

/// Hebbian learning kernel for hippocampus
pub const HEBBIAN_LEARNING_KERNEL: &str = r#"
extern "C" __global__ void hebbian_learning(
    float* weights,
    const float* pre_activity,
    const float* post_activity,
    const int n_pre,
    const int n_post,
    const float learning_rate,
    const float weight_decay
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int post_idx = tid / n_pre;
    int pre_idx = tid % n_pre;

    if (post_idx >= n_post) return;

    // Hebbian rule: Δw = η * pre * post - decay * w
    float w = weights[tid];
    float dw = learning_rate * pre_activity[pre_idx] * post_activity[post_idx];
    dw -= weight_decay * w;

    weights[tid] = w + dw;

    // Clamp weights
    weights[tid] = fminf(fmaxf(weights[tid], -1.0f), 1.0f);
}
"#;

// ============================================================================
// KERNEL WRAPPERS
// ============================================================================

/// Wrapper for spike correlation kernel
pub struct SpikeCorrelationKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl SpikeCorrelationKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(SPIKE_CORRELATION_KERNEL)?;
        device.load_ptx(ptx, "cognitive_spike_corr", &["spike_correlation"])?;
        let function = device.get_func("cognitive_spike_corr", "spike_correlation").unwrap();
        Ok(Self { device, function })
    }

    pub fn launch(
        &self,
        config: super::KernelConfig,
        query: &cudarc::driver::CudaSlice<f32>,
        keys: &cudarc::driver::CudaSlice<f32>,
        scores: &mut cudarc::driver::CudaSlice<f32>,
        n_keys: i32,
        dim: i32,
        query_idx: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (query, keys, scores, n_keys, dim, query_idx);
        unsafe {
            self.function.clone().launch(config.to_launch_config(), params)?;
        }
        Ok(())
    }
}

/// Wrapper for cosine similarity kernel
pub struct CosineSimilarityKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl CosineSimilarityKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(COSINE_SIMILARITY_KERNEL)?;
        device.load_ptx(ptx, "cognitive_cosine", &["cosine_similarity"])?;
        let function = device.get_func("cognitive_cosine", "cosine_similarity").unwrap();
        Ok(Self { device, function })
    }

    pub fn launch(
        &self,
        config: super::KernelConfig,
        query: &cudarc::driver::CudaSlice<f32>,
        patterns: &cudarc::driver::CudaSlice<f32>,
        similarities: &mut cudarc::driver::CudaSlice<f32>,
        n_patterns: i32,
        pattern_dim: i32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (query, patterns, similarities, n_patterns, pattern_dim);
        unsafe {
            self.function.clone().launch(config.to_launch_config(), params)?;
        }
        Ok(())
    }
}

/// Wrapper for token encoding kernel
pub struct TokenEncodingKernel {
    device: Arc<CudaDevice>,
    function: CudaFunction,
}

impl TokenEncodingKernel {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, Box<dyn std::error::Error>> {
        let ptx = cudarc::nvrtc::compile_ptx(TOKEN_ENCODING_KERNEL)?;
        device.load_ptx(ptx, "cognitive_token", &["token_encoding"])?;
        let function = device.get_func("cognitive_token", "token_encoding").unwrap();
        Ok(Self { device, function })
    }

    pub fn launch(
        &self,
        config: super::KernelConfig,
        tokens: &cudarc::driver::CudaSlice<i32>,
        patterns: &mut cudarc::driver::CudaSlice<f32>,
        n_tokens: i32,
        pattern_dim: i32,
        sparsity_threshold: f32,
    ) -> Result<(), cudarc::driver::DriverError> {
        let params = (tokens, patterns, n_tokens, pattern_dim, sparsity_threshold);
        unsafe {
            self.function.clone().launch(config.to_launch_config(), params)?;
        }
        Ok(())
    }
}
