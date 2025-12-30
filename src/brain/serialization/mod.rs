//! Neuromorphic Model Serialization
//!
//! Binary format for spiking neural network models with biological dynamics.
//! This stores biological dynamics: spikes, plasticity, temporal state.
//!
//! Format (.nrx binary):
//! - Sparse synaptic connectivity (CSR)
//! - Dynamic weights (trained via STDP)
//! - Neuron parameters (thresholds, time constants)
//! - Plasticity traces (short-term & long-term)
//! - Homeostatic state
//!
//! Designed for neuromorphic computing, not traditional deep learning.

use crate::brain::connectivity::SparseConnectivity;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic bytes for neuromorphic model files
const MAGIC: &[u8; 4] = b"NRXB"; // NeuroX Binary

/// Current serialization version
const VERSION: u32 = 1;

/// Complete neuromorphic model state with biological dynamics
#[derive(Debug)]
pub struct NeuromorphicModel {
    /// Network metadata
    pub metadata: ModelMetadata,

    /// Sparse synaptic connectivity (CSR format)
    pub connectivity: SparseConnectivity,

    /// Neuron parameters (biological, not activations!)
    pub neuron_params: NeuronParameters,

    /// Plasticity state (STDP traces, STP dynamics)
    pub plasticity: PlasticityState,
}

/// Network architecture metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Total neurons
    pub n_neurons: usize,

    /// Total synapses
    pub n_synapses: usize,

    /// Simulation timestep (ms)
    pub dt: f32,

    /// Network architecture description
    pub architecture: String,

    /// Training info
    pub training_epochs: usize,
    pub final_accuracy: f32,
}

/// Biological neuron parameters
#[derive(Debug, Clone)]
pub struct NeuronParameters {
    /// Spike thresholds (mV)
    pub thresholds: Vec<f32>,

    /// Membrane time constants (ms)
    pub tau_m: Vec<f32>,

    /// Reset potentials (mV)
    pub v_reset: Vec<f32>,

    /// Current membrane potentials (mV) - dynamic state!
    pub membrane_v: Option<Vec<f32>>,
}

/// Synaptic plasticity state for biological learning mechanisms
#[derive(Debug, Clone)]
pub struct PlasticityState {
    /// STDP pre-synaptic traces
    pub pre_traces: Option<Vec<f32>>,

    /// STDP post-synaptic traces (first)
    pub post_traces_1: Option<Vec<f32>>,

    /// STDP post-synaptic traces (second, for triplet rule)
    pub post_traces_2: Option<Vec<f32>>,

    /// STP facilitation state
    pub stp_u: Option<Vec<f32>>,

    /// STP depression state
    pub stp_x: Option<Vec<f32>>,
}

impl NeuromorphicModel {
    /// Create new model from components
    pub fn new(
        metadata: ModelMetadata,
        connectivity: SparseConnectivity,
        neuron_params: NeuronParameters,
        plasticity: PlasticityState,
    ) -> Self {
        Self {
            metadata,
            connectivity,
            neuron_params,
            plasticity,
        }
    }

    /// Save model to binary file (.nrx format)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;

        // Write metadata
        self.write_metadata(&mut writer)?;

        // Write sparse connectivity (CSR)
        self.write_connectivity(&mut writer)?;

        // Write neuron parameters
        self.write_neuron_params(&mut writer)?;

        // Write plasticity state
        self.write_plasticity(&mut writer)?;

        writer.flush()?;
        Ok(())
    }

    /// Load model from binary file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err("Invalid neuromorphic model file (wrong magic bytes)".into());
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(format!("Unsupported model version: {}", version).into());
        }

        // Read all components
        let metadata = Self::read_metadata(&mut reader)?;
        let connectivity = Self::read_connectivity(&mut reader)?;
        let neuron_params = Self::read_neuron_params(&mut reader)?;
        let plasticity = Self::read_plasticity(&mut reader)?;

        Ok(Self::new(metadata, connectivity, neuron_params, plasticity))
    }

    // === Write methods ===

    fn write_metadata<W: Write>(&self, writer: &mut W) -> Result<(), Box<dyn std::error::Error>> {
        writer.write_all(&self.metadata.n_neurons.to_le_bytes())?;
        writer.write_all(&self.metadata.n_synapses.to_le_bytes())?;
        writer.write_all(&self.metadata.dt.to_le_bytes())?;

        // Architecture string
        let arch_bytes = self.metadata.architecture.as_bytes();
        writer.write_all(&(arch_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(arch_bytes)?;

        writer.write_all(&self.metadata.training_epochs.to_le_bytes())?;
        writer.write_all(&self.metadata.final_accuracy.to_le_bytes())?;

        Ok(())
    }

    fn write_connectivity<W: Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Write CSR arrays
        writer.write_all(&(self.connectivity.row_ptr.len() as u32).to_le_bytes())?;
        for &val in &self.connectivity.row_ptr {
            writer.write_all(&val.to_le_bytes())?;
        }

        writer.write_all(&(self.connectivity.col_idx.len() as u32).to_le_bytes())?;
        for &val in &self.connectivity.col_idx {
            writer.write_all(&val.to_le_bytes())?;
        }

        writer.write_all(&(self.connectivity.weights.len() as u32).to_le_bytes())?;
        for &val in &self.connectivity.weights {
            writer.write_all(&val.to_le_bytes())?;
        }

        Ok(())
    }

    fn write_neuron_params<W: Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Thresholds
        writer.write_all(&(self.neuron_params.thresholds.len() as u32).to_le_bytes())?;
        for &val in &self.neuron_params.thresholds {
            writer.write_all(&val.to_le_bytes())?;
        }

        // Tau_m
        writer.write_all(&(self.neuron_params.tau_m.len() as u32).to_le_bytes())?;
        for &val in &self.neuron_params.tau_m {
            writer.write_all(&val.to_le_bytes())?;
        }

        // V_reset
        writer.write_all(&(self.neuron_params.v_reset.len() as u32).to_le_bytes())?;
        for &val in &self.neuron_params.v_reset {
            writer.write_all(&val.to_le_bytes())?;
        }

        // Optional membrane potentials
        self.write_optional_vec(writer, &self.neuron_params.membrane_v)?;

        Ok(())
    }

    fn write_plasticity<W: Write>(&self, writer: &mut W) -> Result<(), Box<dyn std::error::Error>> {
        self.write_optional_vec(writer, &self.plasticity.pre_traces)?;
        self.write_optional_vec(writer, &self.plasticity.post_traces_1)?;
        self.write_optional_vec(writer, &self.plasticity.post_traces_2)?;
        self.write_optional_vec(writer, &self.plasticity.stp_u)?;
        self.write_optional_vec(writer, &self.plasticity.stp_x)?;
        Ok(())
    }

    fn write_optional_vec<W: Write>(
        &self,
        writer: &mut W,
        vec: &Option<Vec<f32>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match vec {
            Some(v) => {
                writer.write_all(&1u8.to_le_bytes())?; // Present flag
                writer.write_all(&(v.len() as u32).to_le_bytes())?;
                for &val in v {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
            None => {
                writer.write_all(&0u8.to_le_bytes())?; // Not present
            }
        }
        Ok(())
    }

    // === Read methods ===

    fn read_metadata<R: Read>(reader: &mut R) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
        let n_neurons = Self::read_usize(reader)?;
        let n_synapses = Self::read_usize(reader)?;
        let dt = Self::read_f32(reader)?;

        let arch_len = Self::read_u32(reader)? as usize;
        let mut arch_bytes = vec![0u8; arch_len];
        reader.read_exact(&mut arch_bytes)?;
        let architecture = String::from_utf8(arch_bytes)?;

        let training_epochs = Self::read_usize(reader)?;
        let final_accuracy = Self::read_f32(reader)?;

        Ok(ModelMetadata {
            n_neurons,
            n_synapses,
            dt,
            architecture,
            training_epochs,
            final_accuracy,
        })
    }

    fn read_connectivity<R: Read>(
        reader: &mut R,
    ) -> Result<SparseConnectivity, Box<dyn std::error::Error>> {
        let row_ptr = Self::read_vec_i32(reader)?;
        let col_idx = Self::read_vec_i32(reader)?;
        let weights = Self::read_vec_f32(reader)?;

        let nnz = weights.len();
        let n_neurons = row_ptr.len() - 1;

        Ok(SparseConnectivity {
            row_ptr,
            col_idx,
            weights,
            nnz,
            n_neurons,
        })
    }

    fn read_neuron_params<R: Read>(
        reader: &mut R,
    ) -> Result<NeuronParameters, Box<dyn std::error::Error>> {
        let thresholds = Self::read_vec_f32(reader)?;
        let tau_m = Self::read_vec_f32(reader)?;
        let v_reset = Self::read_vec_f32(reader)?;
        let membrane_v = Self::read_optional_vec_f32(reader)?;

        Ok(NeuronParameters {
            thresholds,
            tau_m,
            v_reset,
            membrane_v,
        })
    }

    fn read_plasticity<R: Read>(
        reader: &mut R,
    ) -> Result<PlasticityState, Box<dyn std::error::Error>> {
        let pre_traces = Self::read_optional_vec_f32(reader)?;
        let post_traces_1 = Self::read_optional_vec_f32(reader)?;
        let post_traces_2 = Self::read_optional_vec_f32(reader)?;
        let stp_u = Self::read_optional_vec_f32(reader)?;
        let stp_x = Self::read_optional_vec_f32(reader)?;

        Ok(PlasticityState {
            pre_traces,
            post_traces_1,
            post_traces_2,
            stp_u,
            stp_x,
        })
    }

    // Helper read functions

    fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Box<dyn std::error::Error>> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_usize<R: Read>(reader: &mut R) -> Result<usize, Box<dyn std::error::Error>> {
        let mut bytes = [0u8; 8];
        reader.read_exact(&mut bytes)?;
        Ok(usize::from_le_bytes(bytes))
    }

    fn read_f32<R: Read>(reader: &mut R) -> Result<f32, Box<dyn std::error::Error>> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_vec_i32<R: Read>(reader: &mut R) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
        let len = Self::read_u32(reader)? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            vec.push(i32::from_le_bytes(bytes));
        }
        Ok(vec)
    }

    fn read_vec_f32<R: Read>(reader: &mut R) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let len = Self::read_u32(reader)? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(Self::read_f32(reader)?);
        }
        Ok(vec)
    }

    fn read_optional_vec_f32<R: Read>(
        reader: &mut R,
    ) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut flag = [0u8; 1];
        reader.read_exact(&mut flag)?;

        if flag[0] == 1 {
            Ok(Some(Self::read_vec_f32(reader)?))
        } else {
            Ok(None)
        }
    }
}
