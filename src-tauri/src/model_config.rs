use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Configuration {
    pub model: Model,
}
impl Default for Configuration {
    fn default() -> Self {
        Self {
            model: Model {
                path: "models/llama-2-13b-chat.ggmlv3.q4_0.bin".into(),
                context_token_length: 2000,
                architecture: llm::ModelArchitecture::Llama.to_string(),
                prefer_mmap: true,
                use_gpu: true,
                gpu_layers: Some(50000),
            },
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Model {
    pub path: PathBuf,
    pub context_token_length: usize,
    pub architecture: String,
    pub prefer_mmap: bool,
    pub use_gpu: bool,
    pub gpu_layers: Option<usize>,
}
impl Model {
    pub fn architecture(&self) -> Option<llm::ModelArchitecture> {
        self.architecture.parse().ok()
    }
}
