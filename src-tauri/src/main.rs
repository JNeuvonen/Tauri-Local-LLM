// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod model_config;
mod server;

use model_config::Configuration;
use std::{
    convert::Infallible,
    io::Write,
    sync::{Arc, Mutex},
    thread,
};
fn main() -> Result<(), anyhow::Error> {
    let config = Configuration::default();

    let model = llm::load_dynamic(
        config.model.architecture(),
        &config.model.path,
        llm::TokenizerSource::Embedded,
        llm::ModelParameters {
            prefer_mmap: config.model.prefer_mmap,
            context_size: config.model.context_token_length,
            use_gpu: config.model.use_gpu,
            gpu_layers: config.model.gpu_layers,
            ..Default::default()
        },
        llm::load_progress_callback_stdout,
    )?;

    let mut session = model.start_session(Default::default());
    let session = Arc::new(Mutex::new(session));
    let model = Arc::new(Mutex::new(model));

    let session_for_thread = Arc::clone(&session);
    let model_for_thread = Arc::clone(&model);

    let server_handle = thread::spawn(move || {
        server::run_server(session_for_thread, model_for_thread).expect("Failed to start server");
    });

    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running tauri application");

    Ok(())
}
