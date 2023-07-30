use actix::{Actor, Addr, Context, Handler, Message};
use actix_web::{
    rt,
    web::{self, Data, Json},
    App, HttpResponse, HttpServer, Responder,
};
use llm::{InferenceSession, Model};
use serde::Deserialize;
use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
};

#[derive(Deserialize)]
pub struct CompletionRequest {
    prompt_tokens: String,
    completion_max_len: String,
}

struct InferenceActor {
    session: Arc<Mutex<InferenceSession>>,
    model: Arc<Mutex<Box<dyn Model>>>,
}

impl Actor for InferenceActor {
    type Context = Context<Self>;
}

struct InferenceRequest {
    data: CompletionRequest,
}

impl Message for InferenceRequest {
    type Result = Result<String, String>;
}

impl Handler<InferenceRequest> for InferenceActor {
    type Result = Result<String, String>;

    fn handle(&mut self, msg: InferenceRequest, _: &mut Context<Self>) -> Self::Result {
        let prompt_tokens = msg.data.prompt_tokens;
        let completion_max_len = msg.data.completion_max_len;

        let mut session = self.session.lock().unwrap();
        let model = self.model.lock().unwrap();

        let mut output_buffer = String::new();

        let _ = session.infer::<Infallible>(
            model.as_ref(),
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(&prompt_tokens),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: Some(completion_max_len.parse::<usize>().unwrap_or(500)),
            },
            &mut Default::default(),
            |r| match r {
                llm::InferenceResponse::InferredToken(t) => {
                    output_buffer.push_str(&t);

                    Ok(llm::InferenceFeedback::Continue)
                }

                _ => Ok(llm::InferenceFeedback::Continue),
            },
        );

        Ok(output_buffer)
    }
}

async fn index() -> impl Responder {
    HttpResponse::Ok().body("pong")
}

async fn completion(
    actor: web::Data<Addr<InferenceActor>>,
    json_data: Json<CompletionRequest>,
) -> impl Responder {
    let result = actor
        .send(InferenceRequest {
            data: json_data.into_inner(),
        })
        .await;

    match result {
        Ok(Ok(result)) => HttpResponse::Ok().body(result),
        Ok(Err(e)) => HttpResponse::InternalServerError().body(e.to_string()),
        Err(_) => HttpResponse::InternalServerError().body("Actor communication error"),
    }
}

pub fn run_server(
    session: Arc<Mutex<InferenceSession>>,
    model: Arc<Mutex<Box<dyn Model>>>,
) -> std::io::Result<()> {
    rt::System::new().block_on(
        HttpServer::new(move || {
            let addr = InferenceActor {
                session: Arc::clone(&session),
                model: Arc::clone(&model),
            }
            .start();

            App::new()
                .app_data(Data::new(addr))
                .service(web::resource("/").route(web::get().to(index)))
                .service(web::resource("/completion").route(web::post().to(completion)))
        })
        .bind(("127.0.0.1", 8080))?
        .run(),
    )
}
