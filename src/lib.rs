use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

trait GeminiAPI {
    fn get_api_key(&self) -> String;
    fn get_endpoint(&self) -> String;
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    text: String,
    // inline_data
    // function_call
    // function_response
    // file_data
    // executable_code
    // code_execution_result
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Content {
    parts: Vec<Part>,
    role: String,
}

#[derive(Error, Debug)]
pub enum GeminiError {
    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("No response candidates available")]
    NoCandidates,
}

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/";
const DEFAULT_MODEL: &str = "gemini-1.5-flash";

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    content: Content,
    // finish_reason
    // safety_ratings
    // citation_metadata
    //token_count: u32,
    // grounding_attributions
    // groundint_metadata
    // avg_logprops
    // logprobs_result
    //index: u32,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    model: String,
    contents: Vec<Content>,
    // tools
    // tool_config
    // safety_settings
    system_instruction: Option<Content>,
    // generation_config
    // cached_content
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    candidates: Vec<Candidate>,
    // prompt_feedback
    // usage_metadata
    model_version: String,
}

impl GenerateContentResponse {
    pub fn get_text(&self) -> Result<String, GeminiError> {
        self.candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .ok_or(GeminiError::NoCandidates)
    }
}

impl Content {
    fn new(text: String, role: &str) -> Self {
        Self {
            parts: vec![Part { text }],
            role: role.to_string(),
        }
    }
}

#[derive(Debug)]
pub struct Conversation {
    api_key: String,
    base_url: String,
    model: String,
    history: Vec<Content>,
    system_instruction: Option<Content>,
    client: Client,
}

impl Conversation {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            history: Vec::new(),
            system_instruction: None,
            client: Client::new(),
        }
    }

    pub fn set_system_instruction_text(&mut self, text: String) {
        self.system_instruction = Some(Content::new(text, "system"));
    }

    pub async fn talk(&mut self, text: String) -> Result<GenerateContentResponse, GeminiError> {
        self.history.push(Content::new(text, "user"));

        let request = GenerateContentRequest {
            model: self.model.clone(),
            contents: self.history.clone(),
            system_instruction: self.system_instruction.clone(),
        };

        let response = self
            .client
            .post(self.get_endpoint())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .json(&request)
            .send()
            .await?
            .json::<GenerateContentResponse>()
            .await?;

        let response_text = response.get_text()?;
        self.history.push(Content::new(response_text, "model"));
        Ok(response)
    }

    pub async fn summarize(&mut self) -> Result<(), GeminiError> {
        let mut conversation = Self::new(self.api_key.clone());
        conversation.history = self.history.clone();

        let system_instruction = "
            Previous conversations need to be compressed to save the tokens needed to process the generative AI.
            You are the AI assistant that summarises the conversation for this purpose."
            .to_string();
        conversation.set_system_instruction_text(system_instruction);

        let summary = conversation
            .talk("Summarize the conversation".to_string())
            .await?
            .get_text()?;

        self.history.clear();
        self.history.push(Content::new(summary, "model"));

        Ok(())
    }
}

impl GeminiAPI for Conversation {
    fn get_api_key(&self) -> String {
        self.api_key.clone()
    }

    fn get_endpoint(&self) -> String {
        format!(
            "{}models/{}:generateContent?key={}",
            self.base_url,
            self.model,
            self.get_api_key()
        )
    }
}
