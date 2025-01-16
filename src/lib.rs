use reqwest::Client;
use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Candicate {
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
    candidates: Vec<Candicate>,
    // prompt_feedback
    // usage_metadata
    model_version: String,
}

impl GenerateContentResponse {
    pub fn get_text(&self) -> String {
        self.candidates[0].content.parts[0].text.clone()
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
            base_url: "https://generativelanguage.googleapis.com/v1beta/".to_string(),
            model: "gemini-1.5-flash".to_string(),
            history: Vec::new(),
            system_instruction: None,
            client: Client::new(),
        }
    }

    pub fn set_system_instruction_text(&mut self, text: String) {
        let system_instruction = Content {
            parts: vec![Part { text }],
            role: "system".to_string(),
        };
        self.system_instruction = Some(system_instruction);
    }

    pub async fn talk(&mut self, text: String) -> Result<GenerateContentResponse, reqwest::Error> {
        self.history.push(Content {
            parts: vec![Part { text }],
            role: "user".to_string(),
        });

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
            .await?;

        let response = response.json::<GenerateContentResponse>().await?;
        self.history.push(Content {
            parts: vec![Part {
                text: response.get_text(),
            }],
            role: "model".to_string(),
        });
        Ok(response)
    }

    pub async fn summarize(&mut self) -> Result<(), reqwest::Error> {
        let mut conversation = Conversation::new(self.api_key.clone());
        conversation.history = self.history.clone();

        let text = "
        Previous conversations need to be compressed to save the tokens needed to process the generative AI.
        You are the AI assistant that summarises the conversation for this purpose."
        .to_string();
        conversation.set_system_instruction_text(text);

        let summary = conversation
            .talk("Summarize the conversation".to_string())
            .await?
            .get_text();

        self.history.clear();
        self.history.push(Content {
            parts: vec![Part { text: summary }],
            role: "model".to_string(),
        });

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
