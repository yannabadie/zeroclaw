//! OneDrive tool — access personal OneDrive files via Microsoft Graph API.
//!
//! Authentication uses the OAuth2 Device Code Flow for personal Microsoft accounts.
//! Tokens are cached and refreshed automatically.

use super::traits::{Tool, ToolResult};
use crate::config::OneDriveConfig;
use crate::security::SecurityPolicy;
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Microsoft login endpoint for personal (consumer) accounts.
const AUTH_BASE: &str = "https://login.microsoftonline.com/consumers/oauth2/v2.0";
/// Microsoft Graph API base URL.
const GRAPH_BASE: &str = "https://graph.microsoft.com/v1.0";
/// Scopes required for OneDrive read + write access.
const SCOPES: &str = "Files.Read Files.ReadWrite offline_access";

/// Cached OAuth2 token on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenCache {
    access_token: String,
    refresh_token: String,
    /// Unix timestamp when the access token expires.
    expires_at: u64,
}

impl TokenCache {
    fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Consider expired 60s before actual expiry for safety margin.
        now >= self.expires_at.saturating_sub(60)
    }
}

/// OneDrive tool for reading/writing files via Microsoft Graph API.
pub struct OneDriveTool {
    #[allow(dead_code)]
    security: Arc<SecurityPolicy>,
    config: OneDriveConfig,
    token: Mutex<Option<TokenCache>>,
}

impl OneDriveTool {
    pub fn new(security: Arc<SecurityPolicy>, config: OneDriveConfig) -> Self {
        Self {
            security,
            config,
            token: Mutex::new(None),
        }
    }

    /// Resolve the token cache path, expanding ~ to $HOME.
    fn token_path(&self) -> PathBuf {
        let expanded = shellexpand::tilde(&self.config.token_path).into_owned();
        PathBuf::from(expanded)
    }

    /// Load token from disk cache.
    async fn load_cached_token(&self) -> Option<TokenCache> {
        let path = self.token_path();
        let data = tokio::fs::read_to_string(&path).await.ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Save token to disk cache.
    async fn save_token(&self, token: &TokenCache) -> Result<()> {
        let path = self.token_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let json = serde_json::to_string_pretty(token)?;
        tokio::fs::write(&path, json).await?;
        Ok(())
    }

    /// Get a valid access token, refreshing if needed.
    async fn get_access_token(&self) -> Result<String> {
        let mut guard = self.token.lock().await;

        // Try memory cache first.
        if let Some(ref cached) = *guard {
            if !cached.is_expired() {
                return Ok(cached.access_token.clone());
            }
        }

        // Try disk cache.
        if let Some(cached) = self.load_cached_token().await {
            if !cached.is_expired() {
                let token = cached.access_token.clone();
                *guard = Some(cached);
                return Ok(token);
            }
            // Token expired — try refresh.
            if !cached.refresh_token.is_empty() {
                match self.refresh_token(&cached.refresh_token).await {
                    Ok(new_token) => {
                        let access = new_token.access_token.clone();
                        self.save_token(&new_token).await.ok();
                        *guard = Some(new_token);
                        return Ok(access);
                    }
                    Err(e) => {
                        tracing::warn!("OneDrive token refresh failed: {e}");
                    }
                }
            }
        }

        bail!(
            "OneDrive authentication required. Run the device code flow first.\n\
             Use action \"auth\" to start the authentication process."
        );
    }

    /// Refresh an expired access token using the refresh token.
    async fn refresh_token(&self, refresh_token: &str) -> Result<TokenCache> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{AUTH_BASE}/token"))
            .form(&[
                ("client_id", self.config.client_id.as_str()),
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("scope", SCOPES),
            ])
            .send()
            .await
            .context("Failed to refresh OneDrive token")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("Token refresh failed: {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

        Ok(TokenCache {
            access_token: body["access_token"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            refresh_token: body["refresh_token"]
                .as_str()
                .unwrap_or(refresh_token)
                .to_string(),
            expires_at: now + expires_in,
        })
    }

    /// Start the Device Code Flow authentication.
    async fn start_device_code_flow(&self) -> Result<String> {
        if self.config.client_id.is_empty() {
            bail!(
                "OneDrive client_id is not configured. Register an Azure app at \
                 https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps and \
                 set [onedrive].client_id in config.toml"
            );
        }

        let client = reqwest::Client::new();

        // Step 1: Request device code.
        let resp = client
            .post(format!("{AUTH_BASE}/devicecode"))
            .form(&[
                ("client_id", self.config.client_id.as_str()),
                ("scope", SCOPES),
            ])
            .send()
            .await
            .context("Failed to request device code")?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            bail!("Device code request failed: {body}");
        }

        let dc: serde_json::Value = resp.json().await?;
        let device_code = dc["device_code"]
            .as_str()
            .context("Missing device_code in response")?;
        let user_code = dc["user_code"]
            .as_str()
            .context("Missing user_code in response")?;
        let verification_uri = dc["verification_uri"]
            .as_str()
            .unwrap_or("https://microsoft.com/devicelogin");
        let interval = dc["interval"].as_u64().unwrap_or(5);
        let expires_in = dc["expires_in"].as_u64().unwrap_or(900);

        let instructions = format!(
            "OneDrive Authentication Required:\n\
             1. Open: {verification_uri}\n\
             2. Enter code: {user_code}\n\
             3. Sign in with your Microsoft account\n\
             \n\
             Waiting for authorization (expires in {expires_in}s)..."
        );

        // Step 2: Poll for token.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(expires_in);
        let mut poll_interval = std::time::Duration::from_secs(interval);

        loop {
            if std::time::Instant::now() >= deadline {
                bail!("Device code authorization timed out");
            }

            tokio::time::sleep(poll_interval).await;

            let poll_resp = client
                .post(format!("{AUTH_BASE}/token"))
                .form(&[
                    ("client_id", self.config.client_id.as_str()),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", device_code),
                ])
                .send()
                .await;

            let poll_resp = match poll_resp {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!("Device code poll error: {e}");
                    continue;
                }
            };

            let body: serde_json::Value = poll_resp.json().await.unwrap_or_default();

            if let Some(access_token) = body["access_token"].as_str() {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let expires_in_secs = body["expires_in"].as_u64().unwrap_or(3600);

                let token = TokenCache {
                    access_token: access_token.to_string(),
                    refresh_token: body["refresh_token"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    expires_at: now + expires_in_secs,
                };

                self.save_token(&token).await?;
                *self.token.lock().await = Some(token);

                return Ok(format!(
                    "{instructions}\n\nAuthentication successful! OneDrive is now connected."
                ));
            }

            let error = body["error"].as_str().unwrap_or("unknown");
            match error {
                "authorization_pending" => continue,
                "slow_down" => {
                    poll_interval += std::time::Duration::from_secs(5);
                    continue;
                }
                "authorization_declined" => bail!("User declined the authorization"),
                "expired_token" => bail!("Device code expired before authorization"),
                other => bail!("Device code flow error: {other}"),
            }
        }
    }

    /// List files in a OneDrive folder.
    async fn list_files(&self, path: &str) -> Result<String> {
        let token = self.get_access_token().await?;
        let client = reqwest::Client::new();

        let url = if path.is_empty() || path == "/" {
            format!("{GRAPH_BASE}/me/drive/root/children")
        } else {
            let clean_path = path.trim_start_matches('/');
            format!("{GRAPH_BASE}/me/drive/root:/{clean_path}:/children")
        };

        let resp = client
            .get(&url)
            .bearer_auth(&token)
            .query(&[("$select", "name,size,lastModifiedDateTime,folder,file")])
            .send()
            .await
            .context("Failed to list OneDrive files")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("OneDrive list failed ({status}): {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let items = body["value"].as_array().cloned().unwrap_or_default();

        let mut output = format!("Files in /{path}:\n\n");
        for item in &items {
            let name = item["name"].as_str().unwrap_or("?");
            let is_folder = item["folder"].is_object();
            let size = item["size"].as_u64().unwrap_or(0);
            let modified = item["lastModifiedDateTime"].as_str().unwrap_or("?");

            if is_folder {
                output.push_str(&format!("  [DIR]  {name}/  ({modified})\n"));
            } else {
                let size_display = format_size(size);
                output.push_str(&format!("  [FILE] {name}  {size_display}  ({modified})\n"));
            }
        }
        output.push_str(&format!("\n{} item(s)", items.len()));

        Ok(output)
    }

    /// Read file content from OneDrive.
    async fn read_file(&self, path: &str) -> Result<String> {
        let token = self.get_access_token().await?;
        let client = reqwest::Client::new();

        let clean_path = path.trim_start_matches('/');
        let url = format!("{GRAPH_BASE}/me/drive/root:/{clean_path}:/content");

        let resp = client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .context("Failed to read OneDrive file")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("OneDrive read failed ({status}): {body}");
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        // For text-like files, return content directly.
        if is_text_content(&content_type) || is_text_extension(clean_path) {
            let text = resp.text().await?;
            // Truncate very large files.
            if text.len() > 100_000 {
                Ok(format!(
                    "{}\n\n[Truncated — showing first 100KB of {} bytes]",
                    &text[..100_000],
                    text.len()
                ))
            } else {
                Ok(text)
            }
        } else {
            // Binary file — return metadata only.
            let size = resp.content_length().unwrap_or(0);
            Ok(format!(
                "Binary file: {clean_path} ({}, {})",
                content_type,
                format_size(size)
            ))
        }
    }

    /// Write content to a file in the write_root folder.
    async fn write_file(&self, path: &str, content: &str) -> Result<String> {
        let token = self.get_access_token().await?;
        let client = reqwest::Client::new();

        // Enforce write restriction to the configured write_root.
        let write_root = self.config.write_root.trim_matches('/');
        let clean_path = path.trim_start_matches('/');

        if !clean_path.starts_with(write_root) {
            bail!(
                "Write access denied. ZeroClaw can only write to /{write_root}/. \
                 Requested path: /{clean_path}"
            );
        }

        let url = format!("{GRAPH_BASE}/me/drive/root:/{clean_path}:/content");

        let resp = client
            .put(&url)
            .bearer_auth(&token)
            .header("Content-Type", "text/plain")
            .body(content.to_string())
            .send()
            .await
            .context("Failed to write OneDrive file")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("OneDrive write failed ({status}): {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let size = body["size"].as_u64().unwrap_or(0);
        Ok(format!("Written /{clean_path} ({})", format_size(size)))
    }

    /// Search for files in OneDrive.
    async fn search_files(&self, query: &str) -> Result<String> {
        let token = self.get_access_token().await?;
        let client = reqwest::Client::new();

        let url = format!("{GRAPH_BASE}/me/drive/root/search(q='{query}')");

        let resp = client
            .get(&url)
            .bearer_auth(&token)
            .query(&[("$select", "name,size,lastModifiedDateTime,parentReference")])
            .query(&[("$top", "20")])
            .send()
            .await
            .context("Failed to search OneDrive")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("OneDrive search failed ({status}): {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let items = body["value"].as_array().cloned().unwrap_or_default();

        let mut output = format!("Search results for \"{query}\":\n\n");
        for item in &items {
            let name = item["name"].as_str().unwrap_or("?");
            let parent = item["parentReference"]["path"]
                .as_str()
                .unwrap_or("")
                .replace("/drive/root:", "");
            let size = item["size"].as_u64().unwrap_or(0);
            output.push_str(&format!("  {parent}/{name}  ({})\n", format_size(size)));
        }
        output.push_str(&format!("\n{} result(s)", items.len()));

        Ok(output)
    }

    /// Get file/folder metadata.
    async fn get_metadata(&self, path: &str) -> Result<String> {
        let token = self.get_access_token().await?;
        let client = reqwest::Client::new();

        let clean_path = path.trim_start_matches('/');
        let url = if clean_path.is_empty() {
            format!("{GRAPH_BASE}/me/drive/root")
        } else {
            format!("{GRAPH_BASE}/me/drive/root:/{clean_path}")
        };

        let resp = client
            .get(&url)
            .bearer_auth(&token)
            .send()
            .await
            .context("Failed to get OneDrive metadata")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("OneDrive metadata failed ({status}): {body}");
        }

        let body: serde_json::Value = resp.json().await?;
        let output = serde_json::to_string_pretty(&body)?;
        Ok(output)
    }
}

#[async_trait]
impl Tool for OneDriveTool {
    fn name(&self) -> &str {
        "onedrive"
    }

    fn description(&self) -> &str {
        "Access personal OneDrive files via Microsoft Graph API. \
         Supports listing, reading, searching, and writing files. \
         Read access to entire drive; write restricted to /ZeroClaw/ folder."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'auth' (start authentication), 'list' (list folder), 'read' (read file), 'write' (write file to /ZeroClaw/), 'search' (search files), 'metadata' (get file/folder info)",
                    "enum": ["auth", "list", "read", "write", "search", "metadata"]
                },
                "path": {
                    "type": "string",
                    "description": "File or folder path in OneDrive (e.g. '/Documents/notes.txt', '/ZeroClaw/output.md'). Defaults to root."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' action)."
                },
                "query": {
                    "type": "string",
                    "description": "Search query (required for 'search' action)."
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        if !self.config.enabled {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(
                    "OneDrive integration is disabled. Set [onedrive].enabled = true in config.toml"
                        .into(),
                ),
            });
        }

        let action = args["action"].as_str().unwrap_or("").to_lowercase();
        let path = args["path"].as_str().unwrap_or("/").to_string();

        let result = match action.as_str() {
            "auth" => self.start_device_code_flow().await,
            "list" => self.list_files(&path).await,
            "read" => self.read_file(&path).await,
            "write" => {
                let content = args["content"].as_str().unwrap_or("");
                if content.is_empty() {
                    Err(anyhow::anyhow!(
                        "'content' parameter is required for write action"
                    ))
                } else {
                    self.write_file(&path, content).await
                }
            }
            "search" => {
                let query = args["query"].as_str().unwrap_or("");
                if query.is_empty() {
                    Err(anyhow::anyhow!(
                        "'query' parameter is required for search action"
                    ))
                } else {
                    self.search_files(query).await
                }
            }
            "metadata" => self.get_metadata(&path).await,
            _ => Err(anyhow::anyhow!(
                "Unknown action '{action}'. Valid: auth, list, read, write, search, metadata"
            )),
        };

        match result {
            Ok(output) => Ok(ToolResult {
                success: true,
                output,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }
}

/// Format byte size to human-readable string.
fn format_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Check if the content type indicates a text-like file.
fn is_text_content(content_type: &str) -> bool {
    content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript")
        || content_type.contains("csv")
        || content_type.contains("yaml")
        || content_type.contains("toml")
}

/// Check if the file extension is text-like.
fn is_text_extension(path: &str) -> bool {
    let text_exts = [
        "txt",
        "md",
        "rs",
        "py",
        "js",
        "ts",
        "json",
        "toml",
        "yaml",
        "yml",
        "xml",
        "html",
        "htm",
        "css",
        "csv",
        "log",
        "sh",
        "bash",
        "zsh",
        "fish",
        "conf",
        "cfg",
        "ini",
        "env",
        "gitignore",
        "dockerfile",
        "makefile",
        "cmake",
        "c",
        "cpp",
        "h",
        "hpp",
        "java",
        "go",
        "rb",
        "php",
        "sql",
        "r",
        "tex",
    ];
    let lower = path.to_lowercase();
    text_exts
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(512), "512 B");
    }

    #[test]
    fn format_size_kb() {
        let result = format_size(2048);
        assert!(result.contains("KB"));
    }

    #[test]
    fn format_size_mb() {
        let result = format_size(5 * 1024 * 1024);
        assert!(result.contains("MB"));
    }

    #[test]
    fn format_size_gb() {
        let result = format_size(2 * 1024 * 1024 * 1024);
        assert!(result.contains("GB"));
    }

    #[test]
    fn is_text_content_types() {
        assert!(is_text_content("text/plain"));
        assert!(is_text_content("application/json"));
        assert!(is_text_content("text/xml"));
        assert!(!is_text_content("application/pdf"));
        assert!(!is_text_content("image/png"));
    }

    #[test]
    fn is_text_extension_known() {
        assert!(is_text_extension("readme.md"));
        assert!(is_text_extension("main.rs"));
        assert!(is_text_extension("data.json"));
        assert!(!is_text_extension("image.png"));
        assert!(!is_text_extension("archive.zip"));
    }

    #[test]
    fn token_cache_expiry() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let fresh = TokenCache {
            access_token: "test".into(),
            refresh_token: "test".into(),
            expires_at: now + 3600,
        };
        assert!(!fresh.is_expired());

        let expired = TokenCache {
            access_token: "test".into(),
            refresh_token: "test".into(),
            expires_at: now - 10,
        };
        assert!(expired.is_expired());
    }

    #[test]
    fn tool_metadata() {
        let config = OneDriveConfig::default();
        let security = Arc::new(SecurityPolicy::default());
        let tool = OneDriveTool::new(security, config);

        assert_eq!(tool.name(), "onedrive");
        assert!(!tool.description().is_empty());
        let schema = tool.parameters_schema();
        assert!(schema["properties"]["action"].is_object());
    }

    #[tokio::test]
    async fn disabled_tool_returns_error() {
        let config = OneDriveConfig {
            enabled: false,
            ..OneDriveConfig::default()
        };
        let security = Arc::new(SecurityPolicy::default());
        let tool = OneDriveTool::new(security, config);

        let result = tool.execute(json!({"action": "list"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("disabled"));
    }

    #[tokio::test]
    async fn unknown_action_returns_error() {
        let config = OneDriveConfig {
            enabled: true,
            client_id: "test".into(),
            ..OneDriveConfig::default()
        };
        let security = Arc::new(SecurityPolicy::default());
        let tool = OneDriveTool::new(security, config);

        let result = tool.execute(json!({"action": "unknown"})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Unknown action"));
    }
}
