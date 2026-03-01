//! Speech synthesis — text-to-speech via Piper (ONNX, offline, CPU).
//!
//! Converts text to OGG/Opus audio suitable for Telegram voice messages.
//! Uses Piper TTS as a subprocess (text → WAV) then ffmpeg (WAV → OGG/Opus).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use crate::config::SynthesisConfig;

/// Maximum text length accepted for synthesis (characters).
const MAX_TEXT_CHARS: usize = 4000;

/// Synthesize text to an OGG/Opus audio file suitable for Telegram voice messages.
///
/// Returns the path to the generated `.ogg` file.  The caller is responsible
/// for sending the file and cleaning it up afterwards.
pub async fn synthesize_speech(
    text: &str,
    config: &SynthesisConfig,
    output_dir: &Path,
) -> Result<PathBuf> {
    if text.is_empty() {
        bail!("Cannot synthesize empty text");
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        bail!(
            "Text too long for synthesis ({} chars, max {MAX_TEXT_CHARS})",
            text.chars().count()
        );
    }

    match config.backend.as_str() {
        "piper" => synthesize_piper(text, config, output_dir).await,
        "edge-tts" => synthesize_edge_tts(text, config, output_dir).await,
        other => bail!("Unknown synthesis backend: {other}"),
    }
}

/// Piper TTS backend: pipes text to the `piper` CLI, produces WAV, converts to OGG/Opus.
async fn synthesize_piper(
    text: &str,
    config: &SynthesisConfig,
    output_dir: &Path,
) -> Result<PathBuf> {
    let piper_bin = &config.piper_path;
    let model_path = &config.model_path;

    if model_path.is_empty() {
        bail!("synthesis.model_path is required for the piper backend");
    }

    tokio::fs::create_dir_all(output_dir).await.ok();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let wav_path = output_dir.join(format!("tts_{ts}.wav"));
    let ogg_path = output_dir.join(format!("tts_{ts}.ogg"));

    // Piper: text on stdin → WAV file
    let mut piper = tokio::process::Command::new(piper_bin)
        .args([
            "--model",
            model_path,
            "--output_file",
            wav_path.to_str().unwrap_or("tts.wav"),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("Failed to spawn piper — is it installed?")?;

    if let Some(mut stdin) = piper.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .context("Failed to write text to piper stdin")?;
        // Drop stdin to signal EOF
    }

    let output = piper
        .wait_with_output()
        .await
        .context("Failed to wait for piper")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Piper TTS failed ({}): {stderr}", output.status);
    }

    // Convert WAV → OGG/Opus via ffmpeg
    wav_to_ogg(&wav_path, &ogg_path).await?;

    // Clean up WAV
    let _ = tokio::fs::remove_file(&wav_path).await;

    Ok(ogg_path)
}

/// Edge-TTS backend: calls the `edge-tts` CLI (requires Python + edge-tts package).
async fn synthesize_edge_tts(
    text: &str,
    config: &SynthesisConfig,
    output_dir: &Path,
) -> Result<PathBuf> {
    let voice = if config.voice.is_empty() {
        "fr-FR-DeniseNeural"
    } else {
        &config.voice
    };

    tokio::fs::create_dir_all(output_dir).await.ok();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mp3_path = output_dir.join(format!("tts_{ts}.mp3"));
    let ogg_path = output_dir.join(format!("tts_{ts}.ogg"));

    let output = tokio::process::Command::new("edge-tts")
        .args([
            "--voice",
            voice,
            "--text",
            text,
            "--write-media",
            mp3_path.to_str().unwrap_or("tts.mp3"),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("Failed to run edge-tts — is it installed? (pip install edge-tts)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("edge-tts failed ({}): {stderr}", output.status);
    }

    // Convert MP3 → OGG/Opus via ffmpeg
    mp3_to_ogg(&mp3_path, &ogg_path).await?;

    // Clean up MP3
    let _ = tokio::fs::remove_file(&mp3_path).await;

    Ok(ogg_path)
}

/// Convert WAV to OGG/Opus using ffmpeg.
async fn wav_to_ogg(wav_path: &Path, ogg_path: &Path) -> Result<()> {
    audio_to_ogg(wav_path, ogg_path).await
}

/// Convert MP3 to OGG/Opus using ffmpeg.
async fn mp3_to_ogg(mp3_path: &Path, ogg_path: &Path) -> Result<()> {
    audio_to_ogg(mp3_path, ogg_path).await
}

/// Resolve ffmpeg binary — check common user-local paths first, then PATH.
fn resolve_ffmpeg() -> &'static str {
    static FFMPEG: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    FFMPEG.get_or_init(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        let local = format!("{home}/.local/bin/ffmpeg");
        if std::path::Path::new(&local).exists() {
            local
        } else {
            "ffmpeg".to_string()
        }
    })
}

/// Generic audio → OGG/Opus conversion via ffmpeg.
async fn audio_to_ogg(input: &Path, output: &Path) -> Result<()> {
    let result = tokio::process::Command::new(resolve_ffmpeg())
        .args([
            "-y",
            "-i",
            input.to_str().unwrap_or("input"),
            "-c:a",
            "libopus",
            "-b:a",
            "48k",
            "-ar",
            "48000",
            "-ac",
            "1",
            output.to_str().unwrap_or("output.ogg"),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("Failed to run ffmpeg — is it installed?")?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        bail!("ffmpeg conversion failed ({}): {stderr}", result.status);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_text_length_constant() {
        assert_eq!(MAX_TEXT_CHARS, 4000);
    }

    #[tokio::test]
    async fn rejects_empty_text() {
        let config = SynthesisConfig::default();
        let err = synthesize_speech("", &config, Path::new("/tmp"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_oversized_text() {
        let long_text: String = "a".repeat(MAX_TEXT_CHARS + 1);
        let config = SynthesisConfig::default();
        let err = synthesize_speech(&long_text, &config, Path::new("/tmp"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("too long"), "got: {err}");
    }

    #[tokio::test]
    async fn rejects_unknown_backend() {
        let mut config = SynthesisConfig::default();
        config.backend = "unknown".into();
        let err = synthesize_speech("hello", &config, Path::new("/tmp"))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Unknown synthesis backend"),
            "got: {err}"
        );
    }
}
