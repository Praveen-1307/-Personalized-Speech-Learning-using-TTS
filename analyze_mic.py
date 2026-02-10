
import click
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from personalization_engine.feature_extractor import FeatureExtractor
from personalization_engine.emotion_detector import EmotionDetector

console = Console()

@click.command()
@click.option('--duration', '-d', default=5, help='Duration of recording in seconds')
@click.option('--sample-rate', '-r', default=22050, help='Sample rate')
def record_analyze(duration, sample_rate):
    """Record microphone input and analyze voice immediately."""
    
    console.print(Panel.fit(f"[bold cyan]Recording for {duration} seconds...[/bold cyan]\n[yellow]Speak naturally![/yellow]"))
    
    # 1. Record Audio
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        # Show specific progress bar
        with Live(Text("Recording...", style="red blink"), refresh_per_second=4) as live:
            for i in range(duration):
                time.sleep(1)
                live.update(Text(f"Recording... {duration - i - 1}s remaining", style="red"))
                
        sd.wait()
        console.print("[green]Recording complete![/green]")
        
        # Flatten to 1D array
        audio_data = recording.flatten()
        
    except Exception as e:
        console.print(f"[bold red]Microphone Error:[/bold red] {e}")
        console.print("[yellow]Make sure you have a microphone connected and 'sounddevice' installed.[/yellow]")
        console.print("Try: pip install sounddevice taskport")
        return

    # 2.5 Check for silence
    max_amp = np.max(np.abs(audio_data))
    if max_amp < 0.01:
        console.print("[bold red]Warning: Audio appears to be silent![/bold red]")
        console.print(f"Max Amplitude: {max_amp:.4f} (Threshold: 0.01)")
        console.print("Check your microphone settings.")
        return

    # 3. Analyze
    console.print("\n[bold]Analyzing Voice Features...[/bold]")
    
    extractor = FeatureExtractor()
    emotion_detector = EmotionDetector()
    
    with console.status("Extracting features..."):
        # Features
        features = extractor.extract_all_features(audio_data)
        
        # Emotion
        emotion_result = emotion_detector.detect(audio_data)
    
    # 4. Display Results
    _display_live_results(features, emotion_result)

def _display_live_results(features, emotion_result):
    """Show interactive dashboard of results."""
    
    # Extract Key Metrics
    f0 = features.get('prosodic', {}).get('f0_mean', 0)
    # Handle NaN if present
    if np.isnan(f0): f0 = 0.0
        
    wpm = features.get('speaking_pattern', {}).get('words_per_minute', 0)
    emotion = emotion_result.get('emotion', 'unknown')
    confidence = emotion_result.get('confidence', 0)
    
    # Determine Tone Description
    tone = "Neutral"
    if f0 > 220: tone = "High / Thin"
    elif f0 < 120: tone = "Deep / Bass"
    elif f0 > 0: tone = "Mid-Range"
    
    # Construct message safely
    lines = [
        "[bold underline]Analysis Result[/bold underline]",
        "",
        f"🎤 [cyan]Pitch (F0):[/cyan]      {f0:.2f} Hz  ({tone})",
        f"🏃 [cyan]Speed:[/cyan]           {wpm:.1f} WPM",
        f"🎭 [cyan]Emotion:[/cyan]         {emotion.title()} ({confidence:.0%})"
    ]
    message = "\n".join(lines)
    
    console.print(Panel(message, title="Voice DNA", border_style="green", expand=False))

if __name__ == '__main__':
    record_analyze()
