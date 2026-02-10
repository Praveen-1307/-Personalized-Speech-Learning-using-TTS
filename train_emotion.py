
import os
import sys
import requests
import zipfile
import io
import shutil
from rich.console import Console

console = Console()

def download_emodb():
    url = "http://emodb.bilderbar.info/download/download.zip"
    target_dir = "data/emodb"
    
    if os.path.exists(target_dir):
        console.print(f"[yellow]EMO-DB already exists in {target_dir}. Skipping download.[/yellow]")
        return target_dir
        
    os.makedirs(target_dir, exist_ok=True)
    console.print(f"[cyan]Downloading EMO-DB from {url}...[/cyan]")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            console.print("[green]Download complete. Extracting...[/green]")
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(target_dir)
            console.print(f"[green]Extracted to {target_dir}[/green]")
            return target_dir
        else:
             console.print(f"[red]Failed to download. Status code: {response.status_code}[/red]")
    except Exception as e:
        console.print(f"[red]Error downloading EMO-DB: {e}[/red]")
        
    return None

def train_emotion_model(data_dir):
    console.print("\n[bold green]Training Emotion Detector...[/bold green]")
    
    # We need to call the internal components
    from personalization_engine.emotion_detector import EmotionDetector
    import glob
    import numpy as np
    import librosa
    
    # EMO-DB has a 'wav' folder inside
    wav_dir = os.path.join(data_dir, "wav")
    if not os.path.exists(wav_dir):
         # Sometimes structure is flat
         wav_dir = data_dir
         
    files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    if not files:
        console.print(f"[red]No wav files found in {wav_dir}[/red]")
        return

    detector = EmotionDetector(model_type='svm')
    
    X = []
    y = []
    
    # EMO-DB encoding in filename (5th character):
    # W=Ärger (Anger), L=Langeweile (Boredom/Calm), E=Ekel (Disgust), 
    # A=Angst (Fear), F=Freude (Happiness), T=Trauer (Sadness), N=Neutral
    emo_map = {
        'W': 'angry',
        'L': 'calm',
        'E': 'disgust',
        'A': 'fear',
        'F': 'happy',
        'T': 'sad',
        'N': 'neutral'
    }
    
    console.print(f"[yellow]Processing {len(files)} files for features...[/yellow]")
    
    processed = 0
    for f in files:
        filename = os.path.basename(f)
        code = filename[5] # 6th char is emotion
        
        if code in emo_map:
            emotion = emo_map[code]
            try:
                # Load audio
                audio, sr = librosa.load(f, sr=22050)
                
                # Extract features
                feats = detector.extract_emotion_features(audio, sr)
                
                X.append(feats)
                y.append(emotion)
                processed += 1
                
                if processed % 50 == 0:
                    print(f"Processed {processed}...")
            except Exception as e:
                pass
                
    if len(X) > 0:
        console.print(f"[green]Training on {len(X)} samples...[/green]")
        detector.train(np.array(X), np.array(y))
        
        # Save model
        os.makedirs("models/emotion", exist_ok=True)
        detector.save_model("models/emotion/svm_model.pkl")
        console.print("[bold green]✓ Emotion model saved to models/emotion/svm_model.pkl[/bold green]")
    else:
        console.print("[red]No valid samples processed for training[/red]")

if __name__ == "__main__":
    data_path = download_emodb()
    if data_path:
        train_emotion_model(data_path)
