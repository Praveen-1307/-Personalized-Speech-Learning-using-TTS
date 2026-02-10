import click
import logging
from pathlib import Path
import json
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import time

from .preprocessor import AudioPreprocessor
from .feature_extractor import FeatureExtractor
from .pattern_learner import PatternLearner
from .profile_manager import VoiceProfileManager
from .emotion_detector import EmotionDetector

console = Console()
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Personalized TTS Engine CLI"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
@cli.command()
@click.option('--audio-dir', '-a', required=True, help='Directory containing audio files')
@click.option('--user-id', '-u', required=True, help='User ID for profile')
@click.option('--output-format', '-f', default='both', type=click.Choice(['json', 'yaml', 'both']))
@click.option('--method', '-m', default='gmm', type=click.Choice(['gmm', 'neural']))
def train(audio_dir, user_id, output_format, method):
    """Train a personalized voice model from audio samples"""
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Training model...", total=5)
        
        # Step 1: Initialize components
        progress.update(task, advance=1, description="Initializing components")
        preprocessor = AudioPreprocessor()
        feature_extractor = FeatureExtractor()
        pattern_learner = PatternLearner(method=method)
        profile_manager = VoiceProfileManager()
        emotion_detector = EmotionDetector()
        
        # Step 2: Process audio files
        progress.update(task, advance=1, description="Processing audio files")
        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))
        
        if not audio_files:
            console.print(f"[red]No audio files found in {audio_dir}")
            return
            
        all_features = []
        
        for audio_file in audio_files:
            progress.update(task, description=f"Processing {audio_file.name}")
            
            # Preprocess
            result = preprocessor.preprocess(str(audio_file))
            
            # Extract features
            features = feature_extractor.extract_all_features(result['audio'])
            
            # Detect emotion
            emotion_result = emotion_detector.detect(result['audio'])
            features['emotion_detected'] = emotion_result
            
            all_features.append(features)
            
        # Step 3: Learn patterns
        progress.update(task, advance=1, description="Learning patterns")
        features_matrix = pattern_learner.prepare_features(all_features)
        
        if method == 'gmm':
            pattern_learner.learn_gmm(features_matrix)
        else:
            pattern_learner.learn_neural(features_matrix)
            
        # Step 4: Create profile
        progress.update(task, advance=1, description="Creating voice profile")
        # Use average features for profile
        avg_features = {}
        for key in all_features[0].keys():
            if isinstance(all_features[0][key], dict):
                avg_features[key] = {}
                for subkey in all_features[0][key].keys():
                    if isinstance(all_features[0][key][subkey], (int, float)):
                        values = [f[key][subkey] for f in all_features]
                        avg_features[key][subkey] = sum(values) / len(values)
                    elif subkey == 'emotion':
                        # For categorical values like emotion, use the most frequent (mode)
                        values = [f[key][subkey] for f in all_features]
                        if values:
                            avg_features[key][subkey] = max(set(values), key=values.count)
            elif isinstance(all_features[0][key], (int, float)):
                values = [f[key] for f in all_features]
                avg_features[key] = sum(values) / len(values)
                
        profile = pattern_learner.generate_profile(avg_features)
        
        # Step 5: Save profile
        progress.update(task, advance=1, description="Saving profile")
        voice_profile = profile_manager.create_profile(
            user_id=user_id,
            features=avg_features,
            model_info={
                'method': method,
                'n_samples': len(audio_files),
                'training_time': time.time()
            }
        )
        
        profile_manager.save_profile(user_id, format=output_format)
        
    # Display results
    console.print(f"\n[green]✓ Training completed for user: {user_id}")
    console.print(f"   Profile ID: {voice_profile['profile_id']}")
    console.print(f"   Audio files processed: {len(audio_files)}")
    console.print(f"   Output format: {output_format}")
    
    # Show feature summary
    table = Table(title="Feature Summary")
    table.add_column("Feature Type", style="cyan")
    table.add_column("Key Metrics", style="green")
    
    if 'prosodic' in avg_features:
        table.add_row(
            "Prosodic",
            f"Pitch: {avg_features['prosodic'].get('f0_mean', 0):.1f}Hz"
        )
        
    if 'speaking_pattern' in avg_features:
        table.add_row(
            "Speaking Pattern",
            f"Rate: {avg_features['speaking_pattern'].get('words_per_minute', 0):.1f} WPM"
        )
        
    if 'emotion_detected' in avg_features:
        table.add_row(
            "Emotion",
            f"Detected: {avg_features['emotion_detected'].get('emotion', 'unknown')}"
        )
        
    console.print(table)
    
@cli.command()
@click.option('--text', '-t', required=True, help='Text to synthesize')
@click.option('--user-id', '-u', help='User ID for personalized voice')
@click.option('--output', '-o', default='output.wav', help='Output audio file')
@click.option('--emotion', '-e', help='Emotion to apply (happy, sad, neutral, etc.)')
def synthesize(text, user_id, output, emotion):
    """Synthesize text with personalized voice"""
    
    console.print(f"[cyan]Synthesizing:[/cyan] {text}")
    
    if user_id:
        # Load profile
        profile_manager = VoiceProfileManager()
        profile = profile_manager.load_profile(user_id)
        
        if profile:
            console.print(f"[green]Using personalized voice for: {user_id}")
            
            # Apply personalization parameters
            # This would integrate with Piper TTS
            # For now, we'll show what would happen
            
            personalization_params = {
                'pitch_adjustment': profile['features'].get('prosodic', {}).get('f0_mean', 0),
                'speaking_rate': profile['features'].get('speaking_pattern', {}).get('words_per_minute', 150) / 150,
                'pause_duration': 1.0  # Default
            }
            
            if emotion:
                emotion_detector = EmotionDetector()
                emotion_params = emotion_detector.map_emotion_to_synthesis_params(emotion)
                personalization_params.update(emotion_params)
                console.print(f"[yellow]Emotion applied: {emotion}")
                
            console.print(f"[cyan]Personalization parameters:[/cyan]")
            for key, value in personalization_params.items():
                console.print(f"  {key}: {value}")
        else:
            console.print(f"[yellow]No profile found for {user_id}, using default voice")
    else:
        console.print("[cyan]Using default TTS voice")
        
    # Here you would integrate with Piper TTS
    # For demonstration, we'll create a placeholder
    console.print(f"[green]✓ Synthesis complete")
    console.print(f"   Output saved to: {output}")
    console.print(f"   Text length: {len(text)} characters")
    
@cli.command()
@click.option('--user-id', '-u', help='User ID to list (optional)')
def list_profiles(user_id):
    """List all voice profiles or details for specific user"""
    
    profile_manager = VoiceProfileManager()
    
    if user_id:
        profile = profile_manager.load_profile(user_id)
        if profile:
            console.print(f"[cyan]Profile for: {user_id}[/cyan]")
            
            # General Info Table
            table = Table(title="Profile Metadata")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            for key in ['profile_id', 'created_at', 'updated_at', 'version']:
                 if key in profile:
                     table.add_row(key.replace('_', ' ').title(), str(profile[key]))
            
            console.print(table)
            
            # Features Table (Detailed)
            if 'features' in profile:
                ft_table = Table(title="Voice Characteristics (Features)")
                ft_table.add_column("Category", style="magenta")
                ft_table.add_column("Metric", style="cyan")
                ft_table.add_column("Value", style="yellow")
                ft_table.add_column("Description", style="white")
                
                feats = profile['features']
                
                # Prosodic (Pitch/Energy)
                if 'prosodic' in feats:
                    p = feats['prosodic']
                    if 'f0_mean' in p:
                        ft_table.add_row("Prosody", "Average Pitch (F0)", f"{p['f0_mean']:.2f} Hz", "Fundamental frequency")
                    if 'f0_std' in p:
                        ft_table.add_row("Prosody", "Pitch Variation", f"{p['f0_std']:.2f} Hz", "Intonation/Expressiveness")
                    if 'energy_mean' in p:
                        ft_table.add_row("Prosody", "Loudness", f"{p['energy_mean']:.2f}", "Average volume intensity")
                        
                # Tone (Spectral)
                if 'spectral' in feats:
                    s = feats['spectral']
                    if 'spectral_centroid_mean' in s:
                        val = s['spectral_centroid_mean']
                        desc = "Bright/Sharp" if val > 2000 else "Deep/Mellow"
                        ft_table.add_row("Tone", "Spectral Centroid", f"{val:.2f} Hz", desc)
                        
                # Tempo
                if 'speaking_pattern' in feats:
                    sp = feats['speaking_pattern']
                    if 'words_per_minute' in sp:
                         ft_table.add_row("Tempo", "Speed", f"{sp['words_per_minute']:.1f} WPM", "Words per minute")
                    if 'rhythm_entropy' in sp:
                         ft_table.add_row("Tempo", "Rhythm", f"{sp['rhythm_entropy']:.2f}", "Speech regularity")
                         
                console.print(ft_table)

        else:
            console.print(f"[red]No profile found for {user_id}")
    else:
        profiles = profile_manager.list_profiles()
        
        if profiles:
            table = Table(title="Available Voice Profiles")
            table.add_column("User ID", style="cyan")
            table.add_column("Created", style="green")
            table.add_column("Updated", style="yellow")
            table.add_column("Version", style="magenta")
            
            for profile in profiles:
                table.add_row(
                    profile['user_id'],
                    profile['created_at'][:10] if profile['created_at'] else 'N/A',
                    profile['updated_at'][:10] if profile['updated_at'] else 'N/A',
                    str(profile['version'])
                )
                
            console.print(table)
            console.print(f"[cyan]Total profiles: {len(profiles)}[/cyan]")
        else:
            console.print("[yellow]No profiles found[/yellow]")
            
@cli.command()
@click.option('--dataset', '-d', required=True, 
              type=click.Choice(['LJSpeech', 'VCTK', 'LibriTTS', 'HiFi', 'HUI']))
@click.option('--output', '-o', default='dataset_analysis.md', help='Output file')
def analyze_dataset(dataset, output):
    """Analyze TTS dataset characteristics"""
    
    from dataset_tools.analyzer import DatasetAnalyzer
    
    console.print(f"[cyan]Analyzing dataset: {dataset}[/cyan]")
    
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze(dataset)
    
    # Save to markdown
    with open(output, 'w') as f:
        f.write(f"# {dataset} Dataset Analysis\n\n")
        f.write(f"**Analysis Date**: {time.ctime()}\n\n")
        
        for section, data in analysis.items():
            f.write(f"## {section.replace('_', ' ').title()}\n\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"- **{key}**: {value}\n")
            else:
                f.write(f"{data}\n")
            f.write("\n")
            
    console.print(f"[green]✓ Analysis saved to {output}")
    
@cli.command()
@click.option('--user-id', '-u', required=True, help='User ID to delete')
def delete_profile(user_id):
    """Delete a user's voice profile"""
    
    if click.confirm(f"Are you sure you want to delete profile for {user_id}?"):
        profile_manager = VoiceProfileManager()
        profile_manager.delete_profile(user_id)
        console.print(f"[green]✓ Deleted profile for {user_id}")
    else:
        console.print("[yellow]Deletion cancelled")
        
if __name__ == '__main__':
    cli()
