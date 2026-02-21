# System Data & Logic Flow

This diagram illustrates the personalized voice cloning pipeline, from raw microphone input to the final synthesized output with interactive analysis.

```mermaid
graph TD
    %% Input Stage
    Input[Microphone Input / .wav] -->|Binary PCM| Rec[Reference Audio Storage]
    
    %% Analysis Stage
    subgraph Analysis [Voice Analysis & Profiling]
        FE[Feature Extractor] -->|Numerical Vectors| Pitch[Pitch F0 & Range]
        FE -->|RMS Energy Tensors| Energy[Stress & Energy]
        FE -->|Cadence Data| Rhythm[Speaking Pattern & Pace]
        
        ED[Emotion Detector] -->|Categorical Label| Emotion[Emotion & Confidence]
    end
    
    Rec -->|Time-Domain Signal| FE
    Rec -->|Spectrograms| ED

    %% Model Integration Stage
    subgraph Model [Qwen3-TTS Integration]
        QTTS[Qwen3-TTS 0.6B Engine]
        Clone[Zero-Shot Voice Cloning]
    end
    
    %% Logical Flow
    Pitch & Energy & Rhythm & Emotion -->|Structured JSON| Profile[Voice Profile Archive]
    Profile -.->|Style Embeddings| QTTS
    Rec -->|Style Reference| Clone
    
    %% Synthesis Stage
    TEXT[Input Text String] -->|NLP Tokens| Clone
    Clone -->|Latent Representations| QTTS
    QTTS -->|Digital Audio Samples| WAV[Output Cloned Audio]
    
    %% Output Metadata
    WAV -->|File Creation| Meta[JSON Voice Analysis]

    %% Styling
    classDef main fill:#f9f,stroke:#333,stroke-width:2px;
    classDef analysis fill:#ccf,stroke:#333,stroke-width:1px;
    classDef model fill:#cfc,stroke:#333,stroke-width:1px;
    
    class QTTS,Clone,FE,ED main;
    class Pitch,Energy,Rhythm,Emotion analysis;
    class WAV,Meta model;
```
