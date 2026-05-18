# Software Engineering Interview Q&A

This document contains a comprehensive list of the questions and answers discussed during the interview preparation session for the Personalized Speech Learning (TTS) project.

## 1. Project & Code Understanding

### Q: What is a simple overview of the interview preparation instructions?
**A:** The interviewers want to verify that you didn't just copy/paste code. They will evaluate you on:
1. **Code Understanding:** Knowing what goes into your functions, what happens inside them, and what comes out. Memorization is not required.
2. **Documentation:** Providing clear architecture diagrams showing how components interact.
3. **Logging & Performance:** Implementing a Python logger to track execution time, memory usage, and input metadata to prove you understand system constraints.

### Q: What does "Code Analysis" mean from a Software Engineering (SWE) Point of View?
**A:** In SWE, analysis means looking past the syntax to understand behavior. It involves:
*   **Tracing the Flow:** Following data from input to output.
*   **Evaluating Tradeoffs:** Knowing *why* you chose a specific tool (e.g., zero-shot cloning vs. training from scratch).
*   **Spotting Edge Cases:** Knowing how the system handles bad inputs.
*   **Understanding Costs:** Knowing the Time (speed) and Space (memory) complexity of your code.

### Q: What are the "Building Blocks" of a project?
**A:** Building blocks are the modular, individual pieces of code that connect like Lego bricks to form an application. For the TTS project:
1.  **Input Block:** Receives text and reference audio.
2.  **Processing Blocks:** The Feature Extractor and Emotion Detector analyze the inputs.
3.  **AI Engine:** The `qwen_adapter` uses PyTorch to generate speech from the inputs.
4.  **Output Block:** Converts raw data into a `.wav` file.
If one block breaks, it can be fixed without rewriting the entire system.

---

## 2. Hardware Fundamentals

### Q: What is the difference between Memory and Storage?
**A:** Think of an office workspace:
*   **Memory (RAM)** is your desk surface. It holds what you are actively working on so you can reach it instantly. It is small, and when the computer turns off (you go home), the desk is wiped clean (it is temporary).
*   **Storage (Hard Drive/SSD)** is the filing cabinet. It permanently holds all your files forever, but it is much slower because the computer has to walk over, find the file, and bring it to the "desk" to use it.

### Q: Is RAM a random access memory which stores data temporarily?
**A:** Yes! It is volatile (wiped when power is off) and "Random Access" means the CPU can fetch data from any random spot in the memory instantly, without having to read through it sequentially.

### Q: What changes happen in the RAM when opening apps like VS Code or running TTS?
**A:** 
1. **PC Off:** RAM is at 0GB.
2. **Boot Up:** OS is loaded from storage to RAM (3-4 GB used).
3. **Open VS Code/Antigravity:** App files are copied from storage to RAM so they run smoothly without lag.
4. **Run TTS Code:** The massive Qwen-TTS AI model weights are pulled from storage into the RAM (or GPU VRAM), causing a massive spike in memory usage. If the RAM gets totally full, the program crashes with an "Out of Memory" error.

### Q: How many bytes are in 1 GB?
**A:** It depends on the system:
*   **Base-10 (Hard drives & macOS):** 1,000,000,000 bytes (1 Billion).
*   **Base-2 (Windows & RAM):** 1,073,741,824 bytes (multiplied by 1024 instead of 1000).

### Q: Explain Storage from a Mobile vs. Computer point of view.
**A:** 
*   **Computer (The Walk-in Closet):** Massive physical space, incredibly fast (NVMe SSDs), and easily upgradable. You can open the case and add more storage anytime.
*   **Mobile (The Sewn-in Backpack):** Microscopic storage chips soldered directly to the motherboard. It uses almost zero battery power, but what you buy on day one is all you get—you cannot upgrade internal phone storage later.

---

## 3. Soft Skills & Psychology

### Q: What are Technical Vocabulary and Observation Skills?
**A:** 
*   **Technical Vocab (The Doctor):** Using precise words so other engineers know exactly what you mean (e.g., saying "Out of Memory Error" instead of "the computer got too full").
*   **Observation Skills (The Detective):** Paying close attention to logs and data to find bugs. Instead of guessing why code crashed, observing that memory spiked exactly 2 seconds before the crash.

### Q: What are Articulation Skills?
**A:** **The Translator.** It is the ability to take highly complex, confusing tech topics and explain them simply so anyone can understand. (e.g., Explaining an engine by saying "a pipe is leaking water" instead of rambling about "combustion friction ratios").

### Q: Why is Articulation important for a SWE?
**A:** You never code alone. You need to explain your logic to teammates so they can work with your code, translate tech problems into business terms for non-technical bosses, and pass interviews by clearly explaining your thought process.

### Q: What are the Feynman Technique and Fluid Intelligence?
**A:** 
*   **Feynman Technique:** The ultimate test of understanding. If you cannot explain a complex topic simply enough for a 12-year-old to understand without jargon, you don't truly understand it.
*   **Fluid Intelligence:** Your raw ability to use logic to solve brand-new, unseen problems on the fly (like debugging an AI error you've never encountered before), rather than just relying on memorized knowledge.

---

## 4. The AI Era of Software Engineering

### Q: What important skills should a SWE have in the current AI-augmented workflow?
**A:** Because AI can type code faster than humans, SWEs must transition from "Code Typers" to "System Directors." Key skills include:
1.  **Advanced Code Reviewing:** Spotting subtle bugs or security flaws in AI-generated code.
2.  **System Architecture:** Designing the massive blueprint that AI struggles to build.
3.  **Prompt Engineering:** Breaking massive problems down into tiny, logical steps for the AI.
4.  **Deep Debugging:** Reading logs to fix complex issues when the AI gets stuck guessing.
5.  **Business Logic:** Understanding what the user actually wants, which AI cannot do.
