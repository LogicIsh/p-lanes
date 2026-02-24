# p-lanes
### persistent local adaptive neural entry system

> **p-lanes: A modular microkernel/wrapper for llama.cpp focused on: home-lab scaled hardware, low-latency, and KV slot pinned users.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

---

## 🚀 What is p-lanes?

p-lanes is a lightweight orchestrator for local AI. This software is designed specifically to optimize the interface of consumer-grade systems with a low, fixed set of users. While most systems prioritize adaptability for a fluctuating user base, p-lanes focuses on minimizing latency and reducing overhead to provide maximum speed and quality for a dedicated home system.

The project was born out of frustration with existing software that didn't fit my specific "Household Scale" goal of:
- Dedicated Identity: 3 named users plus a "guest" account, each with unique AI personalities and unique "system privileges."
- Modular Recovery: A drop-in architecture that allows for heavy tinkering; if I "nuke" a system while experimenting or want to try something new, I don't have to re-code everything.
- The "Instant-On" Goal: Sub 2-second latency when a user activates the assistant, even after a full day of inactivity.

---

🧠 The Philosophy: Why p-lanes?
Engines like Aphrodite or vLLM are engineering marvels designed for enterprise-scale throughput. However, they are built to solve a problem most home-labs don't have: serving hundreds of concurrent users. In a local household environment, these frameworks often force trade-offs that degrade the user experience and punish consumer hardware.

Most frameworks handle your conversation memory (KV Cache) in one of three ways when the system is idle:
- Discarding Cache: The history is thrown away to save space. This leads to full re-tokenization (~2–10+ seconds of latency depending on context size).
- Swapping to RAM: This creates a significant RAM burden (~4GB per 32k token window per user, uncompressed) and minor latency hits as data moves back and forth across the system bus.
- Swapping to SSD: This leads to SSD wear and moderate latency hits (~1–3 seconds for context retrieval from disk).

By using llama.cpp as a lightweight foundation, p-lanes starts with significantly less overhead than feature-rich, memory-heavy alternatives. We then lock the engine into a "reserved seat" configuration. By pinning users to dedicated hardware slots, your memory stays exactly where it belongs—on the GPU—ensuring your assistant is always "warm" and ready for an instant response.

---

✨ Key Features
- Deterministic Slot Mapping: Users are assigned permanent VRAM slots for near-instant response. Fully adjustable capacity (e.g., 2 large-context slots or 10+ small-context slots, adjustable total window size).
- Optional Utility Lane: Background lanes for tasks like RAG retrieval or prompt fixing, stateless. This prevents "dirtying" the primary user history with system-level data.
- Automatic Summarization: Logic to compress older conversation context into summaries to prevent VRAM overflow while preserving long-term memory. Customizable and optional scheduled summarizations.
- Modular "Lane" Architecture: Drop-in discovery for Channels (IO) and Modules (Logic). Isolated lanes ensure a module failure doesn't crash the core kernel.
- Full Customization: Per-user control over model weights, summarization triggers, and KV window sizes.
- Minimalist Overhead: Headless, transparent code designed for 24/7 reliability on consumer-grade hardware.

---

⚠️ Limitations
To achieve deterministic low-latency, p-lanes makes several intentional trade-offs. This is a specialized household tool, not a general-purpose enterprise engine.
- "Always-On" Core Architecture: The p-lanes kernel and primary user slots are designed as persistent server processes. There is currently no mechanism to "load-on-request" for primary user slots; once the core is up, the VRAM for those users is pinned.
- Linear Slot Division: p-lanes operates as an extension of the llama.cpp slot system. VRAM is divided equally among all active slots; you cannot assign different context window sizes to different users in a single instance.
- Hard Slot Boundaries: Each user is locked into their allocated memory. A user's context window cannot "overflow" into another user's slot. If a user hits their limit, the summarization module must be triggered or risk request failure/trunication of context.
- Static VRAM Pre-allocation: Once a slot is pinned, that VRAM is reserved. It cannot be dynamically reclaimed for other tasks (like gaming) without stopping the engine.
- Compute Contention: While the memory is persistent, the GPU's cores are a shared resource. If multiple users trigger a request simultaneously, the tokens-per-second (TPS) will be divided across those active requests.
- Hardware Ceiling: VRAM is your hard limit. Performance is strictly tied to what fits physically on the GPU.

---

## 🔄 Data Flow

- Base Software Flow:
Input -> p-lanes(Provider -> Transport -> Logic -> Enricher -> Context Injector -> Service -> Post-Processor ) -> Output

Provider: The entry point for data. (/chat for text, /voice for a Whisper-transcribed stream, or automated system triggers)
Transport: The internal "plumbing." This standardizes varied input data into a unified format for the rest of the pipeline to read.
Logic: The "Brain" add-on for branching. (Semantic router that identifies a request for the weather and short-circuits the LLM to give an instant, pre-defined response)
Enricher: An injection module for real-world context. (Automatically tagging the current time, location, or live Home Assistant sensor data (e.g., "The living room lamp is currently ON") to the prompt)
Context Injector: An injection module for specialized memory. (Pulling data from a RAG vector store or using llama.cpp for pre-processing tasks)
Service: The primary data processing point. (Your pinned llama.cpp slot for conversation, or a "short-circuit" route for quick, specialized tasks)
Post-Processor: The final cleanup crew. (Stripping out "AI-isms," formatting text for a specific UI, or preparing the output for a Text-to-Speech (TTS) engine)

Example system (same settings, two different inputs):
Example 1: Knowledge-Based Chat (RAG)
A user asks a question about a personal document in a chat window.
1. Input: Text via Chat Box.
2. Provider: /chat endpoint.
3. Transport: Package text with UserID and SessionID, security gate 1 (can user access system).
5. Logic: Verifies intent, security gate 2 (can user access subprocess branch)
6. Context Injector: Security gate 3 (can user access specific module), routes the query to a stateless Utility Slot to perform RAG retrieval and pre-processing without touching the user's primary history (if avalible, if not prompt wrap and use user slot).
7. Service: Hits the user's Pinned llama.cpp Slot for final inference.
8. Post-Processor: Formats the answer with Markdown for the UI.
9. Output: Text displayed in the chat window.

Example 2: Low-Latency Voice Control
A user says "Turn on the kitchen lights".
1. Input: Audio Stream from microphone.
2. Provider: /voice endpoint.
3. Transport: Parallel processing of the audio stream (STT & voice print id), security gate 1 (can user access system).
4. Logic (Short-Circuit): Detects "Intent: Device Control." The logic module short-circuits the primary LLM to avoid inference latency, security gate 2 (can user access subprocess branch).
5. Enricher: Security gate 3 (can user access this module), converts prompt into device call format.
6. Service: Executes the Home Assistant API call directly.
7. Post-Processor: Triggers a success confirmation for the TTS engine.
8. Output: *Physical: Kitchen lights turn on.
9. Audio: Kokoro TTS says "Lights on" through the speakers.

---

## 💻 Requirements

### Theoretical Minimums:
-Hardware: A GPU and a working computer.
-Software: Python 3.12+, llama.cpp server, Home Assistant (currently used for automation logic), and linux OS (may become windows compatable later)

Tested Build (Development Environment):
- CPU: Intel Core Ultra 7
- RAM: 32GB
- SSD: 1TB, NVMe
- GPU: NVIDIA RTX 5060 Ti (16GB)
- OS: Proxmox (Linux VM), HAOS on VM, LXC container for p-lanes + llama.cpp

---

## 🗺️ Roadmap

p-lanes is currently in active development. The transition from a monolithic prototype to a modular microkernel is nearly complete.

Project History
v0.1.0: Monolithic structure. Proven concept with full text-chat functionality. (modular nightmare)
v0.2.0: First major redesign. Ported to Python package format and separated core logic from modules.
v0.3.0: (Current Phase) Architectural split of Providers from the Kernel Core for true modularity.

Development Status
[x] Design layout and system flow.
[x] Kernel Core coding and dispatch logic.
[ ] Core Configuration Implementation (Global slots, user windows, and system permissions).
[ ] Basic Provider (Text/API) setup and stability testing.
[ ] Public Alpha Release: Upload active builds to GitHub.
[ ] Voice Provider Implementation (Whisper + Kokoro integration).
[ ] Basic Logic & Enrichment Modules (Home Assistant integration).
[ ] Universal Installer Script & installer.py for automated setup.
[ ] Provider and Module Templates (Documentation for community contributors).
[ ] Cross-Platform Validation and final stability testing.

---

## ⚖️ License & Attribution

**p-lanes** is licensed under the **GNU AGPL-3.0**. 

This is a **Copyleft** project: you are free to modify and share it, but any derivative works must also be open-source, keep all original author attributions, and be licensed under the AGPL. 

### Third-Party Components
p-lanes is an orchestrator. The respective licenses of all integrated software remain in effect, including (but not limited to):
* **llama.cpp**: MIT License
* **Whisper**: MIT License
* **Kokoro**: Apache 2.0 License

*Original Author: Anthony Root (2026)*
