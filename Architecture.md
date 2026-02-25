# p-lanes — Architecture Reference v0.3.0

---

## System Overview

p-lanes is a self-hosted, multi-user home voice assistant running on a single Proxmox server with an NVIDIA RTX 5060 Ti 16GB GPU. It connects Home Assistant OS (HAOS) to a local LLM (llama.cpp) via a custom FastAPI mediator. The system supports persistent per-user personas, KV cache management, semantic routing, and optional feature modules.

**Design philosophy:** p-lanes is a microkernel. `main.py` is intentionally minimal — it receives a message, builds a context, runs the pipeline, and returns a response. All logic lives in core, service, or modules. `main.py` must never grow beyond its defined responsibilities.

**Extensibility model:** p-lanes has two tiers of drop-in components. Both are self-contained folders discovered automatically at startup — no manual registration, no editing core files.
- **Modules** change what the pipeline *does*. Drop a folder in `modules/`, restart (or `lanes reload`), and it starts firing on matching requests. Remove the folder to uninstall.
- **Providers** change what the core *is*. Swap an input provider and the system identifies users differently. Swap an output provider and responses come back as a different voice. Providers are self-contained folders under `core/providers/` with their own manifests and config.

**Multi-channel:** p-lanes supports simultaneous input channels — voice, chat, and future vision — each with its own input/output provider pair. All channels share the same pipeline, security gates, user slots, and conversation context. Speaking in the kitchen and typing on your phone both land on the same slot as the same user.

**Hardware context:**
- GPU: NVIDIA RTX 5060 Ti 16GB
- RAM: 32GB
- Virtualization: Proxmox VE (bare-metal hypervisor) with LXC containers and GPU passthrough
- Primary LLM: Qwen3-VL 8B Q6_K_M via llama.cpp server
- KV Cache: 5 slots (4 persistent users + 1 utility/guest), 12k context per slot, Q6 compression
- VRAM budget: ~12.25GB used, ~3GB safety buffer

---

## Package Structure

```
p-lanes/
├── main.py                          # Microkernel entry point (~4 lines)
├── config.py                        # Core-only: pipeline, channels, slots, security floors, paths
├── requirements.txt
├── install.sh                       # Python version check only, hands off to setup.py
├── setup.py                         # Guided interactive installer wizard
│
├── core/                            # Always loaded. No feature flags. No optional imports.
│   ├── __init__.py
│   ├── llm.py                       # llama.cpp start, stop, call_slot(), parse response
│   ├── slots.py                     # User objects, slot assignment, locks, ephemeral flag
│   ├── transport.py                 # FastAPI HTTP server, channel routing, transcript SSE
│   ├── context.py                   # Context dataclass — flows through the pipeline
│   ├── registry.py                  # Auto-discovery scanner, @register decorator, stage + intent mapping
│   ├── config_loader.py             # load_addon_config() — reads config.yaml from addon folders
│   ├── responder.py                 # Built-in LLM respond module (registered to "responder" stage)
│   ├── summarizer.py                # Summarization logic, context injection, wipe/reinject
│   │
│   ├── providers/                   # Swappable core components — one active per channel per capability
│   │   ├── __init__.py              # Provider scanner + loader
│   │   ├── base.py                  # Base interfaces, Attachment, ProcessedInput, OutputResult
│   │   ├── input/                   # Input processing providers (each is a self-contained folder)
│   │   │   ├── stt_device_map/      # Pure STT — audio in, text + device_id mapping out
│   │   │   │   ├── provider.yaml    # Manifest: name, capability, description
│   │   │   │   ├── config.yaml      # Provider settings (STT URL, device map, etc.)
│   │   │   │   └── provider.py      # Implementation + provider singleton
│   │   │   ├── stt_voiceprint/      # STT + speaker ID in parallel
│   │   │   │   ├── provider.yaml
│   │   │   │   ├── config.yaml
│   │   │   │   └── provider.py
│   │   │   ├── multimodal/          # Text + audio + images — full multimodal input
│   │   │   │   ├── provider.yaml
│   │   │   │   ├── config.yaml
│   │   │   │   └── provider.py
│   │   │   └── text_only/           # No audio — text API input (chat/testing/dev)
│   │   │       ├── provider.yaml
│   │   │       └── provider.py      # No config.yaml needed — no external services
│   │   └── output/                  # Output processing providers
│   │       ├── kokoro_tts/          # Kokoro TTS
│   │       │   ├── provider.yaml
│   │       │   ├── config.yaml
│   │       │   └── provider.py
│   │       ├── piper_tts/           # Piper TTS (lighter alternative)
│   │       │   ├── provider.yaml
│   │       │   ├── config.yaml
│   │       │   └── provider.py
│   │       └── text_only/           # No audio — text response (chat/testing/dev)
│   │           ├── provider.yaml
│   │           └── provider.py
│   │
│   └── tools/                       # Built-in system tools, accessible via chat ("lanes ...")
│       ├── __init__.py              # Exposes handle(), registers to "processor" stage
│       ├── tool_runner.py           # Parses command, dispatches to correct tool
│       ├── base.py                  # BaseTool class
│       └── builtins/
│           ├── pipeline.py          # Pipeline inspection per intent
│           ├── slots.py             # Slot state viewer
│           ├── debug.py             # Toggle debug mode, trace requests
│           ├── trace.py             # Post-hoc pipeline trace viewer
│           ├── config_view.py       # Read-only config dump
│           ├── history.py           # Conversation/summary viewer
│           ├── health.py            # System health check (GPU, slots, disk, uptime)
│           ├── wipe.py              # Force-wipe KV cache for a user or slot
│           ├── reload.py            # Hot-reload module registry
│           ├── channels.py          # Show active channels and their providers
│           └── test.py              # Run module self-tests
│
├── service/
│   ├── __init__.py
│   └── dispatcher.py               # Walks pipeline stages, enforces Gate 2 + Gate 3, runs modules
│
├── modules/                         # Optional. Each module is a self-contained drop-in folder.
│   ├── intent_classifier/           # Semantic routing module
│   │   ├── module.yaml              # Manifest: name, enabled, stage, intents, security_level
│   │   ├── config.yaml              # Module-specific settings (optional)
│   │   ├── __init__.py              # Exposes handle()
│   │   └── intent_classifier.py     # Runtime logic
│   ├── rag/                         # Retrieval augmented generation module
│   │   ├── module.yaml
│   │   ├── config.yaml
│   │   ├── __init__.py
│   │   └── rag.py
│   ├── rag_processor/               # RAG data processing via utility slot
│   │   ├── module.yaml
│   │   ├── config.yaml
│   │   ├── __init__.py
│   │   └── rag_processor.py
│   ├── ha_bridge/                   # Home Assistant device control
│   │   ├── module.yaml
│   │   ├── config.yaml
│   │   ├── __init__.py
│   │   └── ha_bridge.py
│   └── config_manager/              # Admin config changes via chat
│       ├── module.yaml
│       ├── __init__.py
│       └── config_manager.py
│
├── users/                           # Per-user runtime data (maps to /var/lib/p-lanes/users/)
│   └── {user_id}/
│       ├── profile.json             # Persona, security level, slot, voice enrollment (future)
│       ├── summary.txt              # Current rolling conversation summary
│       └── history.db               # SQLite conversation history
│
└── docs/
    ├── ARCHITECTURE.md              # This file
    ├── INSTALL.md
    ├── MODULES.md                   # Module authoring guide
    └── PROVIDERS.md                 # Provider authoring guide
```

---

## Pipeline Architecture

The pipeline is a sequence of named stages defined in config. Every request flows through all stages in order. Modules register to specific stages and declare which intents they handle. The dispatcher walks the stages, checks security, matches intent, and runs matching modules.

**The LLM is not a special case.** It is a tool that any module can invoke at any stage on any slot via `ctx.call_slot()`. The default user-facing LLM call is a built-in module (`core/responder.py`) registered to the `responder` stage.

**Pipeline stages (defined in config.py):**

```python
PIPELINE = ["classifier", "enricher", "processor", "responder", "finalizer"]
```

| Stage | Purpose | Example modules |
|-------|---------|-----------------|
| `classifier` | Classify intent, set ctx.intent | intent_classifier |
| `enricher` | Gather data, inject context | rag |
| `processor` | Transform data, call utility slot | rag_processor, ha_bridge, system_tools |
| `responder` | Generate user-facing response via LLM | llm_respond (built-in) |
| `finalizer` | Post-response actions, overrides | config_manager |

Adding a new stage is a one-line config change. Module registration determines which modules fire at each stage. The dispatcher has zero knowledge of what modules exist or what they do.

---

## Channels

Channels are named transport paths, each with their own input/output provider pair. All channels share the same pipeline, security gates, user slots, and conversation context. Multiple channels can be active simultaneously.

**Config defines all active channels:**

```python
# config.py
CHANNELS = {
    "voice": {
        "input":  "stt_voiceprint",
        "output": "kokoro_tts",
    },
    "chat": {
        "input":  "text_only",
        "output": "text_only",
    },
    # Future
    # "vision": {
    #     "input":  "multimodal",
    #     "output": "text_only",
    # },
}
```

**Each channel gets its own endpoint:** `/channel/voice`, `/channel/chat`, etc. The transporter routes to the correct provider pair based on the channel name. The channel name is recorded in `ctx.metadata["input"]["channel"]` so modules can optionally adapt behavior per channel (e.g., shorter responses for voice, markdown for chat).

**Same user, same slot, any channel.** You speak in the kitchen and it lands on slot 0 as "dad". You type on your phone and it lands on slot 0 as "dad". The LLM sees one continuous conversation regardless of how input arrived. The slot lock serializes concurrent requests from different channels.

**Response follows the request.** Voice input returns audio through the speaker. Chat input returns text to the chatbox. You won't get TTS blaring from the kitchen speaker because you quietly typed something on your phone at 2 AM.

Adding a new channel is a config change — define the name and its provider pair. No transport changes, no pipeline changes, no module changes.

---

## Transcript Stream

The transporter optionally exposes an SSE (Server-Sent Events) endpoint that pushes a live feed of both sides of a conversation. Any connected client — such as a HAOS dashboard card — receives the user's message and p-lanes' response in real time for any channel.

**Enabled via config flag:**
```python
# config.py
ENABLE_TRANSCRIPT_SSE = True    # Set to False to disable
```

**Endpoint:** `/transcript/{user_id}`

**Events carry:** role (user/assistant), text, source channel, and timestamp.

**Behavior:**
- Voice conversation transcripts appear in the chatbox as they happen
- Chat messages also appear in the stream (unified timeline across channels)
- The transcript stream is read-only — it mirrors the conversation, never alters it
- The SSE endpoint is gated behind Gate 1 — ADMIN can watch any user's stream, USER can only watch their own

**Integration:** After the pipeline runs and the response is sent back through the originating channel, the transporter broadcasts both the user message and the assistant response to all SSE subscribers for that user.

The transcript stream requires no pipeline changes, no module changes, and no provider changes. It is a transport-level concern that reads the same data already flowing through the system.

---

## Providers

Providers are swappable core components that implement a required capability. They run outside the pipeline (before input reaches it, after output leaves it) and are selected by name in channel config. Each provider is a self-contained folder with its own manifest, config, and implementation.

**Modules vs. Providers:**

| | Modules | Providers |
|---|---|---|
| Where | `modules/` | `core/providers/{capability}/{name}/` |
| When | During pipeline stages | Before or after the pipeline |
| Optional | Yes (enabled flag in manifest) | No (exactly one active per channel per capability) |
| Multiple | Many can run per request | One per capability per channel |
| Discovery | Auto-scanned from `modules/` | Auto-scanned from `core/providers/` |
| Config | Own `config.yaml` | Own `config.yaml` |
| Hot-reload | Yes (`lanes reload`) | No (restart required) |
| Custom routes | No | Optional (`register_routes(app)`) |

### Provider Manifest

```yaml
# core/providers/input/stt_voiceprint/provider.yaml
name: stt_voiceprint
capability: input
description: "Parallel STT + speaker identification via voiceprint"
has_custom_routes: false
```

### Provider Self-Contained Config

```yaml
# core/providers/input/stt_voiceprint/config.yaml
stt_url: "http://localhost:8080"
voiceprint_url: "http://localhost:8081"
voiceprint_threshold: 0.82
```

Providers read their own config using the shared config loader:

```python
from core.config_loader import load_addon_config
cfg = load_addon_config(__file__)   # Finds config.yaml next to this file
# cfg.stt_url, cfg.voiceprint_threshold, etc.
```

### Provider Base Interfaces

```python
# core/providers/base.py

@dataclass
class Attachment:
    type: str              # "image", "document", etc.
    data: bytes            # Raw bytes
    mime_type: str         # "image/jpeg", "image/png", "application/pdf"
    metadata: dict = field(default_factory=dict)  # dimensions, source, etc.

@dataclass
class ProcessedInput:
    user_id: str
    message: str
    metadata: dict = field(default_factory=dict)
    attachments: list[Attachment] = field(default_factory=list)
    # metadata can carry: confidence, language, voiceprint_score,
    # device_id, audio_duration, etc.

@dataclass
class OutputResult:
    data: bytes | str             # Audio bytes or text passthrough
    content_type: str             # "audio/wav", "text/plain", etc.

class InputProvider:
    """Transforms raw input into a structured message for the pipeline."""
    name: str

    async def process(self, raw_request: dict) -> ProcessedInput:
        """Returns user_id, message text, attachments, and any metadata."""
        raise NotImplementedError

    async def startup(self) -> None:
        """Called once at boot. Load models, warm up, etc."""
        pass

    async def shutdown(self) -> None:
        """Called at exit. Cleanup."""
        pass

    def register_routes(self, app: FastAPI) -> None:
        """Optional. Register custom endpoints on the FastAPI app.
        Called once during startup. Most providers don't implement this."""
        pass

class OutputProvider:
    """Transforms pipeline text output into a deliverable response."""
    name: str

    async def process(self, text: str, user: User) -> OutputResult:
        """Returns audio bytes, stream handle, or passthrough text."""
        raise NotImplementedError

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def register_routes(self, app: FastAPI) -> None:
        """Optional. Register custom endpoints."""
        pass
```

### Provider Scanner

```python
# core/providers/__init__.py

def scan_providers(capability: str) -> dict[str, ProviderInfo]:
    """Scan a capability folder for provider subfolders with provider.yaml."""
    providers = {}
    base = Path(f"core/providers/{capability}")
    for path in base.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        manifest = path / "provider.yaml"
        if not manifest.exists():
            continue
        info = yaml.safe_load(manifest.read_text())
        providers[info["name"]] = info
    return providers

def load_provider(capability: str, name: str):
    """Load a provider by capability type and name.
    e.g. load_provider("input", "stt_voiceprint")"""
    module = importlib.import_module(f"core.providers.{capability}.{name}.provider")
    provider = module.provider  # each file exposes a singleton
    assert isinstance(provider, BASE_CLASSES[capability])
    return provider
```

### Example Input Provider: stt_voiceprint

Parallel speech-to-text and speaker identification. STT and voiceprint engines are configured per-provider — the core doesn't know or care which STT/TTS software is behind the provider.

```python
# core/providers/input/stt_voiceprint/provider.py

from core.config_loader import load_addon_config

cfg = load_addon_config(__file__)

class SttVoiceprint(InputProvider):
    name = "stt_voiceprint"

    async def startup(self):
        self.stt = SttClient(cfg.stt_url)
        self.voiceprint = VoiceprintClient(cfg.voiceprint_url)

    async def process(self, raw_request: dict) -> ProcessedInput:
        audio = raw_request["audio"]
        transcript, speaker = await asyncio.gather(
            self.stt.transcribe(audio),
            self.voiceprint.identify(audio)
        )
        user_id = speaker.user_id if speaker.confidence >= cfg.voiceprint_threshold else "guest"

        return ProcessedInput(
            user_id=user_id, message=transcript.text,
            metadata={"stt_confidence": transcript.confidence,
                       "speaker_confidence": speaker.confidence,
                       "speaker_id": speaker.user_id, "identified_by": "voiceprint"}
        )

provider = SttVoiceprint()
```

### Example Input Provider: text_only

No audio processing. Accepts text directly. Used for chat channel, development, and testing.

```python
# core/providers/input/text_only/provider.py

class TextOnly(InputProvider):
    name = "text_only"

    async def process(self, raw_request: dict) -> ProcessedInput:
        return ProcessedInput(
            user_id=raw_request.get("user_id", "guest"),
            message=raw_request["message"],
            metadata={"identified_by": "request_field"}
        )

provider = TextOnly()
```

### Example Input Provider: multimodal

Handles text, audio, and images. Used for the future vision channel.

```python
# core/providers/input/multimodal/provider.py

from core.config_loader import load_addon_config

cfg = load_addon_config(__file__)

class Multimodal(InputProvider):
    name = "multimodal"

    async def startup(self):
        self.stt = SttClient(cfg.stt_url)
        self.voiceprint = VoiceprintClient(cfg.voiceprint_url)

    async def process(self, raw_request: dict) -> ProcessedInput:
        result = ProcessedInput(
            user_id=raw_request.get("user_id", "guest"),
            message="", metadata={"identified_by": "request_field"}
        )

        if "message" in raw_request:
            result.message = raw_request["message"]

        if "audio" in raw_request:
            transcript, speaker = await asyncio.gather(
                self.stt.transcribe(raw_request["audio"]),
                self.voiceprint.identify(raw_request["audio"])
            )
            result.message = transcript.text
            if speaker.confidence >= cfg.voiceprint_threshold:
                result.user_id = speaker.user_id
            result.metadata.update({
                "stt_confidence": transcript.confidence,
                "speaker_confidence": speaker.confidence,
                "identified_by": "voiceprint"
            })

        if "images" in raw_request:
            result.attachments = [
                Attachment(
                    type="image",
                    data=base64.b64decode(img["data"]),
                    mime_type=img.get("mime_type", "image/jpeg"),
                )
                for img in raw_request["images"]
            ]

        return result

provider = Multimodal()
```

### Provider Custom Routes

Providers can optionally register custom endpoints on the FastAPI app. This is for oddball integrations like sensor data receivers that don't fit the standard channel model.

```python
# core/providers/input/sensor_bridge/provider.py

class SensorBridge(InputProvider):
    name = "sensor_bridge"

    def register_routes(self, app: FastAPI):
        @app.post("/sensor")
        async def receive_sensor(data: dict):
            self.latest_readings[data["sensor_id"]] = data
            return {"status": "ok"}

        @app.get("/sensor/{sensor_id}")
        async def get_sensor(sensor_id: str):
            return self.latest_readings.get(sensor_id, {})

    async def process(self, raw_request: dict) -> ProcessedInput:
        # Normal input processing — can reference self.latest_readings
        ...
```

The transporter calls `register_routes(app)` on every active provider during startup. Remove the provider, the endpoint disappears. No transport code changes needed.

### Future Provider Capabilities

Adding a new *implementation* of an existing capability requires no code changes — drop a folder with a manifest, reference it in a channel config, restart. Adding a new *capability type* requires a new base class in `base.py` and a new folder under `core/providers/`.

```python
# config.py — as the system grows, new standalone capabilities
EMBED_PROVIDER  = "nomic_embed"        # For RAG embeddings
STORE_PROVIDER  = "chromadb"           # Swap vector store
```

---

## Component Descriptions

### main.py
The microkernel. Builds a Context, runs the pipeline, returns the result. Must never contain business logic.

```python
async def handle_message(user_id: str, message: str,
                         input_metadata: dict = None,
                         attachments: list = None) -> str:
    user = slots.get_user(user_id)
    ctx = Context(user=user, message=message, _llm=llm)
    if input_metadata:
        ctx.metadata["input"] = input_metadata
    if attachments:
        ctx.attachments = attachments
    ctx = await dispatcher.run(ctx)
    return ctx.final_text
```

### config.py
Core-only system configuration. Contains only settings that are truly system-wide. Module-specific and provider-specific settings live in their own `config.yaml` files.

**Config.py contains:**
- LLM server command and all llama.cpp parameters
- Pipeline stage list (PIPELINE)
- Channel definitions (CHANNELS dict with input/output provider names)
- Stage security floors (STAGE_SECURITY)
- Utility slot assignment (UTILITY_SLOT — set to None if unavailable)
- Slot count, context window size, KV cache settings
- File paths for users, models, logs
- Summarization thresholds (THRESHOLD_WARN, THRESHOLD_CRIT)
- SecurityLevel IntEnum definition
- Feature flags for core features (ENABLE_TRANSCRIPT_SSE, etc.)

**Config.py does NOT contain:**
- Module enable flags (lives in `module.yaml`)
- Module permission levels (lives in `module.yaml`)
- Module-specific settings (lives in module's `config.yaml`)
- Provider-specific settings (lives in provider's `config.yaml`)

No addon ever writes to config.py. All components read from config at runtime, never write.

```python
# config.py — complete example

from enum import IntEnum

class SecurityLevel(IntEnum):
    GUEST   = 0
    USER    = 1
    TRUSTED = 2
    ADMIN   = 3

# Pipeline
PIPELINE = ["classifier", "enricher", "processor", "responder", "finalizer"]

# Security floors per stage — only the system owner can change these
STAGE_SECURITY = {
    "classifier": SecurityLevel.GUEST,
    "enricher":   SecurityLevel.USER,
    "processor":  SecurityLevel.USER,
    "responder":  SecurityLevel.GUEST,
    "finalizer":  SecurityLevel.TRUSTED,
}

# Channels
CHANNELS = {
    "voice": {"input": "stt_voiceprint", "output": "kokoro_tts"},
    "chat":  {"input": "text_only",      "output": "text_only"},
}

# LLM
LLM_CMD = "..."
SLOT_COUNT = 5
CONTEXT_SIZE = 12288
UTILITY_SLOT = 4            # Set to None if no utility slot available

# Thresholds
THRESHOLD_WARN = 8000
THRESHOLD_CRIT = 10000

# Features
ENABLE_TRANSCRIPT_SSE = True

# Paths
USERS_DIR = "/var/lib/p-lanes/users"
MIN_ACCESS_LEVEL = SecurityLevel.GUEST
```

### core/config_loader.py
Shared utility for loading self-contained addon config files. Both modules and providers use this.

```python
# core/config_loader.py

def load_addon_config(caller_file: str) -> SimpleNamespace:
    """Load config.yaml from the same directory as the calling file.
    Returns an empty namespace if no config.yaml exists."""
    config_path = Path(caller_file).parent / "config.yaml"
    if not config_path.exists():
        return SimpleNamespace()
    data = yaml.safe_load(config_path.read_text())
    return SimpleNamespace(**data)
```

Usage from any module or provider:
```python
from core.config_loader import load_addon_config
cfg = load_addon_config(__file__)
# cfg.some_setting, cfg.some_url, etc.
```

### core/context.py
The single data object that flows through the entire pipeline. Every module receives it, reads from it, writes to it, and returns it. The context is the API surface between all components.

```python
@dataclass
class Context:
    user: User
    message: str

    # Set by classifier
    intent: str | None = None

    # Attachments — images, documents, etc. from multimodal input
    attachments: list[Attachment] = field(default_factory=list)

    # Module communication — read/write freely
    metadata: dict = field(default_factory=dict)

    # Enrichment data — modules append, responder module injects into prompt
    prompt_extras: list[str] = field(default_factory=list)

    # Response — set by responder module, overridable by finalizer modules
    response: LLMResponse | None = None
    final_text: str | None = None

    # LLM access — injected by main.py at construction, never imported by modules
    _llm: LLMInterface = field(repr=False, default=None)

    async def call_slot(self, slot: int, system: str, content: str) -> LLMResponse:
        """Call a specific slot. Acquires slot lock automatically.
        Ephemeral slots are wiped after each call."""
        return await self._llm.call_slot(slot, system, content)

    async def call_utility(self, system: str, content: str) -> LLMResponse:
        """Call the utility slot if available, otherwise fall back to user slot.
        Modules should use this for all data processing work.
        Caller never needs to know which path was taken."""
        utility = self._llm.get_utility_slot()
        if utility is not None:
            return await self.call_slot(utility.slot, system, content)
        return await self._fallback_user_slot(system, content)

    async def _fallback_user_slot(self, system: str, content: str) -> LLMResponse:
        """Process on the user's slot when no utility slot is available.
        Injects a one-shot processing directive as user-role content
        so it doesn't replace the user's persona or system prompt."""
        wrapped = (
            "[PROCESSING TASK — respond with only the requested output, "
            "no conversation]\n\n"
            f"{system}\n\n{content}"
        )
        response = await self.call_slot(self.user.slot, None, wrapped)
        response.used_utility = False
        self.metadata["utility_fallback"] = True
        return response
```

**The three-tier LLM API on Context:**

| Method | Use case |
|---|---|
| `ctx.call_slot(N, system, content)` | Explicit slot targeting — you know exactly which slot and why |
| `ctx.call_utility(system, content)` | Data processing — use utility if available, degrade gracefully to user slot |
| `ctx.call_slot(ctx.user.slot, ...)` | User-facing response — what `llm_respond` does |

Modules should use `call_utility()` for any processing work. This allows the system to adapt to deployments without a utility slot — zero module changes required, just set `UTILITY_SLOT = None` in config.

**Utility fallback tradeoffs:** When falling back to the user slot, raw data enters the user's context window (consuming budget toward summarization), the persona and history may influence processing output, and the user slot is blocked during processing. The `utility_fallback` flag signals the summarizer to be more aggressive on the next cycle.

Modules never import `llm` directly. They access LLM capabilities exclusively through `ctx.call_slot()` and `ctx.call_utility()`.

### core/llm.py
Manages the llama.cpp server process lifecycle and all LLM communication. Responsibilities:
- Start and stop llama.cpp as a subprocess using LLM_CMD from config
- Expose `call_slot(slot, system, content)` as the single LLM primitive
- Expose `get_utility_slot()` which returns the utility User object or None based on `config.UTILITY_SLOT`
- Acquire the target slot's asyncio.Lock before every call, release on completion or exception
- Always pass full sampling parameters per request (never rely on server defaults)
- Parse response and extract `usage.total_tokens` and `truncated` fields
- Set `user.flag_crit` if `truncated` is True or `total_tokens` exceeds THRESHOLD_CRIT
- Set `user.flag_big` if `total_tokens` exceeds THRESHOLD_WARN
- If the target slot is ephemeral, wipe its KV cache after the response is parsed
- Never accumulate token counts across turns
- When `ctx.attachments` are present, format multimodal content blocks in the llama.cpp request (Qwen3-VL supports image content natively)

### core/slots.py
Manages all user state. Responsibilities:
- Load user profiles from `users/{user_id}/profile.json` at startup
- Assign and track slot numbers per user
- Maintain `asyncio.Lock()` per slot for concurrency control
- Expose `check_permission(user, required_level)` for security enforcement
- Store per-user flags: `flag_big`, `flag_crit`, `is_idle`
- Track ephemeral flag per slot

**User profile schema (profile.json):**
```json
{
    "user_id": "dad",
    "slot": 0,
    "security_level": 3,
    "persona": "System prompt text for this user",
    "voice_id": "dad_voiceprint_embed",
    "rag_scope": [],
    "ephemeral": false
}
```

**Utility slot profile:**
```json
{
    "user_id": "utility",
    "slot": 4,
    "security_level": 3,
    "ephemeral": true
}
```

Ephemeral slots are stateless by design. KV cache is wiped after every `call_slot()` response to prevent cross-user data leakage.

### core/transport.py
FastAPI HTTP server (the "transporter"). Routes requests to channels, manages the optional transcript SSE stream, calls provider custom routes, and enforces Gate 1 security.

```python
# Scan and load all channel provider pairs at startup
channels = {}
for name, cfg in config.CHANNELS.items():
    channels[name] = {
        "input": load_provider("input", cfg["input"]),
        "output": load_provider("output", cfg["output"]),
    }

@app.on_event("startup")
async def startup():
    for ch in channels.values():
        await ch["input"].startup()
        await ch["output"].startup()
        # Register any custom routes the provider declares
        ch["input"].register_routes(app)
        ch["output"].register_routes(app)

@app.post("/channel/{channel_name}")
async def receive(channel_name: str, raw_request: dict):
    channel = channels.get(channel_name)
    if not channel:
        return {"error": "unknown channel"}

    # Input provider handles audio/text/image processing + user identification
    processed = await channel["input"].process(raw_request)

    # Gate 1 — is this user known to the system?
    user = slots.get_user(processed.user_id)
    if not user or user.security_level < config.MIN_ACCESS_LEVEL:
        return {"error": "unauthorized"}

    # Broadcast user message to transcript subscribers (if enabled)
    if config.ENABLE_TRANSCRIPT_SSE:
        await broadcast_transcript(processed.user_id, {
            "role": "user", "text": processed.message,
            "channel": channel_name, "timestamp": datetime.now().isoformat()
        })

    # Pipeline
    response_text = await handle_message(
        processed.user_id, processed.message,
        input_metadata={**processed.metadata, "channel": channel_name},
        attachments=processed.attachments
    )

    # Broadcast assistant response to transcript subscribers (if enabled)
    if config.ENABLE_TRANSCRIPT_SSE:
        await broadcast_transcript(processed.user_id, {
            "role": "assistant", "text": response_text,
            "channel": channel_name, "timestamp": datetime.now().isoformat()
        })

    # Output provider handles TTS or text passthrough
    output = await channel["output"].process(response_text, user)
    return output.to_response()
```

The transporter doesn't know or care what's behind the providers. Adding a new channel is a config change. The transcript stream is a read-only mirror that requires no pipeline or module awareness.

### core/registry.py
Module auto-discovery and registration. At startup, scans `modules/` for folders containing `module.yaml`, imports enabled modules, and maps them to pipeline stages.

```python
# core/registry.py

def auto_discover_modules():
    """Scan modules/ for drop-in folders with module.yaml manifests."""
    for path in Path("modules").iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        manifest = path / "module.yaml"
        if not manifest.exists():
            continue
        cfg = yaml.safe_load(manifest.read_text())
        if not cfg.get("enabled", False):
            continue

        # Import the module — @register decorator fires on import
        module = importlib.import_module(f"modules.{path.name}")

        # Attach manifest metadata to the registered module
        registered = registry[cfg["stage"]][cfg["name"]]
        registered.security_level = cfg.get("security_level", 0)
        registered.description = cfg.get("description", "")
```

- `intents=["*"]` means the module runs for every intent (e.g., intent_classifier, llm_respond).
- Specific intent lists mean the module only runs when `ctx.intent` matches.
- Registration order within a stage determines execution order by default.
- Optional `order=N` parameter allows explicit ordering within a stage for edge cases.

### core/responder.py
The built-in LLM response module. Registered to the `responder` stage with `intents=["*"]`. Calls the user's assigned slot with the full prompt. If `ctx.final_text` is already set by an earlier module, it skips the LLM call. If `ctx.attachments` are present, formats multimodal content blocks into the request.

```python
@register("llm_respond", stage="responder", intents=["*"])
async def handle(ctx: Context) -> Context:
    if ctx.final_text:
        return ctx
    ctx.response = await ctx.call_slot(
        slot=ctx.user.slot,
        system=build_prompt(ctx),
        content=build_content(ctx)  # text + attachments if present
    )
    ctx.final_text = ctx.response.content
    return ctx
```

**Multimodal content building:**
```python
def build_content(ctx: Context) -> str | list[dict]:
    """If attachments are present, return multimodal content blocks.
    Otherwise return plain text."""
    if not ctx.attachments:
        return ctx.message

    content = []
    for att in ctx.attachments:
        if att.type == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{att.mime_type};base64,{b64encode(att.data).decode()}"}
            })
    content.append({"type": "text", "text": ctx.message})
    return content
```

This module is swappable. A custom responder module with a narrower intent filter can override behavior for specific intents.

### core/summarizer.py
Manages conversation summarization and context injection. Responsibilities:
- Generate summary of conversation history using the LLM
- Wipe the slot KV cache
- Reinject: system prompt + persona + summary + recent history
- Triggered by `flag_crit` (immediate) or `flag_big` + `is_idle` (background)
- Acquires `user.slot_lock` before summarizing, releases on completion or exception
- `asyncio.Lock` ensures summarization and incoming messages never collide
- Checks `ctx.metadata["utility_fallback"]` — if a utility fallback occurred, lower thresholds for the next summarization cycle

**Summarization trigger logic:**
```
After every LLM response:
    read total_tokens from usage
    if truncated == True → flag_crit = True
    elif total_tokens > THRESHOLD_CRIT → flag_crit = True
    elif total_tokens > THRESHOLD_WARN → flag_big = True

    if flag_crit:
        fire summarize_if_needed() as background task
        return response to user immediately

Background timer loop:
    if flag_big and user.is_idle():
        trigger summarization
```

**Incoming message lock behavior:**
```
if slot_lock.locked():
    send "Give me just a second..." to user
    wait up to 6 seconds for lock to release
    if still locked after 6s: drop message silently
```

### service/dispatcher.py
The pipeline engine. Walks all stages in order, enforces Gate 2 (stage floor) and Gate 3 (module permission), filters by intent, and runs matching modules. Zero p-lanes-specific logic — it is a generic pipeline executor.

```python
async def run(ctx: Context) -> Context:
    tracing = ctx.metadata.get("debug", False)
    trace = []

    for stage in config.PIPELINE:

        # GATE 2 — stage security floor from core config
        stage_min = config.STAGE_SECURITY.get(stage, SecurityLevel.GUEST)
        if ctx.user.security_level < stage_min:
            continue  # Skip entire stage, no module evaluation

        for module in registry.get(stage, ctx.intent):

            # GATE 3 — module's own declared security level
            if ctx.user.security_level < module.security_level:
                continue  # Skip this module silently

            start = time.monotonic()
            ctx = await module.handle(ctx)
            elapsed = (time.monotonic() - start) * 1000

            if tracing:
                trace.append(f"[debug] {stage}: {module.name} ({elapsed:.0f}ms)")

    if tracing:
        ctx.metadata["trace"] = trace
        trace_buffer.append(TraceEntry(
            timestamp=datetime.now(), user=ctx.user.user_id,
            intent=ctx.intent, trace=trace
        ))
    return ctx
```

The trace buffer is a fixed-size `collections.deque` in memory. No disk IO, no cleanup, auto-evicts old entries.

---

## Security Model

Security is enforced at three hard gates. The LLM is never a decision maker for access control. Each gate can only raise the bar, never lower it.

**Gate 1 — Identity (core/transport.py)**
WHO are you? Runs after the input provider identifies the user but before anything reaches the pipeline. Unknown users are dropped immediately — no response, no pipeline, nothing. The transcript SSE endpoint is also gated — ADMIN can watch any user's stream, USER can only watch their own.

**Gate 2 — Stage Access Floor (service/dispatcher.py)**
WHERE can you go? Core config defines a minimum security level per pipeline stage. The dispatcher checks the user's level against the stage floor before evaluating any modules. If the user is below the floor, the entire stage is skipped. Only the system owner can change these floors in config.py. No addon can modify them.

**Gate 3 — Module Permission (service/dispatcher.py)**
CAN you use this specific thing? Each module declares its own required security level in `module.yaml`. The dispatcher checks the user's level against the module's declared level. A module can make itself stricter than the stage floor but never more permissive — Gate 2 always runs first.

**Gate evaluation order:** Gate 2 runs before Gate 3. A drop-in module declaring `security_level: 0` is meaningless if it registers to a stage with a floor of 1.

```python
# config.py — stage security floors (core-owned, never touched by addons)
STAGE_SECURITY = {
    "classifier": SecurityLevel.GUEST,     # Must run for everyone
    "enricher":   SecurityLevel.USER,      # No enrichment for guests
    "processor":  SecurityLevel.USER,      # No processing for guests
    "responder":  SecurityLevel.GUEST,     # Everyone gets a response
    "finalizer":  SecurityLevel.TRUSTED,   # Post-processing for trusted+
}
```

**The flow for each security level:**

```
GUEST (0)
    Gate 1: ✓ recognized user
    Gate 2: classifier ✓ → enricher ✗ → processor ✗ → responder ✓ → finalizer ✗
    Gate 3: only modules with security_level 0 within cleared stages
    Result: gets classified, gets a basic LLM response, nothing else

USER (1)
    Gate 1: ✓ recognized user
    Gate 2: classifier ✓ → enricher ✓ → processor ✓ → responder ✓ → finalizer ✗
    Gate 3: modules up to level 1
    Result: full pipeline minus finalizer, but modules declaring level 2+ still skipped

TRUSTED (2)
    Gate 1: ✓ recognized user
    Gate 2: all stages ✓
    Gate 3: modules up to level 2
    Result: full pipeline, but ADMIN-only modules still skipped

ADMIN (3)
    Gate 1: ✓ recognized user
    Gate 2: all stages ✓
    Gate 3: all modules ✓
    Result: everything
```

**Config change security:**
```
User prompt
    ▼
[Input provider] Identify user via channel
    ▼
[GATE 1] Transporter — is user known?
    ▼
[classifier] Intent classification
    ▼
[GATE 2] Stage floor — user clears responder + finalizer?
    ▼
[responder] LLM parse → structured change request object only
    ▼
[finalizer] config_manager:
    [GATE 3] Module permission — is user ADMIN?
    Validate structure — whitelisted fields only, type checks, range checks
    Atomic write — all fields validated before any write, no partial writes
```

Whitelist validation throughout. Blacklist validation never used.

---

## System Tools

System tools provide runtime introspection and control through the chat interface. They are part of core (not modules) because they inspect core internals. All gated to ADMIN.

**Invocation:** `lanes` prefix. The classifier sets `ctx.intent = "system_tool"`. Tool runner fires in `processor` stage, sets `ctx.final_text`, responder stage skips.

### BaseTool Contract

```python
class BaseTool:
    name: str
    description: str
    min_security: SecurityLevel
    uses_llm: bool = False

    async def run(self, ctx: Context, args: list[str]) -> str | None:
        """Return string to short-circuit, or None to continue pipeline."""
        raise NotImplementedError
```

### Built-in Tools

```
lanes help            Lists available tools (respects security level)
lanes pipeline [intent]  Shows resolved module execution order for an intent
lanes slots           Dumps slot state (user, tokens, flags, idle, ephemeral)
lanes channels        Shows active channels and their provider pairs
lanes security        Shows stage floors and module permission levels
lanes debug [on|off]  Toggles debug trace on response footer
lanes trace [last|N]  Shows pipeline trace from in-memory ring buffer
lanes config          Read-only core config dump (never exposes secrets)
lanes summary [user]  Shows current summary.txt content
lanes history [user] [N]  Shows last N turns from history.db
lanes health          System health (uptime, GPU, slots, channels, disk)
lanes wipe [user|slot]   Force-wipe KV cache (confirmation required)
lanes reload          Hot-reload module registry (not config or providers)
lanes test [module]   Run module self-test if defined
```

**Example outputs:**

```
> lanes channels
voice: input=stt_voiceprint, output=kokoro_tts
chat:  input=text_only, output=text_only

> lanes security
stage floors:
  classifier: GUEST (0)
  enricher:   USER (1)
  processor:  USER (1)
  responder:  GUEST (0)
  finalizer:  TRUSTED (2)
module levels:
  intent_classifier: GUEST (0)
  rag:               USER (1)
  rag_processor:     USER (1)
  ha_bridge:         USER (1)
  config_manager:    ADMIN (3)

> lanes health
uptime:    4d 12h
llama.cpp: running (pid 4821)
vram:      12.1GB / 16GB (3.9GB free)
slots:     2/5 active
channels:  voice, chat
transcript: enabled
disk:      1.2GB / 50GB (users/)

> lanes debug on → "What time is it?"
"It's 3:47 PM.
---
[debug] channel: voice (45ms) → user=dad, confidence=0.94
[debug] classifier: intent_classifier (2ms) → intent=general_chat
[debug] responder: llm_respond (340ms) → slot=0, tokens=1203
[debug] output: kokoro_tts (180ms)
[debug] total: 567ms"
```

---

## Data Flow

### General Request Flow

```
Client (raw audio, text, or multimodal)
        │
        │  POST /channel/{channel_name}
        ▼
core/transport.py (transporter)
  [Route to channel provider pair]
        │
        ▼
core/providers/input/{channel.input}
  [Process raw input → ProcessedInput(user_id, message, metadata, attachments)]
        │
        ▼
core/transport.py
  [GATE 1: Identity — is user known?]
  [Broadcast user message to transcript SSE (if enabled)]
        │
        ▼
main.py
  [build Context with user, message, input_metadata, attachments, _llm]
        │
        ▼
service/dispatcher.py
  [walk PIPELINE stages in order]
   ┌────────────────────────────────────────────────────────────┐
   │  For each stage:                                           │
   │    [GATE 2] Does user clear the stage security floor?      │
   │    No → skip entire stage                                  │
   │    Yes → for each module registered to this stage:         │
   │      - Does ctx.intent match module's intents?             │
   │      - [GATE 3] Does user clear module's security level?   │
   │      - Run module.handle(ctx) → ctx                        │
   └────────────────────────────────────────────────────────────┘
        │
        ▼
core/transport.py
  [Broadcast assistant response to transcript SSE (if enabled)]
        │
        ▼
core/providers/output/{channel.output}
  [Process text → OutputResult(audio bytes or text)]
        │
        ▼
Client receives response
(Transcript subscribers receive both sides in real time)
```

### Example: Voice Chat ("Tell me a joke")

```
[POST /channel/voice]
[input]      → stt_voiceprint → user=dad, message="Tell me a joke"
[GATE 1]     → user known
[transcript] → broadcast: {role: "user", channel: "voice"}
[classifier] → intent_classifier → ctx.intent = "general_chat"
[enricher]   → (none)
[processor]  → (none)
[responder]  → llm_respond → calls slot 0 → ctx.final_text = joke
[finalizer]  → (none)
[transcript] → broadcast: {role: "assistant", channel: "voice"}
[output]     → TTS provider → audio bytes → speaker
```

### Example: Text Chat (same user, same slot, different channel)

```
[POST /channel/chat]
[input]      → text_only → user=dad, message="That was funny, tell another"
[GATE 1]     → user known
[transcript] → broadcast: {role: "user", channel: "chat"}
[classifier] → intent_classifier → ctx.intent = "general_chat"
[responder]  → llm_respond → calls slot 0 (sees previous joke in history)
                              ctx.final_text = another joke
[transcript] → broadcast: {role: "assistant", channel: "chat"}
[output]     → text_only → JSON text → chatbox
```

### Example: Image Query (future vision channel)

```
[POST /channel/vision]
[input]      → multimodal → user=dad, message="What does this say?"
                             ctx.attachments = [Attachment(type="image", ...)]
[GATE 1]     → user known
[classifier] → intent_classifier → ctx.intent = "vision_query"
[responder]  → llm_respond → build_content() formats multimodal blocks
                              calls slot 0 with text + image content
                              Qwen3-VL processes image natively
                              ctx.final_text = "That's a nutrition label showing..."
[output]     → text_only → JSON text → chatbox
```

### Example: Health Query with Utility Slot

```
[POST /channel/voice]
[input]      → stt_voiceprint → user=dad
[classifier] → intent_classifier → ctx.intent = "health_query"
[enricher]   → rag → ctx.prompt_extras = [47 weight entries]
[processor]  → rag_processor → await ctx.call_utility("Summarize.", raw_entries)
                                utility available → slot 4, wipes after
                                utility unavailable → fallback to slot 0, sets flag
                                ctx.prompt_extras = [compressed summary]
[responder]  → llm_respond → slot 0, "You're down 13 pounds — nice work."
[output]     → TTS provider → audio bytes
```

### Example: Device Control ("Turn on the kitchen lights")

```
[POST /channel/voice]
[input]      → stt_voiceprint → user=dad
[classifier] → intent_classifier → ctx.intent = "device_control"
[processor]  → ha_bridge → converts intent to HA API call, executes it,
                            sets ctx.final_text = "Lights on."
[responder]  → llm_respond → final_text set, skips LLM call
[output]     → TTS provider → audio: "Lights on."
```

### Example: System Tool ("lanes slots")

```
[POST /channel/voice]
[input]      → stt_voiceprint → user=dad, message="lanes slots"
[classifier] → intent_classifier → ctx.intent = "system_tool"
[processor]  → system_tools → ctx.final_text = slot state table
[responder]  → llm_respond → final_text set, skips
[output]     → TTS provider → audio bytes
```

### Example: Config Change

```
[POST /channel/voice]
[input]      → stt_voiceprint → user=dad
[classifier] → intent_classifier → ctx.intent = "config_change"
[responder]  → llm_respond → LLM parses → structured change request
[finalizer]  → config_manager → [GATE 3] ADMIN check → whitelist validate → atomic write
                                ctx.final_text = "Updated THRESHOLD_WARN to 8000."
[output]     → TTS provider → audio bytes
```

---

## Module Contract

Every module is a self-contained drop-in folder discovered automatically at startup.

**Required structure:**
```
modules/{module_name}/
├── module.yaml          # Manifest — name, stage, intents, security, enabled
├── __init__.py          # Must expose handle() at package level
└── {module_name}.py     # Runtime logic
```

**Optional files:**
```
modules/{module_name}/
├── config.yaml          # Module-specific settings (read via load_addon_config)
└── _installers/         # Optional convenience scripts
    ├── install.py       # Interactive: prompts for config.yaml values
    └── uninstall.py     # Deletes the folder
```

### Module Manifest

```yaml
# modules/intent_classifier/module.yaml
name: intent_classifier
enabled: true
stage: classifier
intents: ["*"]
security_level: 0
order: 0                    # Optional: explicit ordering within a stage
description: "Semantic intent classification"
```

The scanner reads the manifest, imports the module if `enabled: true`, and attaches `security_level` to the registered entry for Gate 3 enforcement.

### Module Self-Contained Config

```yaml
# modules/rag/config.yaml
chromadb_url: "http://localhost:8000"
collection_name: "brain_docs"
max_results: 5
embedding_model: "nomic-embed"
```

```python
# modules/rag/rag.py
from core.config_loader import load_addon_config
cfg = load_addon_config(__file__)
# cfg.chromadb_url, cfg.max_results, etc.
```

### Required Interface

```python
from core.registry import register

MODULE_NAME = "{name}"

@register(MODULE_NAME, stage="{stage_name}", intents=["{intent1}", "{intent2}"])
async def handle(ctx: Context) -> Context:
    return ctx
```

**Optional self-test:**
```python
async def test() -> str:
    """Run by 'lanes test {module_name}'. Return PASS/FAIL string."""
```

### Module Rules

1. Modules must never import from other modules.
2. Modules must never import from `service/`.
3. Modules must never import `core/llm.py` directly — use `ctx.call_slot()` or `ctx.call_utility()`.
4. Modules must never write to config.py at runtime.
5. Modules must never check permissions — the dispatcher handles this via Gate 2 and Gate 3.
6. Modules must never block the event loop — use asyncio for all IO.
7. Modules must return `ctx` even if they do nothing.
8. MODULE_NAME must exactly match the folder name.
9. Modules communicate through `ctx.metadata` and `ctx.prompt_extras` only.
10. Modules should use `ctx.call_utility()` for processing work, not hardcoded slot numbers.
11. Modules may read `ctx.attachments` but should not assume they are present.
12. Module settings belong in the module's own `config.yaml`, never in core `config.py`.

---

## Module Installer / Uninstaller Contract

Installers are optional convenience scripts in `_installers/`. They are never imported by the runtime — they help users interactively configure the module's `config.yaml` and `module.yaml`. The system works without them (manual yaml editing is always an option).

**Installer should:**
1. Print banner with module name and description
2. Prompt for module-specific settings (URLs, thresholds, etc.)
3. Write `config.yaml` with the collected values
4. Set `enabled: true` in `module.yaml`
5. Print confirmation

**Uninstaller should:**
1. Warn if module is currently enabled
2. Prompt confirmation
3. Delete the module folder
4. Print confirmation

**Installers never touch config.py, `modules/__init__.py`, or any file outside their own folder.**

**Install/uninstall simplified:**
- Install: drop the folder in `modules/`, run `_installers/install.py` (or manually edit yaml), restart or `lanes reload`
- Uninstall: delete the folder, restart or `lanes reload`

---

## Extensibility Summary

| Layer | What changes | Install method | Restart needed | Uninstall |
|---|---|---|---|---|
| Module | Pipeline behavior | Drop folder in `modules/` | No (`lanes reload`) | Delete folder |
| Provider | Core I/O capability | Drop folder in `core/providers/` | Yes | Delete folder + update channel config |
| Channel | New interface | Add entry to `config.py` CHANNELS | Yes | Remove entry |
| Pipeline stage | Execution structure | Add to `config.py` PIPELINE list | Yes | Remove entry |
| System tool | Admin commands | Drop file in `core/tools/builtins/` | Yes | Delete file |

**Complete extension point map:**

| Feature | Extension point | Core changes |
|---|---|---|
| New TTS engine | Drop provider in `core/providers/output/` | None |
| New STT engine | Drop provider in `core/providers/input/` | None |
| Voiceprint identification | Drop provider in `core/providers/input/` | None |
| Chat interface | Add channel entry in config | None |
| Image/vision input | Add `multimodal` provider + vision channel | None |
| Live transcript | Enable via `ENABLE_TRANSCRIPT_SSE` flag | None |
| Sensor data endpoint | Provider with `register_routes()` | None |
| RAG retrieval | Module in `modules/`, register to `enricher` | None |
| Device control | Module in `modules/`, register to `processor` | None |
| New admin command | Drop file in `core/tools/builtins/` | None |
| New pipeline stage | One line in `config.PIPELINE` | None |
| Alternative vector store | New provider capability type | New base class |

---

## Rules Summary for Contributors

1. `main.py` never grows. Any logic added to main belongs in a subsystem.
2. `core/` never imports from `modules/`. The boundary is absolute.
3. Modules never import from other modules, from `service/`, or from `core/llm.py`.
4. Modules access LLM capabilities through `ctx.call_slot()` and `ctx.call_utility()` only.
5. Modules should prefer `ctx.call_utility()` over `ctx.call_slot()` for processing work.
6. Security is enforced at three gates only. Modules never check their own permissions.
7. Config.py is core-only. Addons never write to it. Module/provider settings live in their own `config.yaml`.
8. Every module exposes exactly one public function: `handle(ctx) -> ctx`.
9. All IO inside modules must be async. No blocking calls on the event loop.
10. Token tracking uses `usage.total_tokens` from the latest LLM response only. Never accumulate.
11. Whitelist validation only. Never blacklist.
12. The LLM is never a security decision maker.
13. The Context object is the only API surface between components. Modules read and write `ctx`.
14. Modules communicate through `ctx.metadata` and `ctx.prompt_extras`, never shared state.
15. Ephemeral slots are wiped after every `call_slot()` response. No exceptions.
16. System tools live in `core/tools/`, not `modules/`, because they inspect core internals.
17. Pipeline stages are defined in `config.PIPELINE`. Adding a stage is a config change.
18. Providers are self-contained folders with `provider.yaml` manifests and optional `config.yaml`.
19. Channels are defined in `config.CHANNELS`. Adding a channel is a config change.
20. All channels share the same pipeline, security gates, and user slots.
21. The transcript SSE stream is optional (`ENABLE_TRANSCRIPT_SSE`) and read-only. It never alters the pipeline.
22. Attachments on Context are optional. Modules must not assume they are present.
23. Gate 2 (stage floor) always runs before Gate 3 (module permission). A module cannot lower the floor.
24. Drop-in discovery: modules and providers are found by scanning for manifests. No manual registration required.
