import outlines
from outlines import models, generate
from outlines.models.transformers import Transformers, TransformerTokenizer # Import the classes

import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# --- Configuration ---
AUDIO_FILE = "audio.wav"
MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "en"
TASK = "transcribe"

# --- 1. Load Model and Processor Manually ---
# Ensure you have `librosa` and `soundfile` installed:
# pip install librosa soundfile
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model_hf = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)

# --- 2. Prepare Whisper Inputs ---
# Load and process audio
try:
    audio_data, sampling_rate = librosa.load(AUDIO_FILE, sr=16000)
except FileNotFoundError:
    print(f"Error: Audio file '{AUDIO_FILE}' not found.")
    # Create a dummy silent audio array if file not found,
    # otherwise WhisperProcessor will error.
    # You should replace this with proper error handling or file creation.
    import numpy as np
    audio_data = np.zeros(16000 * 5, dtype=np.float32) # 5 seconds of silence
    sampling_rate = 16000
    print(f"Warning: Using 5 seconds of silent audio as placeholder.")


input_features = processor(
    audio_data, sampling_rate=sampling_rate, return_tensors="pt"
).input_features.to(device)

# Prepare decoder inputs (control tokens)
# <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>
# We need the processor to force these tokens at the beginning.
# The most reliable way is using `forced_decoder_ids` within the generate call.
# Let's get the specific token IDs for this.
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language=LANGUAGE, task=TASK, no_timestamps=True
)

# --- 3. Instantiate Outlines Wrapper Manually ---
# We bypass `outlines.models.transformers(MODEL_ID)` because it assumes AutoModelForCausalLM
# We pass our already loaded Whisper model and processor instead.
#outlines_tokenizer = TransformerTokenizer(processor.tokenizer) # Use the tokenizer part of the processor
outlines_model = Transformers(model_hf, processor.tokenizer)

# --- 4. Define the Monkey-Patch Function ---
# Store the original method for potential reference (optional)
original_generate_output_seq = outlines_model._generate_output_seq

def patched_generate_output_seq(
    self, prompts, inputs, generation_config, **generation_kwargs
):
    """
    Patched version of _generate_output_seq for Whisper.

    Ignores text `prompts` and `inputs` (derived from text).
    Uses pre-calculated `input_features` from audio.
    Injects `forced_decoder_ids` for Whisper control tokens.
    Passes through Outlines' `generation_config` and `logits_processor`.
    """
    print("--- Using Patched _generate_output_seq ---") # For confirmation

    # We ignore the text-based `inputs` dictionary.
    # We use our pre-computed audio `input_features`.
    whisper_inputs = {"input_features": input_features}

    # Add forced_decoder_ids to the generation config or kwargs
    # Note: generation_config is mutable, modifying it is okay here.
    # Alternatively, add to generation_kwargs if preferred.
    generation_config.forced_decoder_ids = forced_decoder_ids

    # Call the original Hugging Face model's generate method
    output_ids = self.model.generate(
        **whisper_inputs, # Use audio features
        generation_config=generation_config, # Pass outlines config (max_tokens, etc.)
        logits_processor=generation_kwargs.get("logits_processor"), # Pass outlines processor
        # Pass other relevant kwargs if outlines adds more in the future
        # For safety, let's pass the common ones if they exist in kwargs
        # Although generation_config should handle most now.
        # **{k: v for k, v in generation_kwargs.items() if k not in ['logits_processor', 'tokenizer']}
    )

    # --- Output processing (copied from original outlines method) ---
    # Whisper is encoder-decoder, so output_ids are only the generated ones.
    # No need to slice off input_ids like for decoder-only models.
    generated_ids = output_ids

    # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
    num_samples = generation_config.num_return_sequences or 1
    if num_samples > 1 and isinstance(prompts, list):
        # This part might need adjustment if you use batching with Whisper
        # For a single audio file, batch_size is effectively 1
        batch_size = 1 # Assuming single audio input for now
        num_return_sequences = generation_config.num_return_sequences or 1
        generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)

    return generated_ids

# --- 5. Apply the Monkey Patch ---
# Replace the method on our specific instance
outlines_model._generate_output_seq = patched_generate_output_seq.__get__(outlines_model, Transformers)

# --- 6. Run Outlines Generator ---
# The `prompt` here is basically ignored by our patch, but Outlines needs it.
# The regex will still be applied during generation via the logits_processor.
generator = generate.regex(
    outlines_model, # Use our manually created and patched model instance
    r" - [A-Z].*", # Your regex here (.* means anything)
    #r" >> [A-Z].*"
    #r" > [A-Z].*"
    # sampler=..., # Add sampler if needed
)

# The prompt text is not used for Whisper input, but required by the API
dummy_prompt = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
print("Starting Outlines generation...")
answer = generator(dummy_prompt, max_tokens=150) # Increase max_tokens for transcription

print("\nTranscription:")
print(answer)

# --- Optional: Restore original method if needed ---
# outlines_model._generate_output_seq = original_generate_output_seq.__get__(outlines_model, Transformers)
