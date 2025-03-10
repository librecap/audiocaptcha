#!/usr/bin/env python3

import os
import pickle
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import torch
import numpy as np
from diffusers import AudioLDMPipeline
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment


def generate_animal_captchas(
    clips_per_animal=20,
    clip_duration=3,
    batch_size=2,
    num_workers=None,
):
    """
    Generate audio captchas of different animal sounds.

    Args:
        clips_per_animal: Number of clips to generate per animal
        clip_duration: Duration of each clip in seconds
        batch_size: Number of audio samples to generate in parallel on GPU
        num_workers: Number of CPU workers (defaults to number of CPU cores)
    """
    start_time = time.time()
    output_path = os.path.join(os.path.dirname(__file__), "animals.pkl")

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Using {num_workers} CPU workers and batch_size={batch_size}")

    animal_prompts = {
        "dog": [
            "Realistic dog barking sound, clear and loud barking of a large dog",
            "Clear audio recording of multiple dogs barking aggressively",
            "High-quality sound of a dog barking and growling, very realistic",
            "Professional audio recording of a dog barking repeatedly",
            "Crystal clear sound effect of a large dog's deep bark",
        ],
        "cat": [
            "Realistic cat meowing loudly, clear isolated cat sound effect",
            "High-quality recording of a cat's meow, isolated animal sound",
            "Professional sound effect of a cat meowing repeatedly",
            "Clear audio of a domestic cat making meowing sounds",
            "Studio-quality recording of cat meows, isolated feline vocalization",
        ],
        "cow": [
            "Clear audio recording of a cow mooing loudly in a field",
            "Realistic cow moo sound effect, high-quality farm animal recording",
            "Professional sound effect of a cow mooing several times",
            "Studio recording of cow mooing sounds, isolated animal vocalization",
            "High-definition audio of a dairy cow mooing, clear farmyard sound",
        ],
        "horse": [
            "Realistic horse neighing loudly, clear equine vocalization",
            "Professional recording of a horse whinnying and snorting",
            "High-quality sound effect of horse neighs, isolated animal sound",
            "Clear audio of multiple horse neighs, equestrian sound effects",
            "Studio recording of a horse's loud whinny, distinct equine sound",
        ],
        "sheep": [
            "Clear recording of sheep baaing loudly in a field",
            "Realistic sheep bleating sound effect, multiple sheep sounds",
            "Professional audio of sheep baaing, isolated farmyard sound",
            "High-quality recording of a flock of sheep making loud bleating noises",
            "Studio-quality sheep baa sound effect, distinctive farm animal noise",
        ],
        "lion": [
            "Powerful lion roaring loudly, clear big cat vocalization",
            "Realistic recording of a lion's deep roar, high-definition audio",
            "Professional sound effect of a male lion roaring aggressively",
            "Clear isolated audio of a lion's mighty roar, wildlife sound",
            "Studio-quality recording of lion roars, distinctive powerful sound",
        ],
        "elephant": [
            "Realistic elephant trumpeting loudly, clear isolated sound",
            "Professional recording of elephant trumpeting calls",
            "High-quality sound effect of an elephant's powerful trumpet",
            "Clear audio of an elephant making loud trumpeting sounds",
            "Studio recording of elephant trumpeting, distinctive wildlife sound",
        ],
        "wolf": [
            "Realistic wolf howling loudly, clear isolated wildlife sound",
            "Professional recording of wolves howling at night",
            "High-quality sound effect of a wolf's haunting howl",
            "Clear audio of wolf howls, distinctive wilderness sound",
            "Studio-quality recording of wolf howling, isolated animal vocalization",
        ],
        "monkey": [
            "Loud monkey screeching and chattering, clear primate vocalization",
            "Realistic recording of monkeys calling in the jungle",
            "Professional sound effect of monkey screams and calls",
            "High-quality audio of monkey vocalizations, distinctive primate sounds",
            "Studio recording of monkey chattering, isolated wildlife sound",
        ],
        "owl": [
            "Clear recording of an owl hooting loudly at night",
            "Realistic owl hoot sound effect, distinctive bird call",
            "Professional audio of owl hooting repeatedly, nocturnal bird sound",
            "High-quality sound of a great horned owl's deep hoots",
            "Studio-quality owl hooting sound effect, isolated bird vocalization",
        ],
    }

    audio_data = {
        "type": "audio",
        "keys": {animal: [] for animal in animal_prompts},
        "format": "wav",
    }

    if os.path.exists(output_path):
        print(f"Loading existing data from {output_path}")
        with open(output_path, "rb") as f:
            try:
                existing_data = pickle.load(f)

                if "keys" in existing_data and isinstance(existing_data["keys"], dict):
                    for animal, sounds in existing_data["keys"].items():
                        if animal in audio_data["keys"]:
                            audio_data["keys"][animal] = sounds

                animal_counts = {
                    animal: len(clips) for animal, clips in audio_data["keys"].items()
                }
                print(f"Found existing clips: {animal_counts}")
            except Exception as e:
                print(f"Error loading existing data: {e}. Starting fresh.")

    print("Loading AudioLDM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    repo_id = "cvssp/audioldm-l-full"
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    if device == "cuda":
        pipe.enable_model_cpu_offload()

    save_lock = threading.Lock()

    def save_progress():
        with save_lock:
            with open(output_path, "wb") as f:
                pickle.dump(audio_data, f)

    def process_audio(audio_np, sample_rate=16000):
        """Process audio numpy array into WAV bytes"""
        target_length = clip_duration * sample_rate
        if len(audio_np) > target_length:
            audio_np = audio_np[:target_length]
        elif len(audio_np) < target_length:
            pad_length = target_length - len(audio_np)
            audio_np = np.pad(audio_np, (0, pad_length), "constant")

        if audio_np.max() <= 1.0 and audio_np.min() >= -1.0:
            audio_np = (audio_np * 32767).astype(np.int16)

        wav_io = io.BytesIO()
        write_wav(wav_io, sample_rate, audio_np)
        wav_io.seek(0)

        audio_segment = AudioSegment.from_wav(wav_io)

        target_dbfs = -15.0
        change_in_dbfs = target_dbfs - audio_segment.dBFS
        normalized_audio = audio_segment.apply_gain(change_in_dbfs)

        normalized_wav_io = io.BytesIO()
        normalized_audio.export(normalized_wav_io, format="wav")
        normalized_wav_io.seek(0)

        return normalized_wav_io.getvalue()

    def generate_batch(prompts, batch_indices):
        """Generate a batch of animal sounds and process them"""
        batch_prompts = [prompts[i % len(prompts)] for i in batch_indices]

        with torch.no_grad():
            audios = pipe(
                batch_prompts,
                num_inference_steps=200,
                audio_length_in_s=clip_duration,
                guidance_scale=3.5,
            ).audios

        processed_audios = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for audio in audios:
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.squeeze().cpu().numpy()
                else:
                    audio_np = audio
                futures.append(executor.submit(process_audio, audio_np))

            for future in as_completed(futures):
                processed_audios.append(future.result())

        return processed_audios

    for animal, prompts in animal_prompts.items():
        current_count = len(audio_data["keys"][animal])
        if current_count >= clips_per_animal:
            print(
                f"Already have {current_count}/{clips_per_animal} clips for {animal}, skipping."
            )
            continue

        to_generate = clips_per_animal - current_count
        print(f"Generating {to_generate} clips for {animal}...")

        batch_indices = [
            list(range(i, min(i + batch_size, to_generate)))
            for i in range(0, to_generate, batch_size)
        ]

        for batch_idx, indices in enumerate(batch_indices):
            generated_audios = generate_batch(prompts, indices)

            for audio_data_compressed in generated_audios:
                audio_data["keys"][animal].append(audio_data_compressed)

            if (batch_idx + 1) % 2 == 0 or batch_idx == len(batch_indices) - 1:
                save_progress()
                print(
                    f"Saved progress: {min((batch_idx+1)*batch_size, to_generate)}/"
                    f"{to_generate} {animal} clips"
                )

    save_progress()

    total_time = time.time() - start_time
    print(f"Animal sound captchas generation complete in {total_time:.2f} seconds!")
    print(
        f"Average time per clip: "
        f"{total_time / sum(len(clips) for clips in audio_data['keys'].values()):.2f} seconds"
    )

    total_bytes = sum(
        sum(len(clip) for clip in clips) for clips in audio_data["keys"].values()
    )
    total_clips = sum(len(clips) for clips in audio_data["keys"].values())

    for animal, clips in audio_data["keys"].items():
        print(f"{animal}: {len(clips)} clips")

    print(f"Total storage: {total_bytes/1024/1024:.2f} MB for {total_clips} clips")
    print(f"Average size per clip: {total_bytes/total_clips/1024:.2f} KB")
    print(f"Data saved to {output_path}")

    return audio_data


def get_batch_size():
    """Get the optimal batch size for the current GPU"""
    if not torch.cuda.is_available():
        return 2

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_mem > 16:
        return 5

    if gpu_mem > 8:
        return 4

    return 2


if __name__ == "__main__":
    BATCH_SIZE = get_batch_size()
    generate_animal_captchas(
        clips_per_animal=20, clip_duration=3, batch_size=BATCH_SIZE
    )
