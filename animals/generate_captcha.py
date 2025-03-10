#!/usr/bin/env python3

import os
import io
import pickle
import random
from functools import lru_cache
from pydub import AudioSegment


audio_cache = {}


def bytes_to_audio_segment(audio_bytes):
    """Convert bytes directly to AudioSegment without temp files."""
    wav_io = io.BytesIO(audio_bytes)
    return AudioSegment.from_wav(wav_io)


@lru_cache(maxsize=32)
def create_silence(duration_ms):
    """Create a silent AudioSegment with caching."""
    return AudioSegment.silent(duration=duration_ms)


def change_volume(audio_segment, target_dbfs):
    """Normalize the volume of an AudioSegment to target dBFS."""
    change_in_dbfs = target_dbfs - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dbfs)


def mix_audio(audio1, audio2, position_ms=0):
    """Mix two AudioSegments safely handling potential format mismatches."""
    try:
        return audio1.overlay(audio2, position=position_ms)
    except Exception:
        if audio1.frame_rate != audio2.frame_rate:
            audio2 = audio2.set_frame_rate(audio1.frame_rate)
        if audio1.channels != audio2.channels:
            audio2 = audio2.set_channels(audio1.channels)
        if audio1.sample_width != audio2.sample_width:
            audio2 = audio2.set_sample_width(audio1.sample_width)
        return audio1.overlay(audio2, position=position_ms)


def batch_mix_audio(base_audio, segments_with_positions):
    """
    More efficient way to mix multiple audio segments with their positions.

    Args:
        base_audio: Base AudioSegment
        segments_with_positions: List of tuples (segment, position_ms)

    Returns:
        Mixed AudioSegment
    """
    result = base_audio

    segments_with_positions.sort(key=lambda x: x[1])

    batch_size = 10
    for i in range(0, len(segments_with_positions), batch_size):
        batch = segments_with_positions[i : i + batch_size]

        for segment, position in batch:
            result = mix_audio(result, segment, position)

    return result


def generate_combined_captcha(language="en", output_format="mp3", count=5):
    """
    Generate a combined audio captcha that includes number samples (1-5)
    and music samples with one different from the others.

    Args:
        language: Language to use for the captcha
            (en, es, fr, de, it, pt, zh-CN, ja, ko, ru, ar, hi)
        output_format: Format to save the audio file (mp3, ogg, wav)
        count: Number of clips to include in the captcha (2-9)

    Returns:
        Dictionary with challenge information
    """

    if count < 2:
        raise ValueError("Count must be at least 2.")

    if count > 9:
        raise ValueError("Count must be less than 9.")

    numbers_path = os.path.join(os.path.dirname(__file__), "numbers.pkl")
    music_path = os.path.join(os.path.dirname(__file__), "animals.pkl")

    if not os.path.exists(numbers_path):
        raise FileNotFoundError(
            f"Number audio file not found: {numbers_path}. Run generate_audio_captchas.py first."
        )

    if not os.path.exists(music_path):
        raise FileNotFoundError(
            f"Music samples file not found: {music_path}. Run generate_music_captchas.py first."
        )

    with open(numbers_path, "rb") as f:
        numbers_data = pickle.load(f)

    with open(music_path, "rb") as f:
        music_data = pickle.load(f)

    number_samples = []
    for num in range(1, count + 1):
        num_str = str(num)
        if not language in numbers_data["keys"][num_str]:
            print(f"Language {language} not found for number {num_str}.")
            continue

        number_samples.append(numbers_data["keys"][num_str][language])

    if len(number_samples) < count:
        raise ValueError(f"Not enough number samples found for {language}.")

    majority_category = random.choice(list(music_data["keys"].keys()))
    odd_category = None
    while odd_category is None or majority_category == odd_category:
        odd_category = random.choice(list(music_data["keys"].keys()))

    majority_samples = random.sample(music_data["keys"][majority_category], count - 1)
    odd_sample = random.choice(music_data["keys"][odd_category])

    odd_position = random.randint(0, count - 1)

    combined_audio = AudioSegment.empty()
    answer_sequence = []

    segments_with_positions = []
    current_position = 1000

    for i in range(count):
        number_audio = bytes_to_audio_segment(number_samples[i])

        number_audio = change_volume(number_audio, -20.0)

        segments_with_positions.append((number_audio, current_position))
        current_position += len(number_audio) + 500

        if i == odd_position:
            music_sample = odd_sample
            answer_sequence.append(odd_category)
        else:
            music_sample = majority_samples[i if i < odd_position else i - 1]
            answer_sequence.append(majority_category)

        music_audio = bytes_to_audio_segment(music_sample)

        if len(music_audio) > 5000:
            music_audio = music_audio[:5000]

        music_audio = change_volume(music_audio, -18.0)

        segments_with_positions.append((music_audio, current_position))
        current_position += len(music_audio) + 1000

    base_audio = create_silence(current_position)

    combined_audio = batch_mix_audio(base_audio, segments_with_positions)

    combined_audio = change_volume(combined_audio, -16.0)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"captcha_challenge.{output_format}")
    combined_audio.export(output_file, format=output_format)

    challenge = {
        "type": "audio_captcha",
        "file": output_file,
        "odd_position": odd_position,
        "odd_category": odd_category,
        "majority_category": majority_category,
        "sequence": answer_sequence,
        "instruction": (
            "Listen to each audio clip and identify "
            "which one (1-5) does NOT belong with the others."
        ),
    }

    print(f"Captcha generated: {output_file}")
    print(f"The odd one out is clip #{odd_position + 1} ({odd_category})")

    return challenge


if __name__ == "__main__":
    import argparse

    try:
        parser = argparse.ArgumentParser(
            description="Generate a combined audio captcha"
        )
        parser.add_argument(
            "--language", default="en", help="Language code (default: en)"
        )
        parser.add_argument(
            "--format",
            choices=["mp3", "wav", "ogg"],
            default="mp3",
            help="Audio format (default: mp3)",
        )
        parser.add_argument(
            "--count", type=int, default=5, help="Number of clips (default: 5)"
        )
        args = parser.parse_args()

        generate_combined_captcha(args.language, args.format, args.count)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("There is no audio samples. Run generate_audio_captchas.py first.")
    except ValueError as e:
        print(f"Error: {e}")
