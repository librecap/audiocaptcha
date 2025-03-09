#!/usr/bin/env python3
import os
import io
import math
import pickle
import random
import secrets
from functools import lru_cache
import numpy as np
from scipy.io.wavfile import write as write_wav
from pydub import AudioSegment

WAVE_SAMPLE_RATE = 44100  # Hz

audio_cache = {}


def numpy_to_audio_segment(samples, sample_rate=44100):
    """Convert numpy array directly to AudioSegment without temporary files."""
    samples = samples.astype(np.int16)

    wav_io = io.BytesIO()
    write_wav(wav_io, sample_rate, samples)
    wav_io.seek(0)

    return AudioSegment.from_wav(wav_io)


@lru_cache(maxsize=128)
def generate_sine_wave(freq, duration_ms, sample_rate=44100):
    """Generate a sine wave at the specified frequency and duration."""
    cache_key = f"sine_{freq}_{duration_ms}_{sample_rate}"
    if cache_key in audio_cache:
        return audio_cache[cache_key]

    num_samples = int(sample_rate * duration_ms / 1000.0)
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

    beep_segment = numpy_to_audio_segment(samples, sample_rate)

    audio_cache[cache_key] = beep_segment
    return beep_segment


def change_speed(audio_segment, speed=1.0):
    """Change the speed of an AudioSegment."""
    if speed == 1.0:
        return audio_segment

    return audio_segment._spawn(
        audio_segment.raw_data,
        overrides={"frame_rate": int(audio_segment.frame_rate * speed)},
    ).set_frame_rate(audio_segment.frame_rate)


def change_volume(audio_segment, level=1.0):
    """Change the volume of an AudioSegment."""
    if level == 1.0:
        return audio_segment

    db_change = 20 * math.log10(level)
    return audio_segment.apply_gain(db_change)


@lru_cache(maxsize=32)
def create_silence(duration_ms):
    """Create a silent AudioSegment."""
    return AudioSegment.silent(duration=duration_ms)


def create_noise(duration_ms, level=0.05, sample_rate=44100):
    """Create white noise."""
    cache_key = f"noise_{duration_ms}_{level}_{sample_rate}"
    if cache_key in audio_cache:
        return audio_cache[cache_key]

    num_samples = int(sample_rate * duration_ms / 1000.0)
    noise_samples = (np.random.uniform(-1, 1, num_samples) * level * 32767).astype(
        np.int16
    )

    noise_segment = numpy_to_audio_segment(noise_samples, sample_rate)

    audio_cache[cache_key] = noise_segment
    return noise_segment


def mix_audio(audio1, audio2, position_ms=0):
    """Mix two AudioSegments."""
    try:
        return audio1.overlay(audio2, position=position_ms)
    except Exception as e:
        print(f"Warning: Audio overlay failed, using safer method. Error: {e}")
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


def reverse_audio(audio_segment):
    """Reverse an AudioSegment."""
    return AudioSegment(
        data=audio_segment.raw_data,
        sample_width=audio_segment.sample_width,
        frame_rate=audio_segment.frame_rate,
        channels=audio_segment.channels,
    ).reverse()


def add_background_noise(audio_segment, noise_level=0.05):
    """Add background noise to an AudioSegment."""
    noise = create_noise(len(audio_segment), level=noise_level)
    return mix_audio(audio_segment, noise)


def generate_human_sound(sound_type="random", duration_ms=None, sample_rate=44100):
    """
    Generate various human-like sounds that mimic speech/noises but aren't actual characters.
    """

    if duration_ms is None:
        duration_ms = secrets.randbelow(400) + 200

    if sound_type == "random":
        sound_types = ["speech", "whisper", "throat", "click", "hum", "breath"]
        sound_type = random.choice(sound_types)

    num_samples = int(sample_rate * duration_ms / 1000.0)
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)

    samples = np.zeros(num_samples, dtype=np.float64)

    if sound_type == "speech":
        base_freq = secrets.randbelow(175) + 80

        frequencies = [
            (base_freq, 1.0),
            (base_freq * 2, 0.5),
            (base_freq * 3, 0.25),
            (secrets.randbelow(500) + 300, 0.7),
            (secrets.randbelow(1000) + 800, 0.5),
            (secrets.randbelow(1000) + 1800, 0.3),
        ]

        for freq, amp in frequencies:
            fm_rate = secrets.randbelow(6) + 2
            fm_depth = secrets.randbelow(10) + 5
            freq_mod = freq + np.sin(2 * np.pi * fm_rate * t) * fm_depth
            am_depth = 0.5 + np.sin(2 * np.pi * (fm_rate + 1) * t) * 0.5

            component = np.sin(2 * np.pi * freq_mod * t) * am_depth * amp
            samples += component

    elif sound_type == "whisper":
        white_noise = np.random.normal(0, 0.3, num_samples)
        samples = white_noise * 0.7

        formants = [
            (secrets.randbelow(500) + 500, 0.5),
            (secrets.randbelow(1000) + 1500, 0.7),
            (secrets.randbelow(1500) + 3000, 0.4),
        ]

        for freq, amp in formants:
            formant = np.sin(2 * np.pi * freq * t) * amp
            samples += formant * white_noise * 0.3

    elif sound_type == "throat":
        base_freq = secrets.randbelow(60) + 40

        for i, amp in enumerate([1.0, 0.7, 0.5, 0.2]):
            freq = base_freq * (i + 1)
            samples += np.sin(2 * np.pi * freq * t) * amp

        burst_pos = int(num_samples * 0.3)
        burst_length = int(sample_rate * 0.03)
        burst = np.random.normal(0, 1, burst_length) * 1.5

        end_pos = min(burst_pos + burst_length, num_samples)
        actual_burst_length = end_pos - burst_pos

        burst_env = np.zeros(actual_burst_length)
        if actual_burst_length > 0:
            if actual_burst_length >= 3:
                third_length = actual_burst_length // 3
                burst_env[:third_length] = np.linspace(0, 1, third_length)
                burst_env[-third_length:] = np.linspace(1, 0, third_length)
                if third_length < actual_burst_length - 2 * third_length:
                    burst_env[third_length:-third_length] = 1.0
            else:
                burst_env = np.linspace(0, 1, actual_burst_length // 2 + 1)
                if actual_burst_length > 1:
                    burst_env = np.concatenate(
                        [
                            burst_env,
                            np.linspace(1, 0, actual_burst_length - len(burst_env)),
                        ]
                    )

            samples[burst_pos:end_pos] += burst[:actual_burst_length] * burst_env

    elif sound_type == "click":
        click_pos = int(num_samples * 0.2)
        click_length = int(sample_rate * 0.02)

        click_freq = secrets.randbelow(2000) + 1000
        click_env = np.zeros(num_samples)
        click_env[click_pos : click_pos + click_length] = np.linspace(
            0, 1, click_length
        ) * np.linspace(1, 0, click_length)

        click_sound = np.sin(2 * np.pi * click_freq * t)
        samples += click_sound * click_env * 2.0

    elif sound_type == "hum":
        notes = [110, 146.83, 196, 220, 261.63, 293.66, 329.63, 392]
        base_freq = random.choice(notes)

        vibrato_rate = 5
        vibrato_depth = base_freq * 0.03
        freq_with_vibrato = base_freq + vibrato_depth * np.sin(
            2 * np.pi * vibrato_rate * t
        )

        samples = np.sin(2 * np.pi * freq_with_vibrato * t)

        for i in range(2, 5):
            harmonic_amp = 1.0 / i
            samples += np.sin(2 * np.pi * i * freq_with_vibrato * t) * harmonic_amp

    else:
        noise = np.random.normal(0, 1, num_samples)
        samples = noise * 0.2

        resonances = [
            (secrets.randbelow(500) + 300, 0.6),
            (secrets.randbelow(800) + 1000, 0.4),
            (secrets.randbelow(1000) + 2000, 0.2),
        ]

        for freq, amp in resonances:
            resonance = np.sin(2 * np.pi * freq * t) * noise * amp
            samples += resonance

    attack = int(num_samples * 0.2)
    decay = int(num_samples * 0.3)

    envelope = np.ones(num_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)

    samples = samples * envelope

    max_amplitude = np.max(np.abs(samples))
    if max_amplitude > 0:
        samples = samples / max_amplitude * 0.9

    return numpy_to_audio_segment(samples * 32767, sample_rate)


def apply_audio_effects(audio_segment, effect_level="medium"):
    """
    Apply various audio effects to make sounds more distorted and varied.
    """
    if effect_level == "low":
        speed_range = (95, 105)
        volume_range = (95, 105)
    elif effect_level == "high":
        speed_range = (70, 130)
        volume_range = (70, 130)
    else:
        speed_range = (80, 120)
        volume_range = (80, 120)

    speed = (
        secrets.randbelow(speed_range[1] - speed_range[0]) + speed_range[0]
    ) / 100.0
    audio_segment = change_speed(audio_segment, speed)

    volume = (
        secrets.randbelow(volume_range[1] - volume_range[0]) + volume_range[0]
    ) / 100.0
    audio_segment = change_volume(audio_segment, volume)

    if secrets.randbelow(100) < 50:
        noise_level = (secrets.randbelow(5) + 1) / 100.0
        audio_segment = add_background_noise(audio_segment, noise_level)

    if effect_level != "low" and secrets.randbelow(100) < 30:
        reverb_level = (secrets.randbelow(20) + 5) / 100.0
        reverb_delay = secrets.randbelow(100) + 50

        silence = AudioSegment.silent(duration=reverb_delay)
        reverb_tail = change_volume(audio_segment, reverb_level)
        delayed_reverb = silence + reverb_tail

        audio_segment = mix_audio(audio_segment, delayed_reverb)

    return audio_segment


def bytes_to_audio_segment(audio_bytes):
    """Convert bytes directly to AudioSegment without temp files."""
    wav_io = io.BytesIO(audio_bytes)
    return AudioSegment.from_wav(wav_io)


def generate_character_challenge(
    character_type="mixed", language="en", count=6, output_format="mp3", obfuscate=True
):
    """
    Generate a character identification captcha where the user must identify
    a sequence of characters (numbers, letters, or mixed).
    """
    characters_path = os.path.join(os.path.dirname(__file__), "characters.pkl")

    if not os.path.exists(characters_path):
        raise FileNotFoundError(
            f"Characters file not found: {characters_path}. Run generate_characters.py first."
        )

    with open(characters_path, "rb") as f:
        character_data = pickle.load(f)

    available_numbers = []
    available_letters = []

    for char, langs in character_data["keys"].items():
        if language in langs:
            if char in "0123456789":
                available_numbers.append(char)
            elif char in "abcdefghijklmnopqrstuvwxyz":
                available_letters.append(char)

    if character_type == "numbers" and len(available_numbers) < count:
        raise ValueError(
            (
                f"Not enough number characters available in {language}. Need {count}, "
                f"found {len(available_numbers)}."
            )
        )

    if character_type == "letters" and len(available_letters) < count:
        raise ValueError(
            (
                f"Not enough letter characters available in {language}. Need {count}, "
                f"found {len(available_letters)}."
            )
        )

    if (
        character_type == "mixed"
        and (len(available_numbers) + len(available_letters)) < count
    ):
        raise ValueError(
            (
                f"Not enough characters available in {language}. Need {count}, found "
                f"{len(available_numbers) + len(available_letters)}."
            )
        )

    selected_chars = []
    if character_type == "numbers":
        selected_chars = random.sample(available_numbers, count)
    elif character_type == "letters":
        selected_chars = random.sample(available_letters, count)
    else:
        num_count = random.randint(1, min(count - 1, len(available_numbers)))
        letter_count = count - num_count

        selected_chars = random.sample(available_numbers, num_count) + random.sample(
            available_letters, letter_count
        )
        random.shuffle(selected_chars)

    combined_audio = AudioSegment.empty()

    intro_silence_duration = secrets.randbelow(200) + 200
    combined_audio += AudioSegment.silent(duration=intro_silence_duration)

    nonsense_sounds = []
    num_nonsense_sounds = 0

    if obfuscate:
        num_nonsense_sounds = secrets.randbelow(count * 2) + count * 3

        sound_types = ["speech", "whisper", "throat", "click", "hum", "breath"]

        for _ in range(num_nonsense_sounds):
            sound_type = random.choice(sound_types)
            human_noise = generate_human_sound(sound_type)

            effect_level = random.choice(["low", "medium", "high"])
            human_noise = apply_audio_effects(human_noise, effect_level)

            target_dbfs = -25.0 + secrets.randbelow(15)
            change_in_dbfs = target_dbfs - human_noise.dBFS
            human_noise = human_noise.apply_gain(change_in_dbfs)

            nonsense_sounds.append(human_noise)

    character_segments = []

    for char in selected_chars:
        char_audio_bytes = character_data["keys"][char][language]

        char_audio = bytes_to_audio_segment(char_audio_bytes)

        target_dbfs = -18.0
        change_in_dbfs = target_dbfs - char_audio.dBFS
        char_audio = char_audio.apply_gain(change_in_dbfs)

        if obfuscate:
            char_audio = apply_audio_effects(char_audio, "low")

        character_segments.append(char_audio)

    char_positions = []

    current_position = 0
    for i, char_audio in enumerate(character_segments):
        if i > 0:
            padding = secrets.randbelow(1000) + 500
            current_position += padding

        char_positions.append((current_position, len(char_audio)))

        current_position += len(char_audio)

    total_duration = current_position + secrets.randbelow(700) + 500

    distractor_placements = []

    if obfuscate and nonsense_sounds:
        for i in range(len(char_positions) - 1):
            start_of_next = char_positions[i + 1][0]
            end_of_current = char_positions[i][0] + char_positions[i][1]
            gap = start_of_next - end_of_current

            num_in_gap = 1
            if gap > 1000:
                num_in_gap = secrets.randbelow(3) + 1

            for _ in range(num_in_gap):
                nonsense_idx = secrets.randbelow(len(nonsense_sounds))
                ns_duration = len(nonsense_sounds[nonsense_idx])

                usable_gap = int(gap * 0.7)
                min_position = end_of_current + int(gap * 0.15)

                if usable_gap > ns_duration:
                    position = min_position + secrets.randbelow(
                        max(1, usable_gap - ns_duration)
                    )
                else:
                    position = min_position

                distractor_placements.append((position, nonsense_idx))

        for pos, dur in char_positions:
            if secrets.randbelow(100) < 70:
                nonsense_idx = secrets.randbelow(len(nonsense_sounds))
                ns_duration = len(nonsense_sounds[nonsense_idx])

                overlap_type = secrets.randbelow(3)

                if overlap_type == 0:
                    overlap_pos = max(0, pos - ns_duration // 2)
                elif overlap_type == 1:
                    overlap_pos = pos + dur - ns_duration // 2
                else:
                    overlap_pos = pos + dur // 2 - ns_duration // 4

                overlap_pos = max(0, overlap_pos)
                distractor_placements.append((overlap_pos, nonsense_idx))

        num_random = min(len(nonsense_sounds) - len(distractor_placements), count * 2)
        for _ in range(num_random):
            nonsense_idx = secrets.randbelow(len(nonsense_sounds))
            ns_duration = len(nonsense_sounds[nonsense_idx])
            position = secrets.randbelow(max(1, total_duration - ns_duration))
            distractor_placements.append((position, nonsense_idx))

    base_audio = AudioSegment.silent(duration=total_duration)

    all_segments = []

    for i, (pos, _) in enumerate(char_positions):
        all_segments.append((character_segments[i], pos))

    for pos, nonsense_idx in distractor_placements:
        if nonsense_idx < len(nonsense_sounds):
            all_segments.append((nonsense_sounds[nonsense_idx], pos))

    base_audio = batch_mix_audio(base_audio, all_segments)

    if obfuscate:
        noise_level = (secrets.randbelow(4) + 1) / 100.0
        base_audio = add_background_noise(base_audio, noise_level)

        if secrets.randbelow(100) < 80:
            num_noise_chars = secrets.randbelow(count) + 2
            noise_char_segments = []

            for _ in range(num_noise_chars):
                try:
                    noise_char = random.choice(selected_chars)
                    char_audio_bytes = character_data["keys"][noise_char][language]

                    noise_char_audio = bytes_to_audio_segment(char_audio_bytes)

                    noise_char_audio = reverse_audio(noise_char_audio)
                    noise_char_audio = change_speed(
                        noise_char_audio, (secrets.randbelow(50) + 70) / 100.0
                    )
                    noise_char_audio = change_volume(
                        noise_char_audio, (secrets.randbelow(15) + 8) / 100.0
                    )

                    max_position = max(0, len(base_audio) - len(noise_char_audio))
                    if max_position > 0:
                        position = secrets.randbelow(max_position)
                        noise_char_segments.append((noise_char_audio, position))
                except Exception as e:
                    print(f"Warning: Could not add noise character: {e}")
                    continue

            if noise_char_segments:
                base_audio = batch_mix_audio(base_audio, noise_char_segments)

    combined_audio = base_audio

    final_target_dbfs = -16.0
    final_change_in_dbfs = final_target_dbfs - combined_audio.dBFS
    combined_audio = combined_audio.apply_gain(final_change_in_dbfs)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"character_challenge.{output_format}")
    combined_audio.export(output_file, format=output_format)

    challenge = {
        "type": "character_captcha",
        "file": output_file,
        "characters": selected_chars,
        "character_type": character_type,
        "language": language,
        "instruction": f"Listen to the sequence of {count} characters and enter them in order.",
        "obfuscated": obfuscate,
    }

    print(f"Character challenge generated: {output_file}")
    print(f"Characters: {''.join(selected_chars)}")
    if obfuscate:
        print("Audio obfuscation: Enabled")

    audio_cache.clear()

    return challenge


if __name__ == "__main__":
    import argparse

    try:
        parser = argparse.ArgumentParser(
            description="Generate a character audio captcha"
        )
        parser.add_argument(
            "--type",
            choices=["numbers", "letters", "mixed"],
            default="mixed",
            help="Type of characters to use (default: mixed)",
        )
        parser.add_argument(
            "--language", default="en", help="Language code (default: en)"
        )
        parser.add_argument(
            "--count", type=int, default=6, help="Number of characters (default: 6)"
        )
        parser.add_argument(
            "--format",
            choices=["mp3", "wav", "ogg"],
            default="mp3",
            help="Audio format (default: mp3)",
        )
        parser.add_argument(
            "--no-obfuscate",
            action="store_true",
            help="Disable audio obfuscation techniques",
        )
        args = parser.parse_args()

        challenge = generate_character_challenge(
            character_type=args.type,
            language=args.language,
            count=args.count,
            output_format=args.format,
            obfuscate=not args.no_obfuscate,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Make sure to run generate_characters.py first to create the character audio samples."
        )
    except ValueError as e:
        print(f"Error: {e}")
