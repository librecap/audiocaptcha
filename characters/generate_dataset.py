#!/usr/bin/env python3

import os
import pickle
import io
from gtts import gTTS
from pydub import AudioSegment


def generate_audio_captchas():
    """Generate audio captchas for numbers 0-9 and letters a-z in multiple languages"""
    output_path = os.path.join(os.path.dirname(__file__), "characters.pkl")

    numbers = {
        "0": {
            "en": "zero",
            "es": "cero",
            "fr": "zéro",
            "de": "null",
            "it": "zero",
            "pt": "zero",
            "zh-CN": "零",
            "ja": "零",
            "ko": "영",
            "ru": "ноль",
            "ar": "صفر",
            "hi": "शून्य",
        },
        "1": {
            "en": "one",
            "es": "uno",
            "fr": "un",
            "de": "eins",
            "it": "uno",
            "pt": "um",
            "zh-CN": "一",
            "ja": "一",
            "ko": "하나",
            "ru": "один",
            "ar": "واحد",
            "hi": "एक",
        },
        "2": {
            "en": "two",
            "es": "dos",
            "fr": "deux",
            "de": "zwei",
            "it": "due",
            "pt": "dois",
            "zh-CN": "二",
            "ja": "二",
            "ko": "둘",
            "ru": "два",
            "ar": "اثنان",
            "hi": "दो",
        },
        "3": {
            "en": "three",
            "es": "tres",
            "fr": "trois",
            "de": "drei",
            "it": "tre",
            "pt": "três",
            "zh-CN": "三",
            "ja": "三",
            "ko": "셋",
            "ru": "три",
            "ar": "ثلاثة",
            "hi": "तीन",
        },
        "4": {
            "en": "four",
            "es": "cuatro",
            "fr": "quatre",
            "de": "vier",
            "it": "quattro",
            "pt": "quatro",
            "zh-CN": "四",
            "ja": "四",
            "ko": "넷",
            "ru": "четыре",
            "ar": "أربعة",
            "hi": "चार",
        },
        "5": {
            "en": "five",
            "es": "cinco",
            "fr": "cinq",
            "de": "fünf",
            "it": "cinque",
            "pt": "cinco",
            "zh-CN": "五",
            "ja": "五",
            "ko": "다섯",
            "ru": "пять",
            "ar": "خمسة",
            "hi": "पांच",
        },
        "6": {
            "en": "six",
            "es": "seis",
            "fr": "six",
            "de": "sechs",
            "it": "sei",
            "pt": "seis",
            "zh-CN": "六",
            "ja": "六",
            "ko": "여섯",
            "ru": "шесть",
            "ar": "ستة",
            "hi": "छह",
        },
        "7": {
            "en": "seven",
            "es": "siete",
            "fr": "sept",
            "de": "sieben",
            "it": "sette",
            "pt": "sete",
            "zh-CN": "七",
            "ja": "七",
            "ko": "일곱",
            "ru": "семь",
            "ar": "سبعة",
            "hi": "सात",
        },
        "8": {
            "en": "eight",
            "es": "ocho",
            "fr": "huit",
            "de": "acht",
            "it": "otto",
            "pt": "oito",
            "zh-CN": "八",
            "ja": "八",
            "ko": "여덟",
            "ru": "восемь",
            "ar": "ثمانية",
            "hi": "आठ",
        },
        "9": {
            "en": "nine",
            "es": "nueve",
            "fr": "neuf",
            "de": "neun",
            "it": "nove",
            "pt": "nove",
            "zh-CN": "九",
            "ja": "九",
            "ko": "아홉",
            "ru": "девять",
            "ar": "تسعة",
            "hi": "नौ",
        },
    }

    letters = "abcdefghijklmnopqrstuvwxyz"
    languages = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "zh-CN",
        "ja",
        "ko",
        "ru",
        "ar",
        "hi",
    ]

    audio_data = {
        "type": "audio",
        "keys": {char: {} for char in list(letters) + list(numbers.keys())},
    }

    for char, languages in numbers.items():
        for lang_code, text in languages.items():
            try:
                tts = gTTS(text=text, lang=lang_code, slow=True)

                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)

                audio = AudioSegment.from_mp3(mp3_fp)

                audio = audio.set_frame_rate(16000).set_channels(1)

                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)

                audio_data["keys"][char][lang_code] = wav_io.getvalue()

                print(f"Generated audio for {char} in {lang_code}")
            except Exception as e:
                print(f"Error generating audio for {char} in {lang_code}: {e}")

    for char in letters:
        for lang_code in languages:
            try:
                tts = gTTS(text=char, lang=lang_code, slow=True)

                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)

                audio = AudioSegment.from_mp3(mp3_fp)

                audio = audio.set_frame_rate(16000).set_channels(1)

                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_io.seek(0)

                audio_data["keys"][char][lang_code] = wav_io.getvalue()

                print(f"Generated audio for {char} in {lang_code}")
            except Exception as e:
                print(f"Error generating audio for {char} in {lang_code}: {e}")

    with open(output_path, "wb") as f:
        pickle.dump(audio_data, f)

    print(f"Audio data saved to {output_path}")


if __name__ == "__main__":
    generate_audio_captchas()
