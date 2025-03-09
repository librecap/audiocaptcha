# AudioCaptcha

Generate highly secure, accessible audio CAPTCHAs. Human recognizable but bot-resistant audio challenges with sequences of spoken characters and sophisticated obfuscation techniques.

## Quick Start

```bash
git clone https://github.com/tn3w/audiocaptcha.git
cd audiocaptcha/characters
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional: python generate_dataset.py (to generate a new dataset)
python generate_captcha.py
```

## Overview

AudioCaptcha is a Python library for generating audio CAPTCHA challenges that are accessible to humans but difficult for bots to solve. The library provides various types of audio challenges with customizable difficulty levels and obfuscation techniques to enhance security while maintaining accessibility.

This is a Prove of Concept implementation intended to be used for a Rust implementation for the LibreCap project.

## Features/Types

### Currently Available

#### Character Captchas
- **Description**: Audio clips of spoken characters (letters and digits) in sequence
- **Languages**: Multiple language support using Google Text-to-Speech
- **Customization**: Adjustable length, character set, and background noise levels
- **Accessibility**: Clear pronunciation with configurable speech rate

### Planned Features

#### Animal Sounds
- Audio clips of various animal sounds that users need to identify

#### Distorted Music
- Identify distorted or otherwise altered music between harmonic sounds

#### Rhythm Challenge
- Find the odd one out in a sequence of rhythmic sounds

## Installation

```bash
git clone https://github.com/tn3w/audiocaptcha.git
cd audiocaptcha/characters
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional: python generate_dataset.py (to generate a new dataset)
python generate_captcha.py
```

### Requirements
- Python 3.7+
- pydub
- gTTS (Google Text-to-Speech)

## Advanced Configuration

```python
usage: generate_captcha.py [-h] [--type {numbers,letters,mixed}]
                           [--language LANGUAGE] [--count COUNT]
                           [--format {mp3,wav,ogg}] [--no-obfuscate]

Generate a character audio captcha

options:
  -h, --help            show this help message and exit
  --type {numbers,letters,mixed}
                        Type of characters to use (default: mixed)
  --language LANGUAGE   Language code (default: en)
  --count COUNT         Number of characters (default: 6)
  --format {mp3,wav,ogg}
                        Audio format (default: mp3)
  --no-obfuscate        Disable audio obfuscation techniques
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/tn3w/audiocaptcha.git
cd audiocaptcha

# Choose a captcha type
cd characters

# Create a new virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Copyright 2025 LibreCap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.