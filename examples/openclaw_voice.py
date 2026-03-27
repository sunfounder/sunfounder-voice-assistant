#!/usr/bin/env python3
"""
Voice Chat with OpenClaw
===================================
Uses sunfounder-voice-assistant.stt for wake word detection and speech recognition.
Integrates with OpenClaw for LLM responses and TTS.

Usage:
    python3 openclaw_voice.py

Wake words: "Amy"
"""

import time
import subprocess
from pathlib import Path
from sunfounder_voice_assistant.stt import STT
from sunfounder_voice_assistant.tts import Piper
import random
import json

# Configuration
WAKE_WORDS = ["amy", "hello amy", "hi amy", "hey amy"]
STT_LANGUAGE = 'en-us'
TTS_MODEL = 'en_US-amy-medium'
WORKSPACE = Path.home() / '.openclaw' / 'workspace'
MEMORY_FILE = WORKSPACE / 'memory' / 'voice_chat.md'
REPLAY_WAKES = ["I'm here", "Yes", "What can I do for you", "Please go ahead", "Hello", "How can I help"]

class VoiceChat:
    def __init__(self):
        print("🎙️  Initializing Voice Chat...")
        self.stt = STT(language=STT_LANGUAGE)
        self.stt.set_wake_words(WAKE_WORDS)
        print(f"📢 Wake words: {WAKE_WORDS}")
        print(f"✅ STT Ready - Model: {self.stt.get_model_name(STT_LANGUAGE)}")
        self.tts = Piper()
        self.tts.set_model(TTS_MODEL)
        print(f"✅ TTS Ready - Using Piper, model: {TTS_MODEL}")

    def get_llm_response(self, text):
        """Send text to OpenClaw and get LLM response"""
        try:
            # Add system prompt for voice-friendly responses
            voice_prompt = (
                "This is a voice conversation. Please respond in a way that is easy to read aloud by TTS:\n"
                "- Keep responses concise and conversational\n"
                "- Do NOT use Markdown formatting (no **, ##, -, `, etc.)\n"
                "- Do NOT use emojis or special symbols\n"
                "- Do NOT use code blocks or technical formatting\n"
                "- Speak naturally as if talking to someone\n"
                "- Avoid lists, bullet points, or numbered items\n"
                "\n"
                "User's message: "
            )
            
            cmd = [
                'openclaw',
                'agent',
                '--agent', 'main',
                '--channel', 'last',
                '--message', voice_prompt + text,
                '--json',
                '--log-level', 'silent',
                '--timeout', '60'
            ]

            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            if result.returncode == 0:
                # Filter out warning lines
                lines = result.stdout.splitlines()
                filtered_lines = [line for line in lines if "Config warnings" not in line]
                result.stdout = "\n".join(filtered_lines)
                # Parse JSON response
                response = json.loads(result.stdout)
                # Extract actual message content
                if isinstance(response, dict):
                    return True, response["result"]["payloads"][0]["text"]
                return True, str(response)
            else:
                print(f" ❌ Agent error: {result.stderr.strip()}")
                return False, f"Sorry, an error occurred: {result.stderr.strip()[:100]}"

        except Exception as e:
            return False, f"Sorry, connection failed: {e}"

    def log_interaction(self, user_text, bot_response):
        """Log voice interactions to memory file"""
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(MEMORY_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n## {timestamp}\n")
                f.write(f"**You said**: {user_text}\n")
                f.write(f"**I said**: {bot_response}\n")
        except Exception as e:
            print(f"⚠️  Could not log: {e}")

    def run(self):
        """Main voice chat loop"""
        print("=" * 50)
        print("🎙️  Voice Chat Started!")
        print("   Say a wake word to begin...")
        print("   Press Ctrl+C to stop")
        print("=" * 50)
        print()

        while True:
            # Start listening for wake words
            self.stt.start_listening_wake_words()

            # Wait for wake word
            print("⏳ Waiting for wake word...", end=' ', flush=True)
            while not self.stt.is_waked():
                time.sleep(0.5)

            if self.stt.is_waked():
                print("✅ Waked!")
                self.tts.say(random.choice(REPLAY_WAKES))

                # Listen for user's speech
                print("👂 Listening...", flush=True)
                for result in self.stt.listen(stream=True):
                    if result["done"]:
                        print(f"\r\x1b[Kfinal: {result['final']}")
                    else:
                        print(f"\r\x1b[Kpartial: {result['partial']}", end="", flush=True)

                if result:
                    user_text = result.strip()
                    print(f"\n🗣️  You said: {user_text}")

                    # Check for exit command
                    # if user_text.lower() in ['stop', 'exit', 'quit', 'goodbye', 'bye']:
                    #     print("👋 Goodbye!")
                    #     self.tts.say("Goodbye! Call me if you need me again")
                    #     break

                    # Get LLM response
                    print("🤔 Thinking...", end=' ', flush=True)
                    success, bot_response = self.get_llm_response(user_text)
                    if success:
                        print(f"\n💬 Response: {bot_response}")
                        self.tts.say(bot_response)
                    else:
                        self.tts.say("Sorry, error occurred.")

                    self.log_interaction(user_text, bot_response)
                else:
                    print("❌ No speech detected")

                # Reset wake state
                time.sleep(0.5)


if __name__ == '__main__':
    chat = VoiceChat()
    chat.run()
