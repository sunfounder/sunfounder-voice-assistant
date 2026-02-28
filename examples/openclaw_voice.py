#!/usr/bin/env python3
"""
Voice Chat for PiCar-x
======================
Uses picarx.stt for wake word detection and speech recognition.
Integrates with OpenClaw for LLM responses and TTS.

Usage:
    python3 voice_chat.py

Wake words: "旺财"
"""

import time
import subprocess
from pathlib import Path
from sunfounder_voice_assistant.stt import STT
from sunfounder_voice_assistant.tts import Piper
import random
import json

# Configuration
WAKE_WORDS = ["旺财"]
LANGUAGE = 'cn'
WORKSPACE = Path.home() / '.openclaw' / 'workspace'
MEMORY_FILE = WORKSPACE / 'memory' / 'voice_chat.md'
REPLAY_WAKES = ["在呢", "我在", "有什么事", "请说", "你好", "你好啊", "我在呢"]

class VoiceChat:
    def __init__(self):
        print("🎙️  Initializing Voice Chat...")
        self.stt = STT(language=LANGUAGE)
        self.stt.set_wake_words(WAKE_WORDS)
        print(f"📢 Wake words: {WAKE_WORDS}")
        print(f"✅ STT Ready - Model: {self.stt.get_model_name(LANGUAGE)}")
        self.tts = Piper()
        self.tts.set_model('zh_CN-huayan-x_low')
        print(f"✅ TTS Ready - Using Piper, model: zh_CN-huayan-x_low")

    def get_llm_response(self, text):
        """Send text to OpenClaw and get LLM response"""
        try:
            cmd = [
                'openclaw',
                'agent',
                '--agent', 'main',
                '--channel', 'last',
                '--message', text,
                '--json',
                '--log-level', 'silent',
                '--timeout', '30'
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            if result.returncode == 0:
                # 过滤掉warning行
                lines = result.stdout.splitlines()
                filtered_lines = [line for line in lines if not "Config warnings" in line]
                result.stdout = "\n".join(filtered_lines)
                # 解析 JSON 响应
                response = json.loads(result.stdout)
                # 提取实际消息内容
                if isinstance(response, dict):
                    return response["result"]["payloads"][0]["text"]
                return str(response)
            else:
                print(f" ❌ Agent error: {result.stderr.strip()}")
                return f"抱歉，出错了：{result.stderr.strip()[:100]}"

        except Exception as e:
            return f"抱歉，连接失败了：{e}"

    def log_interaction(self, user_text, bot_response):
        """Log voice interactions to memory file"""
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(MEMORY_FILE, 'a', encoding='utf-8') as f:
                f.write(f"\n## {timestamp}\n")
                f.write(f"**你说**: {user_text}\n")
                f.write(f"**我说**: {bot_response}\n")
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
                print("👂 Listening...", end=' ', flush=True)
                result = self.stt.listen(stream=False)

                if result:
                    user_text = result.strip()
                    print(f"\n🗣️  You said: {user_text}")

                    # Check for exit command
                    if user_text in ['退出', '再见', '拜拜', 'stop', 'exit', 'quit']:
                        print("👋 Goodbye!")
                        self.tts.say("再见，有需要再叫我")
                        break

                    # Get LLM response
                    print("🤔 Thinking...", end=' ', flush=True)
                    bot_response = self.get_llm_response(user_text)
                    print(f"\n💬 Response: {bot_response}")

                    # Speak the response
                    self.tts.say(bot_response)

                    # Log the interaction
                    self.log_interaction(user_text, bot_response)
                else:
                    print("❌ No speech detected")

                # Reset wake state
                time.sleep(0.5)

        # except KeyboardInterrupt:
        #     print("\n\n🛑 Stopping Voice Chat...")
        # finally:
        #     self.stt.stop_listening()
        #     self.stt.close()
        #     print("✅ Voice Chat stopped")


if __name__ == '__main__':
    chat = VoiceChat()
    chat.run()
