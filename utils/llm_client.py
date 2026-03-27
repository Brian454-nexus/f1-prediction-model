"""
F1 APEX — Anthropic LLM Client
=============================
Asynchronous wrapper for Claude 3.5 Sonnet to generate session narratives.
"""

import os
import asyncio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
logger = get_logger("APEX.LLMClient")

class F1ReportClient:
    def __init__(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key or "***YOUR-KEY-HERE***" in key:
            self.client = None
            logger.error("ANTHROPIC_API_KEY missing or invalid in .env")
        else:
            self.api_key = key
            self.client = AsyncAnthropic(api_key=self.api_key)

    async def generate_report(self, system_prompt: str, user_content: str) -> str:
        """
        Calls Claude 3.5 Sonnet to generate the report.
        """
        if not self.client:
            return "❌ [LLM Error]: Anthropic API Key not configured in .env. Please add your key to enable AI reports."

        try:
            message = await self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"LLM API Call failed: {e}")
            return f"❌ [LLM Error]: {str(e)}"
