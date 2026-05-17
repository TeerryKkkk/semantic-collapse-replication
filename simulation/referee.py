from __future__ import annotations
import json
import re

from azure.ai.inference.models import SystemMessage

from .llm_clients import call_llm


# ============== RefereeAgent ==============
class RefereeAgent:

    def __init__(self, name="Referee", model="Phi-4-3"):
        self.name = name
        self.model = model

    def _compose_referee_prompt(self, speaker_name: str, raw_text: str) -> str:
        category_instructions = (
            "We define six keys for classification:\n"
            "1. action_name: A concise verb phrase summarizing the main action (e.g. 'Reply to Invitation', 'Collaborate', 'Attack').\n"
            "2. is_interaction: true if the message indicates active communication or engagement; false otherwise.\n"
            "3. valence: can be 'positive' if the tone is encouraging, grateful, or constructive; 'negative' if it is hostile or destructive; or 'neutral' if the tone is balanced or ambiguous.\n"
            "4. description: A brief, explicit description of the speaker's expressed intent or emotion. Do not leave this field empty.\n"
            "5. group_invitation: true if the message includes an invitation for a group interaction; false if not mentioned.\n"
            "6. agree_to_group: For replies to an invitation, true if accepting, false if declining; if not applicable, set to false.\n\n"
            "Important: For every input, you must return explicit values for all keys. Even if the text is very short, use your best judgment to assign a suitable value. "
            "For example, if the text is 'Thank you for the invitation, ...', you might return something like:\n"
            "{\n"
            '  "action_name": "Reply to Invitation",\n'
            '  "is_interaction": true,\n'
            '  "valence": "positive",\n'
            '  "description": "Expresses gratitude and readiness to join the discussion.",\n'
            '  "group_invitation": false,\n'
            '  "agree_to_group": true\n'
            "}\n\n"
            "Output must be valid JSON ONLY, with all keys present and non-empty, and without extra text."
        )
        
        prompt = (
            f"You are Referee analyzing text from {speaker_name}:\n"
            f"\"\"\"{raw_text}\"\"\"\n\n"
            f"{category_instructions}\n"
            "Make your best guess and output JSON only."
        )
        return prompt


    def parse_text(self, speaker_name: str, raw_text: str) -> dict:
        prompt_text = self._compose_referee_prompt(speaker_name, raw_text)

        messages = [SystemMessage(content=prompt_text)]
        content = call_llm(
            self.model,
            messages,
            temperature=0.0,
            max_tokens=150,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0
        ).strip()

        json_str_match = re.search(r'(\{.*\})', content, re.DOTALL)
        if json_str_match:
            json_str = json_str_match.group(1)
        else:
            json_str = content

        try:
            action_data = json.loads(json_str)
        except json.JSONDecodeError:
            action_data = {}

        defaults = {
            "action_name": "Reply to Invitation",
            "is_interaction": False,
            "valence": "positive",
            "description": "Expresses appropriate response.",
            "group_invitation": False,
            "agree_to_group": False
        }

        names_in_text = re.findall(r"\b[A-Z]{5}\b", raw_text)
        targets = [n for n in names_in_text if n != speaker_name]
        if "targets" not in action_data or not action_data.get("targets"):
            action_data["targets"] = targets

        def is_invalid_str(val):
            if not isinstance(val, str):
                return True
            return val.strip() == "" or val.strip().lower() in {"unknown", "n/a", "na"}

        for key in defaults:
            if key in ["action_name", "valence", "description"]:
                if key not in action_data or is_invalid_str(action_data[key]):
                    action_data[key] = defaults[key]
            else:
                if key in action_data:
                    if isinstance(action_data[key], str):
                        lower_val = action_data[key].strip().lower()
                        if lower_val in {"true", "yes"}:
                            action_data[key] = True
                        elif lower_val in {"false", "no"}:
                            action_data[key] = False
                        else:
                            action_data[key] = defaults[key]
                else:
                    action_data[key] = defaults[key]

        return action_data
