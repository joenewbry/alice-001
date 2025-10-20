#!/usr/bin/env python3
"""
LLM-powered command assistant for handling unrecognized commands
and providing helpful suggestions.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List
from .prompts import SYSTEM_PROMPT, get_suggestion_prompt, get_clarification_prompt


class CommandAssistant:
    """LLM assistant for command interpretation and suggestions"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize command assistant.
        
        Args:
            model: OpenAI model to use (gpt-4o-mini is cost-effective)
            temperature: Sampling temperature (0.7 for balanced creativity)
        """
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('OPEN_AI_KEY')
        if not api_key:
            raise ValueError("OPEN_AI_KEY not found in environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.conversation_history = []
        
        print(f"✓ Command assistant initialized with {model}")
    
    def get_suggestion(self, unrecognized_text: str, context: Optional[str] = None) -> str:
        """
        Get command suggestions for unrecognized text.
        
        Args:
            unrecognized_text: The text that wasn't recognized
            context: Optional context about recent commands
            
        Returns:
            Helpful suggestion text
        """
        try:
            # Generate prompt
            user_prompt = get_suggestion_prompt(unrecognized_text, context or "")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            
            suggestion = response.choices[0].message.content.strip()
            
            print(f"🤖 LLM suggestion: {suggestion}")
            
            return suggestion
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            # Fallback to generic suggestion
            return "Try saying 'move left', 'grasp', or 'please dance'"
    
    def get_clarification(self, ambiguous_text: str, possible_commands: List[str]) -> str:
        """
        Get clarification for ambiguous commands.
        
        Args:
            ambiguous_text: The ambiguous command
            possible_commands: List of possible interpretations
            
        Returns:
            Clarification question
        """
        try:
            user_prompt = get_clarification_prompt(ambiguous_text, possible_commands)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=100
            )
            
            clarification = response.choices[0].message.content.strip()
            return clarification
            
        except Exception as e:
            print(f"❌ LLM error: {e}")
            return f"Did you mean {possible_commands[0]} or {possible_commands[1]}?"
    
    def add_to_context(self, command: str, result: str):
        """
        Add command and result to conversation context.
        
        Args:
            command: The command that was executed
            result: The result (success/failure)
        """
        self.conversation_history.append({
            'command': command,
            'result': result
        })
        
        # Keep only last 5 commands for context
        if len(self.conversation_history) > 5:
            self.conversation_history.pop(0)
    
    def get_context_summary(self) -> str:
        """
        Get summary of recent conversation context.
        
        Returns:
            Context summary string
        """
        if not self.conversation_history:
            return ""
        
        summary = "Recent commands: "
        summary += ", ".join([
            f"{item['command']} ({item['result']})"
            for item in self.conversation_history[-3:]
        ])
        
        return summary
    
    def fuzzy_match_command(self, text: str, valid_commands: List[str]) -> Optional[str]:
        """
        Use LLM to find the best matching command from a list.
        
        Args:
            text: Input text
            valid_commands: List of valid command examples
            
        Returns:
            Best matching command or None
        """
        try:
            prompt = f"""The user said: "{text}"

Which of these commands is most similar?
{chr(10).join([f'- {cmd}' for cmd in valid_commands])}

Respond with ONLY the matching command, or "NONE" if no match."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a command matching assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            match = response.choices[0].message.content.strip()
            
            if match == "NONE" or match not in valid_commands:
                return None
            
            return match
            
        except Exception as e:
            print(f"❌ Fuzzy match error: {e}")
            return None


def demo_assistant():
    """Demo command assistant functionality"""
    assistant = CommandAssistant()
    
    # Test unrecognized commands
    test_cases = [
        "make it spin around",
        "pick up the bottle",
        "wave hello",
        "do a backflip",
        "go to sleep"
    ]
    
    print("=== Testing Command Assistant ===\n")
    
    for text in test_cases:
        print(f"Input: \"{text}\"")
        suggestion = assistant.get_suggestion(text)
        print(f"Suggestion: {suggestion}\n")
    
    # Test fuzzy matching
    print("\n=== Testing Fuzzy Matching ===\n")
    
    valid_commands = [
        "move left",
        "move right",
        "grasp",
        "release",
        "please dance"
    ]
    
    fuzzy_tests = [
        "go left",
        "turn right",
        "grab it",
        "let go",
        "do a dance"
    ]
    
    for text in fuzzy_tests:
        print(f"Input: \"{text}\"")
        match = assistant.fuzzy_match_command(text, valid_commands)
        if match:
            print(f"  ✓ Matched: {match}")
        else:
            print(f"  ❌ No match")
        print()


if __name__ == "__main__":
    demo_assistant()

