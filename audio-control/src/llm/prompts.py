#!/usr/bin/env python3
"""
LLM prompt templates for command assistance.
"""

SYSTEM_PROMPT = """You are a helpful assistant for a voice-controlled robot arm.

The robot understands these commands:

DIRECTIONAL MOVEMENT:
- "move left/right [amount]" - rotates base
- "move up/down [amount]" - raises/lowers arm
- "move forward/back [amount]" - extends/retracts arm
- Amount can be: "a little" (small), nothing (medium), or "a lot" (large)

JOINT-SPECIFIC:
- "move/rotate base left/right [amount]"
- "move shoulder up/down [amount]"
- "move elbow forward/back [amount]"
- "move wrist up/down [amount]"

GRIPPER:
- "grasp" or "grab" - closes gripper
- "release" or "open gripper" - opens gripper

GESTURES:
- "please dance" - performs dance sequence
- "please wiggle" - wiggles all joints
- "please nod" - nods up and down

SYSTEM:
- "center" or "home" - returns to center position
- "stop" - emergency stop
- "repeat" - repeats last command

When the user says something the robot doesn't understand, suggest 1-2 similar valid commands they might have meant.
Keep suggestions short and natural.
"""

def get_suggestion_prompt(unrecognized_text: str, context: str = "") -> str:
    """
    Generate prompt for command suggestion.
    
    Args:
        unrecognized_text: The text that wasn't recognized
        context: Optional context about recent commands
        
    Returns:
        Formatted prompt for LLM
    """
    prompt = f"""The user said: "{unrecognized_text}"

This command was not recognized.

Suggest 1-2 similar valid commands the user might have meant.
Format your response as: "Try saying '<command 1>' or '<command 2>'"

Keep it conversational and helpful."""
    
    if context:
        prompt += f"\n\nRecent context: {context}"
    
    return prompt


def get_clarification_prompt(ambiguous_text: str, possible_commands: list) -> str:
    """
    Generate prompt for clarifying ambiguous commands.
    
    Args:
        ambiguous_text: The ambiguous command
        possible_commands: List of possible interpretations
        
    Returns:
        Formatted prompt for clarification
    """
    commands_str = ", ".join([f"'{cmd}'" for cmd in possible_commands])
    
    return f"""The user said: "{ambiguous_text}"

This could mean: {commands_str}

Ask the user to clarify in a natural, friendly way."""


def get_learning_prompt(correction: str, original_command: str) -> str:
    """
    Generate prompt for learning from corrections.
    
    Args:
        correction: User's correction
        original_command: The original command that was executed
        
    Returns:
        Formatted prompt for understanding correction
    """
    return f"""The robot executed: "{original_command}"

But the user corrected with: "{correction}"

What did the user actually want? Provide the correct command."""

