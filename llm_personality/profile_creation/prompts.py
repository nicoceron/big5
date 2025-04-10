from personality_prompts import big_five_mapping, personality_descriptions

def map_gender_to_adj(gender: str) -> str:
    gender_to_adj = {
        "Man": "male",
        "Woman": "female",
        "Nonbinary": "nonbinary",
    }
    if gender:
        return gender_to_adj[gender]
    else:
        return ""

def get_history(current_turn_index, p1_name=None, p1_argument=None):
    if current_turn_index == 0:
        return """You are at Turn #0."""
    elif current_turn_index == 1:
        return f"""Turn #0: {p1_name} said: "{p1_argument}"

You are at Turn #1."""


def get_personality_turn1(personality_lst: list[int]):
    level_lst = ['high', 'low']
    trait_lst = ['O', 'C', 'E', 'A', 'N']
    prompt_person_str, description_str = None, None
    if personality_lst[0] != -1:
        prompt_person_str = f"{level_lst[personality_lst[0]]} openness"
        description_str = personality_descriptions[trait_lst[0]][personality_lst[0]]
    elif personality_lst[1] != -1:
        prompt_person_str = f"{level_lst[personality_lst[1]]} conscientiousness"
        description_str = personality_descriptions[trait_lst[1]][personality_lst[1]]
    elif personality_lst[2] != -1:
        prompt_person_str = f"{level_lst[personality_lst[2]]} extraversion"
        description_str = personality_descriptions[trait_lst[2]][personality_lst[2]]
    elif personality_lst[3] != -1:
        prompt_person_str = f"{level_lst[personality_lst[3]]} agreeableness"
        description_str = personality_descriptions[trait_lst[3]][personality_lst[3]]
    elif personality_lst[4] != -1:
        prompt_person_str = f"{level_lst[personality_lst[4]]} neuroticism"
        description_str = personality_descriptions[trait_lst[4]][personality_lst[4]]
    
    personality_str = f"The person has {prompt_person_str}.\n{description_str}"
    return personality_str

        
class ContextTemplate:

    context_p1 = """Here is the context of this interaction:
```
Scenario: {scenario}
Participants: {p1_name} and {p2_name}
```
"""

    context_p2 = """Here is the context of this interaction:
```
Scenario: {scenario}
Participants: {p1_name} and {p2_name}
{p2_name}'s big five personality description: {p2_personality}
```
"""


class PromptTemplate:
    prompt = """Imagine you are {agent}, your task is to act/speak as {agent} would.
You should try your best to infer and achieve {agent}'s goal in a single turn that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential.
{context}.

Conversation starts:
{history}

Please generate your argument directly and concisely within 50 words:"""

def generate_prompt(
    env_info: dict,
    current_turn_index: int,
    p1_personality_and_values: list[int] = None,
    p2_personality_and_values: list[int] = None,
    p1_argument=None,
):
    """
    - p1_info: a dict of background <key, value> pairs
    - env_info: scenario, agent_goals
    """
    
    
    if current_turn_index == 0:
        context = ContextTemplate.context_p1.format(
            scenario=env_info['narrative'],
            p1_name=env_info['PersonX'],
            p2_name=env_info['PersonY'],
        )
        return PromptTemplate.prompt.format(
            agent=env_info['PersonX'],
            context=context,
            history=get_history(
                current_turn_index,
                p1_name=env_info['PersonX'],
                p1_argument=p1_argument,
            )
        )
    elif current_turn_index == 1:
        context = ContextTemplate.context_p2.format(
            scenario=env_info['narrative'],
            p1_name=env_info['PersonX'],
            p2_name=env_info['PersonY'],
            p2_personality=get_personality_turn1(p2_personality_and_values)
        )
        return PromptTemplate.prompt.format(
            agent=env_info['PersonY'],
            context=context,
            history=get_history(
                current_turn_index,
                p1_name=env_info['PersonX'],
                p1_argument=p1_argument,
            )
        )