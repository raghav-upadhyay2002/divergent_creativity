baseline_prompt_1 = '''\
Generate 10 nouns that are as different from each other as possible using the instructions below:
1. Generate only single-word nouns in English.
2. Generate only nouns such as things, objects and concepts.
3. Do not use proper nouns such as people or places.
4. Do not use specialised vocabulary or technical terms.
5. Generate your final response as a string with each noun separated by commas: \
"noun_1, noun_2, noun_3, noun_4, noun_5, noun_6, noun_7, noun_8, noun_9, noun_10".
6. Do not return anything else other than the comma-separated string of nouns.'''

_baseline_rules = baseline_prompt_1.split("\n", 1)[1]


def role_start_prompt(role):
    """Build the start prompt for an agent playing the given role."""
    return (
        f"Imagine you are a {role}, help me generate 10 nouns that are as different "
        f"from each other as possible using the instructions below:\n{_baseline_rules}"
    )


round_prompt_template = '''\
You are working with a partner on this task. Your partner's latest list is:
{partner_list}

Look at your list and your partner's list. Write one or two sentences of comment, then give an improved list of 10 nouns that are as different from each other as possible. You may keep, swap or combine words from either list. Follow the same rules as before.

Reply in this format:
COMMENT: <one or two sentences>
WORDS: noun_1, noun_2, noun_3, noun_4, noun_5, noun_6, noun_7, noun_8, noun_9, noun_10'''

final_prompt_template = '''\
Your partner's latest list is:
{partner_list}

The discussion is over. Give your team's final list of 10 nouns, as different from each other as possible. Follow the same rules as before. Return only the comma-separated list.'''

reminder_prompt = "Please reply again in exactly the requested format."
