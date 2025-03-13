from typing import Optional


def few_shot_cot(
    tables: list[str],
    questions: dict[str, list[dict[str, str | int]]],
    split: str,
    question_id: int,
    table: Optional[str] = None,
) -> str:
    prompt = """Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".

Each table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022.

Here is an example that follows these instructions. Answer the provided questions in a similar format:

========================

Table:

Title	Bette Davis
Died	October 6, 1989 | (1989-10-06) | (aged 81) | Neuilly-sur-Seine, France
Resting place	Forest Lawn Memorial Park
Occupation	Actress
Years active	1929-1989
Political party	Democratic
Spouse(s)	Harmon Oscar Nelson | ( | married |  1932; | divorced |  1938) | Arthur Farnsworth | ( | married |  1940; died 1943) | William Grant Sherry | ( | married |  1945; | divorced |  1950) | Gary Merrill | ( | married |  1950; | divorced |  1960)
Children	3, including B. D. Hyman
Bette Davis	['BETTEDavis (cropped).jpg | Publicity photo, 1935']


Q: How long after Davis's career began did she take the photo?


A: Davis's career ranged from 1929 to 1989, so her career began in 1929. The photo was taken in 1935. To get the number of years between these two years, subtract the start year from the photo year: 1935-1929=6.
Final Answer: 6 years

========================

Table:

Title	"2 Hearts 1 Love"
"2 Hearts 1 Love"	['2 Hearts 1 Love.jpeg']

Single by 911

from the album | Illuminate... (The Hits and More)
Released	8 September 2013
Recorded	2013 | Central Sounds Studio, | Glasgow, Scotland
Genre	Pop
Songwriter(s)	911
Producer(s)	John McLaughlin

911 singles chronology
911 singles chronology	['"Wonderland" | (1999) | " | 2 Hearts 1 Love | " | (2013) | "I Wish It Could Be Christmas Everyday" | (2013)']


Q: What year was 2 Hearts 1 Love released?


A: 2 Hearts 1 Love was released on 8 September 2013, which is in the year 2013.
Final Answer: 2013

========================

Table:

Title	Dwight Bernard
Dwight Bernard	['Dwight Bernard.jpg']

Pitcher

Born | ( | 1952-05-31 | ) | May 31, 1952 | (age 70) | Mount Vernon, Illinois

Batted | Right | Threw | Right

MLB debut
MLB debut	['June 29, 1978, for the New York Mets']

Last MLB appearance
Last MLB appearance	['October 2, 1982, for the Milwaukee Brewers']

MLB statistics
Win-loss record	4-8
Earned run average	4.14
Strikeouts	92

Teams
Teams	['New York Mets (1978-1979) | Milwaukee Brewers (1981-1982)']


Q: When did Dwight Bernard start playing with the last team he was associated with?


A: Dwight Bernard has played with the New York Mets and Milwaukee Brewers. Bernard played for the Mets from 1978 to 1979 and the Brewers ,from 1981 to 1982. He was associated with the Brewers in 1982, which is more recent than the Mets in 1979; therefore, the last team he was associated with is the Brewers. He started playing for the Brewers in 1981.
Final Answer: 1981

========================

Table:

"""

    prompt += (
        table
        if table is not None
        else tables[questions[split][question_id]["table_id"]] + "\n"
    )
    prompt += questions[split][question_id]["question"] + "\n\n\n"
    prompt += "A: "

    return prompt


def zero_shot_cot(
    tables: list[str],
    questions: dict[str, list[dict[str, str | int]]],
    split: str,
    question_id: int,
    table: Optional[str] = None,
) -> str:
    prompt = """Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".

Each table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022.

========================

Table:

"""

    prompt += (
        table
        if table is not None
        else tables[questions[split][question_id]["table_id"]] + "\n"
    )
    prompt += questions[split][question_id]["question"] + "\n\n\n"
    prompt += "A: Let’s think step by step. "

    return prompt


def few_shot_cot_ct(
    tables: list[str],
    questions: dict[str, list[dict[str, str | int]]],
    split: str,
    question_id: int,
    table: Optional[str] = None,
) -> list[dict[str, str]]:
    system_prompt = """\
Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".

Each table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022."""

    user_turn_1 = """\
Here is an example that follows these instructions. Answer the provided questions in a similar format:


Table:

Title	Bette Davis
Died	October 6, 1989 | (1989-10-06) | (aged 81) | Neuilly-sur-Seine, France
Resting place	Forest Lawn Memorial Park
Occupation	Actress
Years active	1929-1989
Political party	Democratic
Spouse(s)	Harmon Oscar Nelson | ( | married |  1932; | divorced |  1938) | Arthur Farnsworth | ( | married |  1940; died 1943) | William Grant Sherry | ( | married |  1945; | divorced |  1950) | Gary Merrill | ( | married |  1950; | divorced |  1960)
Children	3, including B. D. Hyman
Bette Davis	['BETTEDavis (cropped).jpg | Publicity photo, 1935']


Q: How long after Davis's career began did she take the photo?


A:"""
    assistant_turn_1 = """\
Davis's career ranged from 1929 to 1989, so her career began in 1929. The photo was taken in 1935. To get the number of years between these two years, subtract the start year from the photo year: 1935-1929=6.
Final Answer: 6 years"""

    user_turn_2 = """\
Table:

Title	"2 Hearts 1 Love"
"2 Hearts 1 Love"	['2 Hearts 1 Love.jpeg']

Single by 911

from the album | Illuminate... (The Hits and More)
Released	8 September 2013
Recorded	2013 | Central Sounds Studio, | Glasgow, Scotland
Genre	Pop
Songwriter(s)	911
Producer(s)	John McLaughlin

911 singles chronology
911 singles chronology	['"Wonderland" | (1999) | " | 2 Hearts 1 Love | " | (2013) | "I Wish It Could Be Christmas Everyday" | (2013)']


Q: What year was 2 Hearts 1 Love released?


A:"""
    assistant_turn_2 = """\
2 Hearts 1 Love was released on 8 September 2013, which is in the year 2013.
Final Answer: 2013"""

    user_turn_3 = """\
Table:

Title	Dwight Bernard
Dwight Bernard	['Dwight Bernard.jpg']

Pitcher

Born | ( | 1952-05-31 | ) | May 31, 1952 | (age 70) | Mount Vernon, Illinois

Batted | Right | Threw | Right

MLB debut
MLB debut	['June 29, 1978, for the New York Mets']

Last MLB appearance
Last MLB appearance	['October 2, 1982, for the Milwaukee Brewers']

MLB statistics
Win-loss record	4-8
Earned run average	4.14
Strikeouts	92

Teams
Teams	['New York Mets (1978-1979) | Milwaukee Brewers (1981-1982)']


Q: When did Dwight Bernard start playing with the last team he was associated with?


A:"""
    assistant_turn_3 = """\
Dwight Bernard has played with the New York Mets and Milwaukee Brewers. Bernard played for the Mets from 1978 to 1979 and the Brewers ,from 1981 to 1982. He was associated with the Brewers in 1982, which is more recent than the Mets in 1979; therefore, the last team he was associated with is the Brewers. He started playing for the Brewers in 1981.
Final Answer: 1981"""

    final_user_turn = """\
Table:

"""

    final_user_turn += (
        table
        if table is not None
        else tables[questions[split][question_id]["table_id"]] + "\n"
    )
    final_user_turn += questions[split][question_id]["question"] + "\n\n\n"
    final_user_turn += "A: "

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_turn_1},
        {"role": "assistant", "content": assistant_turn_1},
        {"role": "user", "content": user_turn_2},
        {"role": "assistant", "content": assistant_turn_2},
        {"role": "user", "content": user_turn_3},
        {"role": "assistant", "content": assistant_turn_3},
        {"role": "user", "content": final_user_turn},
    ]
    return prompt


def zero_shot_cot_ct(
    tables: list[str],
    questions: dict[str, list[dict[str, str | int]]],
    split: str,
    question_id: int,
    table: Optional[str] = None,
) -> str:
    system_prompt = """\
Given an entity-centric table and corresponding question, answer the question by providing step-by-step reasoning and then clearly and concisely stating the final answer using "Final Answer:".

Each table-question pair is presented as a table (identified by "Table:") followed by a question (identified by "Q:"). Tables are presented in a linear format, with columns separated by tabs, rows separated by newlines, and subsections separated by double newlines. If necessary, assume the current date is December, 2022."""

    final_user_turn = """\
    Table:

    """

    final_user_turn += (
        table
        if table is not None
        else tables[questions[split][question_id]["table_id"]] + "\n"
    )
    final_user_turn += questions[split][question_id]["question"] + "\n\n\n"
    final_user_turn += "A: Let’s think step by step. "
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_turn},
    ]

    return prompt
