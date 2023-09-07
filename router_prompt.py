import os
import openai
import pandas as pd


messages = [
    "I'm interested in registering my daughter for GenAI Summer Camp. How can I proceed?",
    "Please let me know the steps to enroll my child in the camp.",
    "I'd like to sign up my son for the camp. What information do you need from me?",
    "I'm ready to register my two kids for the GenAI Summer Camp. What's the next step?",
    "How can I go about enrolling my child in the camp? I'm eager to get them signed up.",
    "Can you tell me about the activities planned for the camp?",
    "I'd like to enroll my two kids in GenAI Summer Camp. How can I register them?",
    "What are the age requirements for the camp, and is there a scholarship program?",
    "My son is really into coding. Is GenAI Summer Camp suitable for young programmers?",
    "When does the registration for GenAI Summer Camp close, and what are the fees?",
    "Is there any discount for early registration?",
    "Are there any special workshops for kids interested in artificial intelligence?",
    "How do you ensure the safety of the campers during the program?",
    "I have a question about dietary restrictions. How is food managed at the camp?",
    "Can you provide information about the camp's location and transportation options?"
]

classifications = [
    "Sign-up request",
    "Sign-up request",
    "Sign-up request",
    "Sign-up request",
    "Sign-up request",
    "Question about the camp",
    "Sign-up request",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp",
    "Question about the camp"
]


def router_prompt(message):
    return [
        {"role": "system",
         "content": "You are a conversational AI assistant for parents interested in the fictional GenAI Summer Camp. You will be provided with a message, and your task is to determine whether the parent's intention is to ask a question about the camp or sign their kid up."},
        {"role": "user", "content": message}
    ]


def test_prompt():
    results = []
    for message, classification in zip(messages, classifications):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=router_prompt(message),
            temperature=0,
            max_tokens=256
        )
        determination = response.choices[0].message.content
        results.append((message, classification, determination))
    tests_df = pd.DataFrame.from_records(results, columns=["Parent Message", "True Classification","AI Classification"])
    tests_df.to_csv('experiments/router_prompt_res.csv')


if __name__ == "__main__":
    test_prompt()
