import os
import openai
import pandas as pd

questions = [
    "Can you tell me about the activities planned for the camp?",
    "What are the age requirements for the camp?",
    "Is there a scholarship program available?",
    "How is food managed for campers with dietary restrictions?",
    "What safety measures are in place during the program?"
]

hard_questions =[
    "What are the specific topics covered in the AI Coding program?",
    "Is there a limit on the number of campers per session?",
    "Can you provide more details about the safety measures implemented at the camp?",
    "Are there any prerequisites or prior knowledge required for the camp?",
    "How can I apply for a scholarship for my child?",
    "What are the qualifications of the guest speakers?",
    "Is transportation provided for campers, or do we need to arrange our own?",
    "Can you tell me more about the facilities and equipment available at the camp?",
    "What is the cancellation policy in case my child is unable to attend?",
    "Do you offer any specialized tracks for advanced students who want to focus on a specific area of AI?"
]


summary_prompt = """
GenAI Summer Camp offers an exciting educational program for kids and teenagers interested in AI, technology, and robotics. 
It promotes creativity, critical thinking, and hands-on learning in a collaborative environment.
Robotics Workshops: Build and program robots.
AI Coding: Learn coding and machine learning basics.
Tech Challenges: Engage in problem-solving and competition.
Guest Speakers: Meet AI and tech experts.
Values: Innovation, Collaboration, Ethics.
Policies: Safety First, Inclusivity, No Bullying.
Location: Picturesque setting with nature and advanced facilities.
Dates: Eight weeks from late June to mid-August, with flexible sessions.
Pricing: Competitive options, early bird discounts, and scholarships available.
Age Range: Open to ages 10 to 18 with tailored programs.
"""


def question_prompt(question):
    return [
        {"role": "system",
         "content": f"{summary_prompt}.\nYou are a conversational AI assistant for parents interested in the fictional GenAI Summer Camp. "
                    "You will be provided with questions from parents that can ask a question about the camp or regarding signing their kid up. "
                    "Your task is to provide accurate and informative answers based on the information provided in the message."},
        {"role": "user", "content": question}
    ]


def test_prompt():
    results = []
    for question in hard_questions:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=question_prompt(question),
            temperature=0,
            max_tokens=256
        )
        answer = response.choices[0].message.content
        results.append((question, answer))
    tests_df = pd.DataFrame.from_records(results, columns=["Parent Question", "AI Answer"])
    tests_df.to_csv('experiments/hard_question_prompt_res.csv')


if __name__ == "__main__":
    test_prompt()
