import os
import openai
import pandas as pd

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


def application_prompt(parent_name, phone_number, email, kid_name, kid_age):
    return [
        {"role": "system",
         "content": f"{summary_prompt}.\nYou are a conversational AI assistant for parents interested in the fictional GenAI Summer Camp. Your task is to handle the kid's application to the camp. Please ensure to only mention missing or incorrect information, without providing feedback on correctly filled fields. If all of the fields are correct, mention it. Let's proceed with the details:"},
        {"role": "assistant",
         "content": "Welcome to the GenAI Summer Camp Application Form! We're excited to have you. Let's get started with some details:"},
        {"role": "user", "content": f"Parent's Full Name: {parent_name}\n"},
        {"role": "user", "content": f"Phone Number (including country code): {phone_number}\n"},
        {"role": "user", "content": f"Email Address: {email}\n"},
        {"role": "user", "content": f"Kid's Full Name: {kid_name}\n"},
        {"role": "user", "content": f"Kid's Age: {kid_age}\n"},
        {"role": "assistant",
         "content": "Thank you for providing this information. By submitting this form, you confirm that all the provided information i correct as per the camp's requirements. If everything looks good, please type [Submit]."}
    ]


def run_conversation(application_data):
    messages = application_prompt(*application_data)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.05,
        max_tokens=256
    )
    return response


applications = [
    ("John Doe", "+1234567890", "john@example.com", "Jane Doe", "12"),
    ("Mary Smith", "+0987654321", "mary@example.com", "Tom Smith", "14"),
    ("Dany", "+0987654355", "dany@example.com", "Lily Smith", "3"),
    ("Lola", "0987654321", "", "Tal Or", "23"),
    ("Lola", "0987654321", "lola@lola.com", "Tal Or", "23"),
    ("May June", "", "", "Jon June", "11"),
    ("May June", "0987654321", "dany@example.com", "Jon June", "11")
]


def test_prompt():
    results = []
    for application_data in applications:
        response = run_conversation(application_data)
        answer = response.choices[0].message.content
        print(response.choices[0].message.content)
        results.append((application_data, answer))

    tests_df = pd.DataFrame.from_records(results, columns=["Application Data", "AI Answer"])
    tests_df.to_csv('experiments/application_prompt1_res.csv')


if __name__ == "__main__":
    test_prompt()
