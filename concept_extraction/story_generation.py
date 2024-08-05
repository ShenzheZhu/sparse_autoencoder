import os
import os

import base64
import requests
import json
import argparse
import pandas as pd
from openai import OpenAI

# parser = argparse.ArgumentParser(description="Extracting question, image path, answer and object scene")
# parser.add_argument("--question_prompt", type=str, default='/disk/nfs/gazinasvolume1/s2521923/Data/CLEVR_v1.0/prompt', help="path to semantic mapping")
# parser.add_argument("--image_path", type=str, default='/disk/nfs/gazinasvolume1/s2521923/Data/CLEVR_v1.0/images')
# parser.add_argument("--save", type=str, default='/disk/nfs/gazinasvolume1/s2521923/Data/CLEVR_v1.0/prompt', help="path for saving the data")

# args = parser.parse_args()


class GPTAssistant:
    def __init__(self, api_key: str, url: str=None, model: str = "gpt-4-turbo"):
        self.client = OpenAI(

            api_key=api_key,
            # base_url=url,
        )


        self.model = model

    def load_json_data(self, json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)


    def get_response_content(self, response):
        try:
            response_json = response.json()

            if 'choices' not in response_json:
                print("Error: 'choices' key not found in the response JSON.")
                return "None"

            choices = response_json['choices']

            # Check if 'choices' is a list and is not empty
            if not isinstance(choices, list) or len(choices) == 0:
                print("Error: 'choices' is not a valid list or is empty.")
                return "None"

            # Check if the first item in 'choices' has the 'message' key
            if 'message' not in choices[0]:
                print("Error: 'message' key not found in the first item of 'choices'.")
                return "None"

            message = choices[0]['message']

            # Check if 'message' has the 'content' key
            if 'content' not in message:
                print("Error: 'content' key not found in the 'message'.")
                return "None"

            # Return the content if all checks pass
            return message['content']

        except requests.exceptions.JSONDecodeError:
            print("Error: Failed to decode JSON from the response.")
            return "None"


    # def generate_programs(self, system_prompt, prompt, question):
    def generate_programs(self, persoanlity):
        # system_prompt = system_prompt.rstrip()
        # prompt = (prompt + question).rstrip()
        # prompt = prompt.rstrip()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"{persoanlity}\nPlease share a personal story in 500 words reflecting your personality by narrating an anecdote from your childhood that still influences your life today. Trying to include more adjetive words to describe your memory. The story should be various and colorful."
                },
                # {
                #     "role": "user",
                #     "content": "You posses an entroverted personlity who is highly sociable, thrives in dynamic environments, loves exploring new ideas, and has a knack for adapting quickly to unexpected situations."
                # }
            ]
        )
        # print("completiom",completion)
        response = completion.choices[0].message.content
        # print(response)
        return response
        # return self.get_response_content(response)



def main():

    personality_desciption = {
        'Extraversion': 'You posses an extraverted personality who thrives on engaging with others, draws energy from social interactions, and enjoys taking the initiative and leading projects with enthusiasm.',
        'Introversion': "You posses an introverted personality who thrives on introspection, finds strength in solitude, and prefers a calm environment to recharge and gain deep insights.",
        'Intuition': "You posses an intuitive personality who loves exploring new ideas, constantly considers future possibilities, and excels at seeing the bigger picture and offering innovative perspectives.",
        'Observant': "You posses an observant personality who focuses on the present, values practicality and simplicity, and excels at taking concrete actions to achieve tangible results.",
        "Thinking": "You are a thinking person who relies on logic and objective information to make decisions, values fairness and effectiveness in relationships, and prioritizes rational solutions over emotional responses.",
        "Feeling": "You are a feeling person who values emotions and compassion, prioritizes the well-being of others in decision-making, and uses empathy and emotional intelligence to guide your actions.",
        "Judging": "You posses a judging personality who values structure and planning, prefers clarity and closure in decision-making, and excels at creating actionable plans to achieve your goals.",
        'Prospecting': "You posses a prospecting personality who hrives on flexibility and spontaneity, enjoys exploring new possibilities, and adapts quickly to changing circumstances with creativity and enthusiasm.",
        "ISTJ": "You are an ISTJ person for MBTI test. Your traits are as follows, 1) Quiet, serious, earn success by being thorough and dependable. 2) Practical, matter-of-fact, realistic, and responsible. 3) Decide logically what should be done and work toward it steadily, regardless of distractions. 4) Take pleasure in making everything orderly and organized their work-your home, your life. 5) Value traditions and loyalty.",
        'ISFJ': 'You are an ISFJ person for MBTI test. Your traits are as follows, 1) Quiet, friendly, responsible, and conscientious. 2) Committed and steady in meeting their obligations. 3) Thorough, painstaking, and accurate. Loyal, considerate, notice and remember specifics about people who are important to them, concerned with how others feel. 4) Strive to create an orderly and harmonious environment at work and at home.',
        'INFJ':'You are an INFJ person for MBTI test. Your traits are as follows, Seek meaning and connection in ideas, relationships, and material possessions. Want to understand what motivates people and are insightful about others. Conscientious and committed to their firm values. Develop a clear vision about how best to serve the common good. Organized and decisive in implementing their vision.',
        'INTJ': 'You are an INTJ person for MBTI test. Your traits are as follows, Have original minds and great drive for implementing their ideas and achieving their goals. Quickly see patterns in external events and develop long-range explanatory perspectives. When committed, organize a job and carry it through. Skeptical and independent, have high standards of competence and performance—for themselves and others.',
        'ISTP': 'You are an ISTP person for MBTI test. Your traits are as follows, Tolerant and flexible, quiet observers until a problem appears, then act quickly to find workable solutions. Analyze what makes things work and readily get through large amounts of data to isolate the core of practical problems. Interested in cause and effect, organize facts using logical principles, value efficiency.',
        'INFP': 'You are an INFP person for MBTI test. Your traits are as follows, Idealistic, loyal to their values and to people who are important to them. Want to live a life that is congruent with their values. Curious, quick to see possibilities, can be catalysts for implementing ideas. Seek to understand people and to help them fulfill their potential. Adaptable, flexible, and accepting unless a value is threatened.',
        'INTP': 'You are an INTP person for MBTI test. Your traits are as follows, Seek to develop logical explanations for everything that interests them. Theoretical and abstract, interested more in ideas than in social interaction. Quiet, contained, flexible, and adaptable. Have unusual ability to focus in depth to solve problems in their area of interest. Skeptical, sometimes critical, always analytical.',
        'ESTP': 'You are an ESTP person for MBTI test. Your traits are as follows, Flexible and tolerant, take a pragmatic approach focused on immediate results. Bored by theories and conceptual explanations; want to act energetically to solve the problem. Focus on the here and now, spontaneous, enjoy each moment they can be active with others. Enjoy material comforts and style. Learn best through doing.',
        'ESFP': 'You are an ESFP person for MBTI test. Your traits are as follows, Outgoing, friendly, and accepting. Exuberant lovers of life, people, and material comforts. Enjoy working with others to make things happen. Bring common sense and a realistic approach to their work and make work fun. Flexible and spontaneous, adapt readily to new people and environments. Learn best by trying a new skill with other people.',
        'ENFP': 'You are an ENFP person for MBTI test. Your traits are as follows, Warmly enthusiastic and imaginative. See life as full of possibilities. Make connections between events and information very quickly, and confidently proceed based on the patterns they see. Want a lot of affirmation from others, and readily give appreciation and support. Spontaneous and flexible, often rely on their ability to improvise and their verbal fluency.',
        'ENTP': 'You are an ENTP person for MBTI test. Your traits are as follows, Quick, ingenious, stimulating, alert, and outspoken. Resourceful in solving new and challenging problems. Adept at generating conceptual possibilities and then analyzing them strategically. Good at reading other people. Bored by routine, will seldom do the same thing the same way, apt to turn to one new interest after another.',
        'ESTJ': 'You are an ESTJ person for MBTI test. Your traits are as follows, Practical, realistic, matter-of-fact. Decisive, quickly move to implement decisions. Organize projects and people to get things done, focus on getting results in the most efficient way possible. Take care of routine details. Have a clear set of logical standards, systematically follow them and want others to also. Forceful in implementing their plans',
        'ESFJ': 'You are an ESFJ person for MBTI test. Your traits are as follows, Warmhearted, conscientious, and cooperative. Want harmony in their environment, work with determination to establish it. Like to work with others to complete tasks accurately and on time. Loyal, follow through even in small matters. Notice what others need in their day-to-day lives and try to provide it. Want to be appreciated for who they are and for what they contribute.',
        'ENFJ': 'You are an ENFJ person for MBTI test. Your traits are as follows, Warm, empathetic, responsive, and responsible. Highly attuned to the emotions, needs, and motivations of others. Find potential in everyone, want to help others fulfill their potential. May act as catalysts for individual and group growth. Loyal, responsive to praise and criticism. Sociable, facilitate others in a group, and provide inspiring leadership.',
        'ENTJ': 'You are an ENTJ person for MBTI test. Your traits are as follows, Frank, decisive, assume leadership readily. Quickly see illogical and inefficient procedures and policies, develop and implement comprehensive systems to solve organizational problems. Enjoy long-term planning and goal setting. Usually well informed, well read, enjoy expanding their knowledge and passing it on to others. Forceful in presenting their ideas.'

    }
    api_key = "Your API KEY"
    # print("new")
    assistant = GPTAssistant(api_key)
    # print("here")
    # response = assistant.generate_programs()
    # print(response)
    response_list = []
    for personality, description in personality_desciption.items():
        for i in range(2):
            print("personality: ", personality)
            print(description)
            response = assistant.generate_programs(description)
            try:
                response = json.loads(response)
            except:
                response = assistant.generate_programs(description)
            response_list.append(response)

    save_path = "personality_mbti.json"
    combined_data = []
    cnt = 0
    for personality, description in personality_desciption.items():
        for i in range(2):
            entry = {
                "personality": personality,
                "description": description,
                "statement": response_list[cnt],
                # "statement_0": response_list[cnt],
                # "statement_1": response_list[cnt+1]
            }
            # print(response_list[cnt], response_list[cnt+1])
            cnt+=1

            combined_data.append(entry)
    with open(save_path, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)
    # print(response)
    # prompt = prompt.rstrip()

    # personas = ["Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    # key_words = {"Agreeableness": "appreciative, forgiving, generous, kind, and sympathetic",
    #             "Conscientiousness": "efficient, organized, planful, reliable, responsible, and thorough",
    #             "Neuroticism": "anxious, self-pitying, tense, touchy, unstable, and worrying",
    #             "Openness": "artistic, curious, imaginative, insightful, and original with wide interests"}
    #
    # response_list = []
    # for persona in personas:
    #
    #     persona_str = "Personality:\n" + persona + '\n'
    #     key_word = "Keyword for this personality:\n"+ key_words[persona] + '\n'
    #     question = persona_str + key_word
    #     response = assistant.generate_programs(system_prompt, prompt, question)
    #     response = json.loads(response)
    #     response_list.append(response)
    # combined_data = []
    # for i in range(len(personas)):
    #     entry = {
    #         "persona": personas[i],
    #         "keyword": key_words[personas[i]],
    #         "response": response_list[i],
    #     }
    #     combined_data.append(entry)
    # save_path = "personality.json"
    # with open(save_path, 'w') as json_file:
    #     json.dump(combined_data, json_file, indent=4)


if __name__ == '__main__':
    main()