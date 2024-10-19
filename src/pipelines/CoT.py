import textwrap
import logging
from typing import Any, Dict
from guidance import gen, system, user, assistant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

# Logger setup for debugging
_logger = logging.getLogger(__file__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())

class CoTPipeline:
    def __init__(self):
        # Initialize GPT-4 model for CoT
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Memory to summarize long conversations while retaining recent exact messages
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=500, verbose=True)

    def diagnosis_guidance(self, symptoms: str):
        """
        Generates step-by-step guidance using the LLM to provide possible diagnoses and follow-up questions.
        """
        prompt = f"""
        Based on the symptoms provided by the user in the following text: {symptoms}, think step by step and provide possible diagnoses across multiple departments.
        Then, ask follow-up questions that would help distinguish between these diagnoses.

        If the user has provided more information, use that to refine the diagnoses and ask more specific follow-up questions, by also considering 
        previous information in the memory: {self.memory.load_memory_variables({})}.

        Upon the user's response, you also need to see if you can rule out certain diagnoses based on the user's answers.
        Your goal is to shorten the list of possible diagnoses and ask the most relevant questions to reach a final diagnosis.
        If you feel confident about a specific diagnosis, provide a final diagnosis and end the conversation.
        """
        response = self.llm.predict(prompt)
        self.memory.save_context({"input": symptoms}, {"output": response})
        return response

    def generate_diagnosis(self, symptoms: str):
        """
        Generate the diagnosis based on user symptoms, asking for follow-up questions and providing a final diagnosis.
        """
        # Step 1: Generate possible diagnoses
        result = self.diagnosis_guidance(symptoms)

        _logger.debug(f"Diagnosis result: {result}")

        # Return the interaction
        return {
            "diagnosis_result": result
        }
