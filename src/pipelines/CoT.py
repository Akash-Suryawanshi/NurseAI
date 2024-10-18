from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationSummaryBufferMemory

class CoTPipeline:
    def __init__(self):
        # Initialize GPT-4 model
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Initialize ConversationSummaryBufferMemory for summary-based memory
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=500)  # Adjust token limit as needed

    def generate_response(self, query: str):
        """
        Generates a response with a Chain-of-Thought (CoT) approach, using summary-based memory to retain conversation history.
        """
        # Get the current conversation context from memory
        context = self.memory.load_memory_variables({})["history"]

        # Chain-of-Thought Prompt Engineering
        cot_prompt = (
            "You are a medical assistant with expert-level reasoning in diagnosis, prognosis, and treatment "
            "across multiple medical departments (e.g., neurology, cardiology, oncology) and diseases. "
            "For the following medical inquiry, provide a step-by-step explanation that thoroughly examines potential diagnoses, "
            "offers prognoses, and suggests appropriate treatments. Integrate relevant knowledge from various departments as needed.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            "Step-by-step reasoning and insights from multiple departments (where applicable):"
        )


        # Send the CoT prompt to the LLM
        messages = [HumanMessage(content=cot_prompt)]
        response = self.llm(messages)

        # Save the new interaction (query + response) in memory with summaries
        self.memory.save_context({"input": query}, {"output": response.content})

        return response
