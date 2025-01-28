from langchain_ollama import OllamaLLM  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
import re
# Prompt template (remove unnecessary "thinking" part)
template = """
Here is the conversation history:
{context}

Question: {question}

Answer: Respond naturally and directly without overthinking.
"""

# Initialize the model and prompt template
model = OllamaLLM(model="deepseek-r1:1.5b")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""  # Conversation context
    print("Welcome to the AI ChatBot! Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("You: ")
            
            # Exit condition
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            # Process user input and generate response
            result = chain.invoke({"context": context, "question": user_input}).strip()

            # This is for clean output from the chatbot
            clean_result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()

            print("Bot:", clean_result)

            # Update context with the current interaction
            context += f"\nUser: {user_input}\nAI: {result}"

        except Exception as e:
            print("An error occurred:", e)

if __name__ == "__main__":
    handle_conversation()
