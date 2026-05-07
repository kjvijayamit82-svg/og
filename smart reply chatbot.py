import random

print("🤖 Smart Reply Chatbot (Advanced Version)")
print("Type 'bye' to exit\n")

# Memory storage
user_name = None

# Response database
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! Nice to talk to you 😊"],
    "how_are_you": ["I'm just code, but I'm running fine!", "Doing great! Thanks for asking 😊"],
    "help": ["You can ask me about myself or say hi!", "Try asking: how are you, what is your name"],
    "unknown": ["Sorry, I didn't understand that.", "Can you rephrase it?", "I'm still learning 🤖"]
}

while True:
    user_input = input("You: ").lower()

    # Exit condition
    if user_input == "bye":
        print("Bot: Goodbye! Have a great day 👋")
        break

    # Greeting detection
    if "hello" in user_input or "hi" in user_input:
        print("Bot:", random.choice(responses["greeting"]))

    # How are you
    elif "how are you" in user_input:
        print("Bot:", random.choice(responses["how_are_you"]))

    # Name memory feature
    elif "my name is" in user_input:
        user_name = user_input.replace("my name is", "").strip()
        print(f"Bot: Nice to meet you, {user_name} 😊")

    elif "what is my name" in user_input:
        if user_name:
            print(f"Bot: Your name is {user_name}")
        else:
            print("Bot: I don't know your name yet. Tell me!")

    # Help section
    elif "help" in user_input:
        print("Bot:", random.choice(responses["help"]))

    # Default response
    else:
        print("Bot:", random.choice(responses["unknown"]))