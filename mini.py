from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def ask_llm(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for h in history:
        messages.append(h)
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # lightweight & fast
        messages=messages
    )
    return resp.choices[0].message.content

history = []

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    reply = ask_llm(user, history)
    print("Bot:", reply)

    history.append({"role": "user", "content": user})
    history.append({"role": "assistant", "content": reply})