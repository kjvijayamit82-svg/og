from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
questions = [
    "hi", "hello", "how are you",
    "bye", "goodbye",
    "i am also fine",
    "what is your name",
    "tell me about india",
    "tell me about ipl"
]

answers = [
    "Hello!", 
    "Hi there!", 
    "I am fine! What about you?",
    "Bye!", 
    "See you!",
    "That's great to hear!",   # ✅ FIXED (added missing answer)
    "I am a chatbot",
    "India is a great place to visit and enjoy culture and food.",
    "CSK won the match against DC by 8 wickets."
]

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# Train
model = MultinomialNB()
model.fit(X, answers)

# Chat loop
while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    user_vec = vectorizer.transform([user])
    response = model.predict(user_vec)

    print("Bot:", response[0])