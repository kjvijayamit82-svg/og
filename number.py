import random

def play_game():
    number = random.randint(1, 100)
    attempts = 0
    max_attempts = 7

    print("\n🎮 Welcome to Number Guessing Game!")
    print("Guess a number between 1 and 100")
    print(f"You have {max_attempts} attempts\n")

    while attempts < max_attempts:
        try:
            guess = int(input("Enter your guess: "))
        except ValueError:
            print("❌ Please enter a valid number!")
            continue

        attempts += 1

        if guess == number:
            print(f"🎉 Correct! You guessed it in {attempts} attempts.")
            break
        elif guess < number:
            print("📉 Too low!")
        else:
            print("📈 Too high!")

        print(f"Attempts left: {max_attempts - attempts}\n")

    else:
        print(f"😢 Game Over! The correct number was {number}")

# Main loop for replay
while True:
    play_game()
    again = input("\nDo you want to play again? (yes/no): ").lower()
    if again != "yes":
        print("👋 Thanks for playing!")
        break