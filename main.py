from src.agent import answer

def main():
    print("Empório da Música — assistant prototype (type 'sair' to quit)")
    while True:
        message = input("\nVocê: ").strip()
        if message.lower() in {"sair", "exit", "quit"}:
            print("Até mais! 🎵")
            break
        print("\nAssistente:", answer(message))

if __name__ == "__main__":
    main()
