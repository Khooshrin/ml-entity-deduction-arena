from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Answerer and Guesser models and tokenizers
answerer_model_name = "gpt2"  # Replace with a suitable model for answerer
guesser_model_name = "gpt2"   # Replace with a suitable model for guesser

answerer_model = AutoModelForCausalLM.from_pretrained(answerer_model_name)
answerer_tokenizer = AutoTokenizer.from_pretrained(answerer_model_name)

guesser_model = AutoModelForCausalLM.from_pretrained(guesser_model_name)
guesser_tokenizer = AutoTokenizer.from_pretrained(guesser_model_name)

# Set the device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
answerer_model.to(device)
guesser_model.to(device)

# Load list of hidden objects from file
with open("data/things/list_of_things_eval.txt", "r") as f:
    hidden_objects = [line.strip() for line in f if line.strip()]
print(hidden_objects)
# Hyperparameter for score calculation
lambda_param = 0.02

# Evaluation metrics initialization
total_turns = 0
total_successes = 0
total_yes_answers = 0
num_games = len(hidden_objects)

def answer_question(question, hidden_object, conversation_history):
    """
    Generate a yes/no/maybe answer based on the hidden object and question.
    """
    prompt = f"The object that you are thinking of is {hidden_object}. The prior conversation is: {conversation_history} The last question asked by the Guesser is {question}.\nAnswer it as if you are the answerer using yes/no/maybe as the only answers."
    inputs = answerer_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = answerer_model.generate(**inputs, max_new_tokens=50)
    raw_answer = answerer_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    
    # Simplify answer to yes/no/maybe
    if "yes" in raw_answer.lower():
        answer = "Yes"
    elif "no" in raw_answer.lower():
        answer = "No"
    else:
        answer = "Maybe"
    
    return answer

def is_valid_question(question):
    """
    Check if the question can be answered with yes/no/maybe.
    """
    return question.lower().startswith(("is", "does", "can", "has", "have"))

def generate_valid_question(conversation_history):
    """
    Generate a question that conforms to the yes/no/maybe format.
    """
    while True:
        # Generate a new question based on the ongoing conversation history
        question_prompt = f"{conversation_history} Since you are the Guesser, ask a question that can be answered using yes/no/maybe only."
        inputs = guesser_tokenizer(question_prompt, return_tensors="pt").to(device)
        outputs = guesser_model.generate(**inputs, max_new_tokens=50)
        question = guesser_tokenizer.decode(outputs[0], skip_special_tokens=True).replace(question_prompt, "").strip()
        print(question)
        
        if is_valid_question(question):
            return question
        else:
            print("Generated an invalid question. Regenerating...")

def play_game(hidden_object):
    """
    Play the 20 Questions game with a specific hidden object and track evaluation metrics.
    """
    conversation_history = f"We are playing the 20 Questions game where there is a guesser and an answerer. The guesser guesses the hidden object while the answerer thinks of the hidden object. The goal of the guesser is to be able to guess the hidden object in 20 or lesser questions. The guesser is only allowed to ask questions that can be answered using yes/no/maybe by the answerer. \n"
    num_turns = 0
    num_yes_answers = 0
    success = False
    
    for i in range(20):  # limit to 20 questions
        num_turns += 1
        
        # Generate a yes/no/maybe question from the Guesser
        question = generate_valid_question(conversation_history)
        print(f"Guesser: {question}")
        
        # Get the answer from the Answerer
        answer = answer_question(question, hidden_object, conversation_history)
        print(f"Answerer: {answer}")
        
        # Update conversation history and count "Yes" answers
        conversation_history += f"Guesser: {question}\nAnswerer: {answer}\n"
        if answer == "Yes":
            num_yes_answers += 1

        # Check if the guesser makes a correct guess
        if ("is it" in question.lower() or "i think it is" in question.lower()) and hidden_object.lower() in question.lower():
            print("Guesser has guessed the correct object!")
            success = True
            break

    # Calculate game score based on success and #Turns
    if success:
        score = 1 - lambda_param * max(num_turns - 5, 0)
    else:
        score = 0

    return num_turns, success, num_yes_answers, score

# Run the game for each hidden object in the list and calculate evaluation metrics
for hidden_object in hidden_objects:
    print(hidden_object)
    turns, success, yes_answers, score = play_game(hidden_object)
    
    print(hidden_object)
    # Update total metrics
    total_turns += turns
    total_successes += int(success)
    total_yes_answers += yes_answers

# Calculate final evaluation metrics
average_turns = total_turns / num_games
success_rate = total_successes / num_games * 100  # percentage
average_yes_answers = total_yes_answers / num_games
average_score = sum([play_game(obj)[3] for obj in hidden_objects]) / num_games  # calculating score across all games

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Average #Turns: {average_turns:.2f}")
print(f"Success Rate: {success_rate:.2f}%")
print(f"Average #Yes: {average_yes_answers:.2f}")
print(f"Average Score: {average_score:.2f}")
