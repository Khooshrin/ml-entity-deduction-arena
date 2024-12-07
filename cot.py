from anthropic import AI_PROMPT, HUMAN_PROMPT
import torch
import openai
import numpy as np
from scipy.stats import entropy
import re

class Node:
    def __init__(
        self,
        parent = None,
        children = []
    ) -> None:
        self.parent = parent
        self.children = children

    def add_children(self, children):
        self.children = self.children + children

    def set_parent(self, parent):
        self.parent = parent

class GuesserNode(Node):
    def __init__(
        self,
        parent = None,
        children = [],
        messages = [],
        reduction_score = None,
    ) -> None:
        super().__init__(parent=parent, children=children)
        self.messages = messages
        self.reduction_score = reduction_score

class AnswererNode(Node):
    def __init__(self, value, parent=None, children=[], messages = [], top_n = []):
        super().__init__(parent, children)
        self.value = value
        self.messages = messages
        self.top_n = top_n

class GuesserModel:
    def make_guess(self, messages):
        pass
    def answer_prompt(self, prompt, max_tokens):
        pass

class ClaudeGuesser(GuesserModel):
    def __init__(
        self,
        model,
        anthropic_api,
        temperature,
        max_tokens_to_sample = 256
    ):
        self.model = model
        self.anthropic_api = anthropic_api
        self.temperature = temperature
        self.max_tokens_to_sample = max_tokens_to_sample

    def make_guess(self, messages):
        prompt = ""
        for item in messages:
            if item["role"].upper() == "USER":
                prompt += f"{HUMAN_PROMPT} {item['content']}"
            elif item["role"].upper() == "ASSISTANT":
                prompt += f"{AI_PROMPT} {item['content']}"

        prompt += f"{AI_PROMPT}"

        return self.answer_prompt(prompt, 256)

    def answer_prompt(self, prompt, max_tokens):
        completion = self.anthropic_api.completions.create(
            model=self.model,
            max_tokens_to_sample=max_tokens,
            prompt=prompt,
            temperature=self.temperature,
        )

        return completion.completion.lstrip()

class HuggingFaceGuesser(GuesserModel):
    def __init__(
        self,
        guesser_prompt,
        tokenizer,
        model,
        kargs
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.kargs = kargs
        self.guesser_prompt = guesser_prompt

    def make_guess(self, messages):
        prompt = self.dialog_history(messages) + " ASSISTANT:"
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)]
        )  # TODO check if huggingface is using the same format.
        input_ids = input_ids.to(self.model.base_model.device)
        attention_mask = None

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.kargs,
            )
            gen_str = (
                self.tokenizer.decode(gen[0][input_ids[0].shape[0] :])
                .split("</s>")[0]
                .split("USER")[0]
                .lstrip()
                .strip()
            )


            return gen_str
    def answer_prompt(self, prompt, max_tokens = None):
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)]
        )  # TODO check if huggingface is using the same format.
        input_ids = input_ids.to(self.model.base_model.device)
        attention_mask = None

        kargs_copy = self.kargs.copy()

        if max_tokens:
            kargs_copy["max_new_tokens"] = max_tokens

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                max_new_tokens = 300,
#                **self.kargs,
            )
            gen_str = (
                self.tokenizer.decode(gen[0],skip_special_tokens=True)
            )


            return gen_str

    def dialog_history(self, messages):
        history = self.guesser_prompt + " "
        for item in messages:
            if item["role"].upper() == "USER":
                history += "USER: " + item["content"]
            elif item["role"].upper() == "ASSISTANT":
                history += " " + "ASSISTANT: " + item["content"] + "</s>"
        return history

class OpenAIGuesser(GuesserModel):
    def __init__(
        self,
        api_base,
        model,
        temperature,
        max_tokens = 64,
        n = 1,
        stop = None,
    ):
        self.api_base = api_base
        self.model = model
        self.temperature = temperature,
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        openai.api_base = self.api_base

    def make_guess(self, messages):
        return self.answer_prompt(self, messages, self.max_tokens)

    def answer_prompt(self, prompt, max_tokens):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=prompt,
            max_tokens=max_tokens,
            n=self.n,
            stop=self.stop,
            temperature=self.temperature,
        )
        return response.choices[0].message.to_dict()["content"].strip()


class COT:
    def __init__(
        self,
        item: str,
        guesser_model,
        guesser_model_name: str,
        num_turns: int,
        temperature: float,
        openai_api: bool,
        guesser_tokenizer,
        guesser_kargs,
        vicuna_prompt: str,
        guesser_api_base: str,
        anthropic_api,
        cot_kargs={},
    ) -> None:
        self.item = item
        self.guesser_model = self.create_guesser(openai_api, guesser_model_name, guesser_model, guesser_tokenizer, guesser_api_base, guesser_kargs, temperature, vicuna_prompt, anthropic_api)
        if "depth" in cot_kargs:
            self.depth = cot_kargs["depth"]
        else:
            self.depth = 1

        if "width" in cot_kargs:
            self.width = cot_kargs["width"]
        else:
            self.width = 5

        if "top_n" in cot_kargs:
            self.top_n = cot_kargs["top_n"]
        else:
            self.top_n = 10
        self.num_turns = num_turns
        self.inference_count = 0
        self.recent_top_n = []

    def guess(self, messages):
        cnt = 0
        while(True):
            cnt += 1
            tree = self.generate_tree(messages)
            response = self.choose_branch(tree)
            if response != "Unable to determine a confident next question.":
                return {
                    "role": "assistant",
                    "content": response,
                    "tree": tree,
                    "inference_count": self.inference_count
                }
            if cnt > 10:
                print("Unable to determine a confident next question.")
                exit()
    def generate_tree(self, messages):
        isGuesser = False
        root = AnswererNode(value="Root", messages=messages, top_n=self.generate_top_n(messages))
        print("Root's Top_n: ", root.top_n)
        queue = [root]

        for i in range(self.depth):
            queue_length = len(queue)
            isGuesser = not isGuesser
            for j in range(queue_length):
                node = queue.pop()
                child_list = []
                if isGuesser:
                    guesses = self.generate_guesses(messages, node, self.width)
                    print("GUESSES: ", (i, j), " ", guesses)
                    for guess in guesses:
                        child = GuesserNode(messages=messages + [guess], parent=node)
                        queue.append(child)
                        child_list.append(child)
                elif i != self.depth - 1:
                    for item in ["Yes", "No", "Maybe"]:
                        answer = self.generate_answer(item)
                        child = AnswererNode(value=item, messages = messages + [answer], top_n=self.generate_top_n(messages + [answer]), parent=node)
                        print("TOP_N: ", (i, j), " ", child.top_n)
                        child_list.append(child)
                else:
                    continue
                node.add_children(child_list)


        return root

    def choose_branch(self, tree):
        # TODO: Given a tree of guesser and answerer nodes, choose one child of the root node and return a response
        def count_unique_candidates(node):
            """Helper function to count unique candidates in the subtree rooted at `node`."""
            queue = [node]
            unique_candidates = set()
            while queue:
                current_node = queue.pop(0)
                if isinstance(current_node, AnswererNode):
                    unique_candidates.update(current_node.top_n)
                queue.extend(current_node.children)
            return len(unique_candidates)

        potential_questions = []

        for child in tree.children:
            potential_questions.append(child.messages[-1]["content"])

        best_question = self.choose_question_based_on_reduction(potential_questions, tree.messages)

        best_branch = None
        for child in tree.children:
            if child.messages[-1]["content"] == best_question:
                best_branch = child
                break

        return best_branch.messages[-1]["content"] if best_branch else "Unable to determine a confident next question."

    def create_guesser(self, openai_api, guesser_model_name, guesser_model, guesser_tokenizer, guesser_api_base, guesser_kargs, temperature, vicuna_prompt, anthropic_api):
        if guesser_model_name.startswith(
            "claude"
        ):
            return ClaudeGuesser(guesser_model, anthropic_api=anthropic_api, temperature=temperature)
        elif openai_api:
            return OpenAIGuesser(api_base=guesser_api_base, model=guesser_model, temperature=temperature)
        else:
            return HuggingFaceGuesser(vicuna_prompt, guesser_tokenizer, guesser_model, guesser_kargs)


    def generate_guesses(self, messages, node, k):
        # TODO: Given a set of messages and a parent node generate a guess. Make sure to decide whether to guess an entity or ask a question in this function
        # based on the previous top-n candidates of the ancestor answerer nodes and whether this is the last guess or not
        if self.recent_top_n and len(self.recent_top_n) > 5:
            for candidate in node.top_n:
                if all(candidate in recent_top for recent_top in self.recent_top_n[-5:]):
                    guess_content = f"Is it a/an {candidate}?"
                    self.inference_count += 1
                    return [{"role": "assistant", "content": guess_content}]

        def generate_20_questions_prompt(candidates):
            prompt = prompt = "We are playing a 20 Questions game where you must deduce the hidden entity. Following is the conversation history:\n" + self.guesser_model.dialog_history(messages) + "\nFollowing are the possible entities: " + str(", ".join(candidates)) + ".\nGenerate 10 questions that are not a part of the conversation history and can be answered only using yes, no or maybe as a numbered list.\nASSISTANT:"
            return prompt
        
        def extract_questions(output):
            # Split the output into lines
            lines = output.split("ASSISTANT:")[-1]
            print(f"ASSISTANT: Split: {lines}")
            lines = lines.split("\n")
            print(f"New Line Split: {lines}")
            questions = []

            # Iterate through lines to extract questions
            for line in lines:
                # Match lines that start with a number followed by a period and a space
                if line.strip() and line.strip()[0].isdigit() and line.strip()[1] == '.':
                    # Extract the question by removing the numbering
                    question = line.strip()[2:].strip()
                    questions.append(question)

            return questions

        prompt = generate_20_questions_prompt(node.top_n)
        print(f"Prompt: {prompt}")
        potential_questions = self.guesser_model.answer_prompt(prompt, 1000)
        print(f"Potential Questions: {potential_questions}")
        potential_questions = extract_questions(potential_questions)
        print(f"Questions Only: {potential_questions}")
        # Ensure unique questions
        potential_questions = list(set(potential_questions))[:k]  # Limit to 10 unique questions
        return [{"role": "assistant", "content": question} for question in potential_questions]

    # def choose_question_based_on_reduction(self, potential_questions, messages):
    #     best_question = None
    #     max_reduction = -1

    #     for question in potential_questions:
    #         reduction_score = 0  # Track subspace reduction for this question

    #         # For each possible answer, simulate how it would impact subspace reduction
    #         for answer in ["Yes", "No", "Maybe"]:
    #             simulated_messages = messages + [{"role": "assistant", "content": question}, {"role": "user", "content": answer}]
    #             top_n_candidates = self.generate_top_n(simulated_messages)

    #             # Calculate a reduction score based on the uniqueness of the remaining candidates
    #             reduction_score += len(set(top_n_candidates))

    #         # If this question has the highest reduction score, select it as the best
    #         if reduction_score > max_reduction:
    #             max_reduction = reduction_score
    #             best_question = question

    #     return best_question

    def choose_question_based_on_reduction(self, potential_questions, messages):
        dialog_history = self.guesser_model.dialog_history(messages)
    
        # Ensure the selected question is not part of the dialog history
        potential_questions_sorted = sorted(
            potential_questions,
            key=lambda q: sum(len(set(self.generate_top_n(messages + [{"role": "assistant", "content": q}, {"role": "user", "content": a}])))
                              for a in ["Yes", "No", "Maybe"]),
            reverse=True
        )
        # Find the next best question not in dialog history
        print(f"Sorted questions: {potential_questions_sorted}")
        for question in potential_questions_sorted:
            if question not in dialog_history:
                print(f"Selected question: {question}")
                return question
    
        return None
    

    def add_reduction_score(self, guess_node: GuesserNode, answerer_node: AnswererNode, base = None):
        top_n = answerer_node.top_n
        answers = []

        for elem in top_n:
            answer = self.guesser_model.answer_prompt(f"Q: For ${elem}, answer the question: ${guess_node.messages[-1]['content']} with Yes/No/Maybe \n A: ", 6)
            answers.append(answer)

        values, counts = np.unique(answers, return_counts=True)
        total = sum(counts)
        guess_node.reduction_score = {}

        for i in range(len(values)):
            guess_node.reduction_score[values[i]] = counts[i] / total





    def generate_answer(self, response):
        # TODO: Given a Yes/No/Maybe response, return a formatted answer
        return {"role": "assistant", "content": response}

    def generate_top_n(self, messages):
        # TODO: Given a set of messages, generate the top n best candidates for the entity
        unique_candidates = set()  # Use a set to keep only unique entities
        def extract_entities(output):
            # Split the output into lines
            lines = output.split("ASSISTANT:")[-1]
            lines = lines.split("\n")
            questions = []

            # Iterate through lines to extract questions
            for line in lines:
                # Match lines that start with a number followed by a period and a space
                if line.strip() and line.strip()[0].isdigit() and line.strip()[1] == '.':
                    # Extract the question by removing the numbering
                    question = line.strip()[2:].strip()
                    questions.append(question)

            return questions

        # Determine which model is being used and query it for top entities
        if isinstance(self.guesser_model, ClaudeGuesser):
            # Claude model execution
            prompt = self.guesser_model.dialog_history(messages) + "Above is the game conversation till now. Predict the unique top " + str(self.top_n) + " entities."
            try:
                completion = self.guesser_model.anthropic_api.completions.create(
                    model=self.guesser_model.model,
                    prompt=prompt,
                    max_tokens_to_sample=64,
                    temperature=self.guesser_model.temperature,
                )
                # Split lines and add unique entries to the set
                unique_candidates.update(completion.completion.splitlines())
            except Exception as e:
                print(f"Claude API error: {e}")

        elif isinstance(self.guesser_model, OpenAIGuesser):
            # OpenAI model execution
            openai.api_base = self.guesser_model.api_base
            try:
                prompt = self.guesser_model.dialog_history(messages) + "Above is the game conversation till now. Predict the unique top " + str(self.top_n) + " entities."
                response = openai.ChatCompletion.create(
                    model=self.guesser_model.model,
                    messages=messages + [{"role": "system", "content": prompt}],
                    max_tokens=self.guesser_model.max_tokens,
                    temperature=self.guesser_model.temperature,
                )
                openai_candidates = response.choices[0].message["content"].strip().splitlines()
                unique_candidates.update(openai_candidates)
            except Exception as e:
                print(f"OpenAI API error: {e}")

        elif isinstance(self.guesser_model, HuggingFaceGuesser):
            # Hugging Face model execution
            prompt = self.guesser_model.dialog_history(messages) + " The previous text contains the conversation history of the game. Predict the unique top " + str(self.top_n) + " entities most likely to be the hidden entity as a numbered list.\nASSISTANT:"
            input_ids = torch.tensor([self.guesser_model.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.guesser_model.model.base_model.device)

            try:
                with torch.no_grad():
                    gen = self.guesser_model.model.generate(input_ids=input_ids, max_new_tokens=200, temperature=0.8, repetition_penalty=1.0, do_sample=True)
                    huggingface_candidates = self.guesser_model.tokenizer.decode(gen[0], skip_special_tokens=True)
                    print(f"Response to Generate Entities: {huggingface_candidates}")
                    huggingface_candidates = extract_entities(huggingface_candidates)
                    print(f"Entities Only: {huggingface_candidates}")
                    unique_candidates.update(huggingface_candidates)
            except Exception as e:
                print(f"Hugging Face API error: {e}")

        # Convert unique set to a sorted list and limit to top_n
        top_n = list(unique_candidates)[:self.top_n]
        self.recent_top_n.append(top_n)
        if len(self.recent_top_n) > 5:
            self.recent_top_n.pop(0)

        return top_n

    def calculate_probs(self, node: GuesserNode, base = None):
        if len(node.children) == 0:
            return list(node.reduction_score.values())

        final_scores = []
        for child in node.children:
            for grandchild in child.children:
                final_scores += [node.reduction_score[child.value] * x for x in list(self.calculate_probs(grandchild))]
        print(f"Final Scores: {final_scores}")

        return final_scores