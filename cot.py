from anthropic import AI_PROMPT, HUMAN_PROMPT
import torch
import openai


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
        messages = []
    ) -> None:
        super().__init__(parent=parent, children=children)
        self.messages = messages

class AnswererNode(Node):
    def __init__(self, parent=None, children=[], messages = [], top_n = []):
        super().__init__(parent, children)
        self.messages = []
        self.top_n = []

class GuesserModel:
    def make_guess(self, messages):
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

        completion = self.anthropic_api.completions.create(
            model=self.model,
            max_tokens_to_sample=256,
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

    def make_guess(self, messages):
        openai.api_base = self.api_base
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
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
            self.depth = 3
        
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
        tree = self.generate_tree(messages)
        response = self.choose_branch(tree)
        return {
            "role": "assistant",
            "content": response,
            "tree": tree,
            "inference_count": self.inference_count
        }
    
    def generate_tree(self, messages):
        isGuesser = False 
        root = AnswererNode(messages=messages, top_n=self.generate_top_n(messages))
        queue = [root]

        for i in range(self.depth):
            queue_length = len(queue)
            isGuesser = not isGuesser
            for j in range(queue_length):
                node = queue.pop()
                child_list = []
                if isGuesser:
                    for k in range(self.width):
                        guess = self.generate_guess(messages, node)
                        child = GuesserNode(messages=messages + [guess], parent=node)
                        queue.append(child)
                        child_list.append(child)
                else:
                    for item in ["Yes", "No", "Maybe"]:
                        answer = self.generate_answer(item)
                        child = AnswererNode(messages = messages + [answer], top_n=self.generate_top_n(messages + [answer]), parent=node)
                        child_list.append(child)
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
        
        best_branch = None
        min_candidate_count = float('inf')
        
        for child in tree.children:
            candidate_count = count_unique_candidates(child)
            if candidate_count < min_candidate_count:
                min_candidate_count = candidate_count
                best_branch = child

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
        

    def generate_guess(self, messages, node):
        # TODO: Given a set of messages and a parent node generate a guess. Make sure to decide whether to guess an entity or ask a question in this function 
        # based on the previous top-n candidates of the ancestor answerer nodes and whether this is the last guess or not
        if self.recent_top_n:
            for candidate in node.top_n:
                if all(candidate in recent_top for recent_top in self.recent_top_n[-5:]):
                    guess_content = f"Is it a/an {candidate}?"
                    self.inference_count += 1
                    return {"role": "assistant", "content": guess_content}
        
        prompt = "Based on the conversation so far, generate the top 10 questions that can help deduce the entity. Only ask questions that can be answered with 'yes', 'no', or 'maybe'."
        potential_questions = self.guesser_model.make_guess([{"role": "system", "content": prompt}] + messages).splitlines()
    
        # Ensure unique questions
        potential_questions = list(set(potential_questions))[:10]  # Limit to 10 unique questions

        # Simulate answers for each question and evaluate their effectiveness
        best_question = None
        max_reduction = -1

        for question in potential_questions:
            reduction_score = 0  # Track subspace reduction for this question

            # For each possible answer, simulate how it would impact subspace reduction
            for answer in ["Yes", "No", "Maybe"]:
                simulated_messages = messages + [{"role": "assistant", "content": question}, {"role": "user", "content": answer}]
                top_n_candidates = self.generate_top_n(simulated_messages)

                # Calculate a reduction score based on the uniqueness of the remaining candidates
                reduction_score += len(set(top_n_candidates))

            # If this question has the highest reduction score, select it as the best
            if reduction_score > max_reduction:
                max_reduction = reduction_score
                best_question = question

        # Return the selected question as the next question to ask
        return {"role": "assistant", "content": best_question or "Unable to determine the next best question."}

    def generate_answer(self, response):
        # TODO: Given a Yes/No/Maybe response, return a formatted answer
        return {"role": "assistant", "content": response}

    def generate_top_n(self, messages):
        # TODO: Given a set of messages, generate the top n best candidates for the entity
        unique_candidates = set()  # Use a set to keep only unique entities

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
            prompt = self.guesser_model.dialog_history(messages) + "Above is the game conversation till now. Predict the unique top " + str(self.top_n) + " entities."
            input_ids = torch.tensor([self.guesser_model.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.guesser_model.model.base_model.device)

            try:
                with torch.no_grad():
                    gen = self.guesser_model.model.generate(input_ids=input_ids, max_length=64, **self.guesser_model.kargs)
                    huggingface_candidates = self.guesser_model.tokenizer.decode(gen[0], skip_special_tokens=True).splitlines()
                    unique_candidates.update(huggingface_candidates)
            except Exception as e:
                print(f"Hugging Face API error: {e}")

        # Convert unique set to a sorted list and limit to top_n
        top_n = list(unique_candidates)[:self.top_n]
        self.recent_top_n.append(top_n)
        if len(self.recent_top_n) > 5:
            self.recent_top_n.pop(0)

        return top_n
