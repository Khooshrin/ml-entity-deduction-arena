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
        guesser_model: str,
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
        self.guesser_model = self.create_guesser(openai_api, guesser_model, guesser_tokenizer, guesser_api_base, guesser_kargs, temperature, vicuna_prompt, anthropic_api)
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
                        child = AnswererNode(messages = messages + [answer], top_n=self.generate_top_n(messages + [answer], self.top_n), parent=node)
                        child_list.append(child)
                node.add_children(child_list)

        
        return root

    def choose_branch(self, tree):
        # TODO: Given a tree of guesser and answerer nodes, choose one child of the root node and return a response
        pass

    def create_guesser(self, openai_api, guesser_model, guesser_tokenizer, guesser_api_base, guesser_kargs, temperature, vicuna_prompt, anthropic_api):
        if isinstance(self.guesser_model, str) and self.guesser_model.startswith(
            "claude"
        ):
            return ClaudeGuesser(guesser_model, anthropic_api=anthropic_api, temperature=temperature)
        if not isinstance(self.guesser_model, str):
            return HuggingFaceGuesser(vicuna_prompt, guesser_tokenizer, guesser_model, guesser_kargs)
        
        if openai_api:
            return OpenAIGuesser(api_base=guesser_api_base, model=guesser_model, temperature=temperature)
        
        raise Exception()

    def generate_guess(self, messages, node):
        # TODO: Given a set of messages and a parent node generate a guess. Make sure to decide whether to guess an entity or ask a question in this function 
        # based on the previous top-n candidates of the ancestor answerer nodes and whether this is the last guess or not
        pass

    def generate_answer(self, response):
        # TODO: Given a Yes/No/Maybe response, return a formatted answer
        pass

    def generate_top_n(self, messages):
        # TODO: Given a set of messages, generate the top n best candidates for the entity
        pass