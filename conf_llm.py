class LLMProcessor:
    """
    A class to handle LLM model loading and inference.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "GPU"

    def load_model(self, llm_model_path, device="GPU"):
        """
        Load the LLM model and tokenizer.
        
        Args:
            llm_model_path (str): Path to the LLM model
            device (str): Device to run the model on ('GPU' or 'CPU')
        """
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoConfig, AutoTokenizer
        self.device = device
        try:
            self.model = OVModelForCausalLM.from_pretrained(
                llm_model_path,
                device=device,
                config=AutoConfig.from_pretrained(llm_model_path),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, instruction, input_text, lang='Chinese'):
        """
        Run prediction with the loaded model.
        
        Args:
            instruction (str): System instruction for the model
            input_text (str): Input text for processing
            
        Returns:
            str: Model's response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Please call load_model first.")

        # Prepare messages
        sysprompt_cn = "你是一个会议书记员，"
        sysprompt_en = "You are a meeting secretary，"
        if lang == 'Chinese':
            sysprompt = sysprompt_cn
        else:
            sysprompt = sysprompt_en
        
        instruction = sysprompt + instruction

        # Sometimes input_text is a string, sometimes a list of messages
        if isinstance(input_text, str):
            messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]
        elif isinstance(input_text, list):
            messages = input_text
        else:
            raise ValueError("input_text must be a string or a list of messages")

        # Process input
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        attention_mask = model_inputs.attention_mask

        # Generate response
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Process output
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]        
        return response, len(model_inputs.input_ids[0]), len(generated_ids[0])

if __name__ == "__main__":
    # Example usage
    model_path = r'.\models\Qwen3-1.7B'
    test_texts = {
        'instruction': "你是一个会议书记员，能够根据用户提问给出会议中的具体内容或者会议总结。",
        'input': "书记员，请给出下面的会议记录的总结，控制在30字以内：人物穿着蓝色衬衫，可能是一个员工或学生。桌子上的物品可能用于工作或学习，比如咖啡杯和水瓶。背景中的黑包和衣架可能表示这个人在工作或学习时的着装和物品。白板和磁铁可能用于展示或记录信息。荧光灯营造出明亮的环境，可能用于需要集中注意力的场合？"
    }
    
    # Initialize and load model
    processor = LLMProcessor()
    if processor.load_model(model_path, device="GPU"):
        # Run inference
        response = processor.predict(
            instruction=test_texts['instruction'],
            input_text=test_texts['input']
        )
        print(response)
    else:
        print("Failed to load model")