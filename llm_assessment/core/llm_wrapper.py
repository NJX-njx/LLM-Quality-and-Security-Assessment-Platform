"""
LLM model wrapper for unified interface across different providers
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time


class BaseLLM(ABC):
    """Base class for LLM models"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
        self.call_count = 0
        self.total_tokens = 0
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens
        }


class MockLLM(BaseLLM):
    """Mock LLM for testing and demonstration"""
    
    def __init__(self, model_name: str = "mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        self.responses = kwargs.get("responses", {})
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        self.call_count += 1
        self.total_tokens += len(prompt.split())
        
        # Return predefined response or a generic one
        if prompt in self.responses:
            return self.responses[prompt]
        
        # Simulate thinking time
        time.sleep(0.1)
        
        # Generic responses based on prompt content
        if "safety" in prompt.lower() or "ethical" in prompt.lower():
            return "I aim to provide helpful, harmless, and honest responses."
        elif "calculate" in prompt.lower() or "what is" in prompt.lower():
            return "42"
        elif "write" in prompt.lower() or "generate" in prompt.lower():
            return "Here is a sample response to your request."
        else:
            return "I understand your question. Here is my response."
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion with mock response"""
        self.call_count += 1
        
        # Get the last user message
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "")
                break
        
        return self.generate(last_message, **kwargs)


class OpenAILLM(BaseLLM):
    """OpenAI LLM wrapper"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        try:
            import openai
            self.client = openai.OpenAI(api_key=kwargs.get("api_key"))
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI"""
        self.call_count += 1
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion using OpenAI"""
        self.call_count += 1
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        
        self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content


def create_llm(provider: str = "mock", **kwargs) -> BaseLLM:
    """Factory function to create LLM instances"""
    
    providers = {
        "mock": MockLLM,
        "openai": OpenAILLM,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider](**kwargs)
