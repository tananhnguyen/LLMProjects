"""
LLM Agent core implementation.
Provides a modular agent framework for task execution using LLMs.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from transformers import AutoModelForCausalLM, AutoTokenizer


class AgentAction(Enum):
    """Enum for agent action types."""
    FINAL_ANSWER = "final_answer"
    SEARCH = "search"
    RETRIEVE = "retrieve"
    CALCULATE = "calculate"
    GENERATE = "generate"
    TRANSLATE = "translate"
    ANALYZE = "analyze"
    QUERY_PUBMED = "query_pubmed"


@dataclass
class AgentMemory:
    """Memory for the agent to store context and history."""
    
    # Conversation history
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Retrieved documents
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tool outputs
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Working memory (temporary storage)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
    def add_document(self, document: Dict[str, Any]):
        """Add a retrieved document to memory."""
        self.documents.append(document)
        
    def add_tool_output(self, tool_name: str, output: Any):
        """Add a tool output to memory."""
        if tool_name not in self.tool_outputs:
            self.tool_outputs[tool_name] = []
            
        self.tool_outputs[tool_name].append({
            "output": output,
            "timestamp": time.time()
        })
        
    def get_conversation_history(self, max_messages: Optional[int] = None) -> str:
        """Get formatted conversation history."""
        history = []
        
        messages = self.messages
        if max_messages is not None:
            messages = messages[-max_messages:]
            
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            history.append(f"{role.capitalize()}: {content}")
            
        return "\n".join(history)
        
    def get_relevant_documents(self, k: int = 3) -> str:
        """Get formatted relevant documents."""
        if not self.documents:
            return ""
            
        # Sort by relevance score if available
        sorted_docs = sorted(
            self.documents,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        # Take top k
        top_docs = sorted_docs[:k]
        
        # Format documents
        formatted_docs = []
        for i, doc in enumerate(top_docs):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            formatted_docs.append(f"Document {i+1} [Source: {source}]:\n{content}")
            
        return "\n\n".join(formatted_docs)
        
    def clear(self):
        """Clear all memory."""
        self.messages = []
        self.documents = []
        self.tool_outputs = {}
        self.working_memory = {}


@dataclass
class AgentTool:
    """Tool that the agent can use."""
    
    name: str
    description: str
    function: Callable
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool function."""
        return self.function(**kwargs)


class LLMAgent:
    """
    LLM-based agent for task execution.
    """
    
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        memory: Optional[AgentMemory] = None,
        tools: Optional[List[AgentTool]] = None,
        system_prompt: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the LLM agent.
        
        Args:
            model: Pre-loaded LLM model
            tokenizer: Pre-loaded tokenizer
            model_name: Name of the model to load (if model not provided)
            device: Device to run the model on
            memory: Agent memory instance
            tools: List of tools available to the agent
            system_prompt: System prompt for the agent
            logger: Logger instance
        """
        self.device = device
        
        # Set up model and tokenizer
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_name is not None:
            self.model, self.tokenizer = self._load_model(model_name)
        else:
            raise ValueError("Either model and tokenizer or model_name must be provided")
            
        # Set up memory
        self.memory = memory or AgentMemory()
        
        # Set up tools
        self.tools = tools or []
        
        # Set up system prompt
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initialized LLM agent with {len(self.tools)} tools")
        
    def _load_model(
        self,
        model_name: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and tokenizer.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device if self.device == "auto" else {"": self.device}
        )
        
        return model, tokenizer
        
    def _default_system_prompt(self) -> str:
        """
        Get the default system prompt.
        
        Returns:
            Default system prompt
        """
        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}" for tool in self.tools
        ])
        
        return f"""You are a helpful AI assistant with access to the following tools:

{tools_desc}

To use a tool, respond with:
```json
{{
  "action": "tool_name",
  "action_input": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

If you have a final answer, respond with:
```json
{{
  "action": "final_answer",
  "action_input": "Your final answer here"
}}
```

Always think step by step and use the appropriate tools when needed.
"""
        
    def add_tool(self, tool: AgentTool):
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        self.tools.append(tool)
        self.logger.info(f"Added tool: {tool.name}")
        
    def get_tool(self, tool_name: str) -> Optional[AgentTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
                
        return None
        
    def generate_prompt(
        self,
        query: str,
        include_history: bool = True,
        include_documents: bool = True,
        max_history_messages: Optional[int] = None
    ) -> str:
        """
        Generate a prompt for the LLM.
        
        Args:
            query: User query
            include_history: Whether to include conversation history
            include_documents: Whether to include retrieved documents
            max_history_messages: Maximum number of history messages to include
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [self.system_prompt]
        
        # Add conversation history
        if include_history and self.memory.messages:
            history = self.memory.get_conversation_history(max_history_messages)
            prompt_parts.append("# Conversation History\n" + history)
            
        # Add retrieved documents
        if include_documents and self.memory.documents:
            documents = self.memory.get_relevant_documents()
            prompt_parts.append("# Retrieved Information\n" + documents)
            
        # Add current query
        prompt_parts.append("# Current Query\nUser: " + query)
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
        
    def parse_response(self, response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Parse the LLM response to extract action and action input.
        
        Args:
            response: LLM response
            
        Returns:
            Tuple of (action_type, action_input)
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                action_data = json.loads(json_str)
                
                action = action_data.get("action")
                action_input = action_data.get("action_input")
                
                return action, action_input
                
            # Try to find JSON without code block markers
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                action_data = json.loads(json_str)
                
                action = action_data.get("action")
                action_input = action_data.get("action_input")
                
                return action, action_input
                
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse JSON from response: {e}")
            
        # If no valid JSON found, treat as final answer
        return "final_answer", response
        
    def execute_action(
        self,
        action: str,
        action_input: Any
    ) -> str:
        """
        Execute an action.
        
        Args:
            action: Action to execute
            action_input: Input for the action
            
        Returns:
            Result of the action
        """
        # Handle final answer
        if action == "final_answer":
            return action_input
            
        # Find the tool
        tool = self.get_tool(action)
        
        if tool is None:
            return f"Error: Tool '{action}' not found"
            
        # Execute the tool
        try:
            if isinstance(action_input, dict):
                result = tool.execute(**action_input)
            else:
                result = tool.execute(input=action_input)
                
            # Add tool output to memory
            self.memory.add_tool_output(action, result)
            
            return f"Tool {action} returned: {result}"
            
        except Exception as e:
            error_msg = f"Error executing tool '{action}': {str(e)}"
            self.logger.error(error_msg)
            return error_msg
            
    def run(
        self,
        query: str,
        max_iterations: int = 5
    ) -> str:
        """
        Run the agent on a query.
        
        Args:
            query: User query
            max_iterations: Maximum number of iterations
            
        Returns:
            Final answer
        """
        self.logger.info(f"Running agent on query: {query}")
        
        # Add query to memory
        self.memory.add_message("user", query)
        
        # Run iterations
        for i in range(max_iterations):
            self.logger.info(f"Iteration {i+1}/{max_iterations}")
            
            # Generate prompt
            prompt = self.generate_prompt(query)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Parse response
            action, action_input = self.parse_response(response)
            
            # Log action
            self.logger.info(f"Action: {action}")
            
            # Execute action
            result = self.execute_action(action, action_input)
            
            # Add result to memory
            if action == "final_answer":
                self.memory.add_message("assistant", result)
                return result
                
            # Add intermediate result to memory
            self.memory.add_message("system", f"Tool result: {result}")
            
        # If we reach max iterations without a final answer
        final_response = "I apologize, but I've reached the maximum number of iterations without finding a complete answer. Here's what I know so far: " + self.memory.get_conversation_history(3)
        self.memory.add_message("assistant", final_response)
        
        return final_response
        
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated response
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Adjust based on model context window
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        # Decode output
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
        
    def reset(self):
        """Reset the agent's memory."""
        self.memory.clear()
        self.logger.info("Agent memory reset")


# Import necessary modules
import re
import torch
