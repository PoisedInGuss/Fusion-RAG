"""
Incoming: prompt, context --- {str, List[Dict]}
Processing: LLM generation --- {1 job: API call}
Outgoing: generated answer --- {Dict with answer, metadata}

Generation Operations
---------------------
LLM generation via LM Studio API (OpenAI-compatible).

Supports multiple task types:
- QA: Answer extraction/generation from context
- FactVerification: 3-way claim classification (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO)

All prompts loaded from config/defaults.yaml - no hardcoding.

STRICT: No silent error handling. Connection failures raise exceptions.
"""

import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Import config - all prompts come from here
from src.config import config


class LMStudioConnectionError(Exception):
    """Raised when LM Studio is not running or unreachable."""
    pass


class GenerationError(Exception):
    """Raised when generation fails."""
    pass


class GenerationOperation:
    """
    LLM Generation via LM Studio API (OpenAI-compatible).
    
    LM Studio runs at localhost:1234 with OpenAI-compatible API.
    
    STRICT: Connection failures raise LMStudioConnectionError.
    """
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """
        Initialize generation operation.
        
        Args:
            base_url: LM Studio API base URL (from config if None)
            api_key: API key for authentication (optional)
        """
        self.base_url = base_url or config.models.lm_studio.base_url
        self.default_model = config.models.lm_studio.default_model
        self.timeout = config.models.lm_studio.timeout_seconds
        self.api_key = api_key
    
    def execute(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response via LM Studio API.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions (from config if None)
            model: Model identifier (from config if None)
            temperature: Sampling temperature (from config if None)
            max_tokens: Maximum tokens to generate (from config if None)
            
        Returns:
            Dict with answer, model, tokens_used, latency_ms, metadata
            
        Raises:
            LMStudioConnectionError: If LM Studio is not running.
            GenerationError: If API call fails.
        """
        # Get defaults from config
        system_prompt = system_prompt or config.generation.system_prompt
        model = model or self.default_model
        temperature = temperature if temperature is not None else config.generation.temperature
        max_tokens = max_tokens or config.generation.max_tokens
        
        start_time = time.time()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers=headers,
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise LMStudioConnectionError(
                f"LM Studio not running at {self.base_url}. "
                f"Start LM Studio and load a model first. Error: {e}"
            )
        except requests.exceptions.Timeout:
            raise GenerationError(f"LM Studio request timed out after {self.timeout}s")
        
        if response.status_code != 200:
            raise GenerationError(
                f"LM Studio API error {response.status_code}: {response.text}"
            )
        
        try:
            data = response.json()
        except ValueError as e:
            raise GenerationError(f"LM Studio returned invalid JSON: {e}")
        
        if "choices" not in data or not data["choices"]:
            raise GenerationError(f"LM Studio response missing choices: {data}")
        
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return {
            "answer": answer,
            "response": answer,
            "model": model,
            "tokens_used": usage.get("total_tokens", 0),
            "latency_ms": (time.time() - start_time) * 1000,
            "metadata": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "temperature": temperature
            }
        }


class ValidateOperation:
    """
    Atomic validation operation.
    
    Validates generated answers against context.
    """
    
    def __init__(self):
        pass
    
    def execute(
        self,
        answer: str,
        query: str,
        context: List[Dict[str, Any]],
        checks: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute answer validation.
        
        Args:
            answer: Generated answer to validate
            query: Original query
            context: Retrieved documents with 'content' field
            checks: Validation checks to run (hallucination, completeness, citation)
            
        Returns:
            Dict with is_valid, checks_passed, checks_failed, confidence, issues
            
        Raises:
            ValueError: If answer or context is empty.
        """
        start_time = time.time()
        
        if not answer:
            raise ValueError("Cannot validate empty answer")
        if not context:
            raise ValueError("Cannot validate without context")
        
        checks = checks or ['hallucination', 'completeness']
        
        checks_passed = []
        checks_failed = []
        issues = []
        
        if 'hallucination' in checks:
            answer_lower = answer.lower()
            context_text = ' '.join([doc['content'].lower() for doc in context])
            
            claims = [s.strip() for s in answer.split('.') if s.strip()]
            hallucinated = False
            
            for claim in claims:
                if len(claim.split()) > 3:
                    if not any(word in context_text for word in claim.lower().split()[:3]):
                        hallucinated = True
                        issues.append(f"Unsupported claim: {claim[:50]}...")
            
            if hallucinated:
                checks_failed.append('hallucination')
            else:
                checks_passed.append('hallucination')
        
        if 'completeness' in checks:
            query_keywords = set(query.lower().split())
            answer_keywords = set(answer.lower().split())
            
            overlap = query_keywords & answer_keywords
            if len(overlap) / len(query_keywords) > 0.3:
                checks_passed.append('completeness')
            else:
                checks_failed.append('completeness')
                issues.append("Answer may not fully address query")
        
        if 'citation' in checks:
            if any(marker in answer for marker in ['[', 'source:', 'according to']):
                checks_passed.append('citation')
            else:
                checks_failed.append('citation')
                issues.append("Missing citations")
        
        processing_time = time.time() - start_time
        is_valid = len(checks_failed) == 0
        confidence = len(checks_passed) / len(checks) if checks else 1.0
        
        return {
            'is_valid': is_valid,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed,
            'confidence': round(confidence, 3),
            'issues': issues,
            'processing_time_ms': processing_time * 1000
        }


# =============================================================================
# Task-Specific Generators (use prompts from config)
# =============================================================================

class TaskGenerator(ABC):
    """Abstract base for task-specific generation."""
    
    def __init__(self, base_url: str = None, api_key: str = None, prompt_variant: str = "default"):
        self._gen_op = GenerationOperation(base_url, api_key)
        self.prompt_variant = prompt_variant
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response for task."""
        pass
    
    def _format_context(self, context: List[Dict[str, str]]) -> str:
        """Format context documents as string."""
        parts = []
        for i, doc in enumerate(context, 1):
            title = doc.get('title', '')
            text = doc.get('text', doc.get('content', ''))
            if title:
                parts.append(f"[{i}] {title}: {text}")
            else:
                parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)


class QAGenerator(TaskGenerator):
    """QA task generator - answers questions from context."""
    
    def generate(
        self,
        query: str,
        context: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate answer for QA task.
        
        Args:
            query: Question text
            context: Retrieved documents
            
        Returns:
            Dict with answer, model, latency_ms, metadata
        """
        # Select prompt based on variant
        if self.prompt_variant == "variant_b":
            prompts = config.generation.prompts.qa_variant_b
            # Variant B: different prompts for k=0 vs k≥1
            is_k0 = len(context) == 0
            if is_k0:
                system_prompt = prompts.system_k0
                user_template = prompts.user_template_k0
                context_str = ""
            else:
                system_prompt = prompts.system_kplus
                user_template = prompts.user_template_kplus
                context_str = self._format_context(context)
        else:
            # Default: original behavior
            prompts = config.generation.prompts.qa
            system_prompt = prompts.system
            user_template = prompts.user_template
            context_str = self._format_context(context)
        
        user_prompt = user_template.format(context=context_str, query=query)
        
        result = self._gen_op.execute(
            prompt=user_prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        result['task'] = 'qa'
        result['query'] = query
        result['prompt_variant'] = self.prompt_variant
        return result


class FactVerificationGenerator(TaskGenerator):
    """Fact verification generator - classifies claims against evidence."""
    
    def generate(
        self,
        query: str,  # claim text
        context: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate verdict for fact verification task.
        
        Args:
            query: Claim text to verify
            context: Evidence documents
            
        Returns:
            Dict with predicted_label, rationale, confidence, model, latency_ms
        """
        # Get prompts from config
        prompts = config.generation.prompts.fact_verification
        system_prompt = prompts.system
        user_template = prompts.user_template
        valid_labels = prompts.labels
        
        context_str = self._format_context(context)
        user_prompt = user_template.format(context=context_str, claim=query)
        
        result = self._gen_op.execute(
            prompt=user_prompt,
            system_prompt=system_prompt,
            **kwargs
        )
        
        # Parse label from response
        response = result['answer']
        predicted_label = self._extract_label(response, valid_labels)
        rationale = self._extract_rationale(response)
        
        result['task'] = 'fact_verification'
        result['claim'] = query
        result['predicted_label'] = predicted_label
        result['rationale'] = rationale
        result['raw_response'] = response
        
        return result
    
    def _extract_label(self, response: str, valid_labels: List[str]) -> str:
        """Extract label from model response."""
        response_upper = response.upper()
        
        # Check for verdict line
        if 'VERDICT:' in response_upper:
            verdict_part = response_upper.split('VERDICT:')[1].strip()
            for label in valid_labels:
                if label in verdict_part:
                    return label
        
        # Fall back to finding label anywhere
        for label in valid_labels:
            if label in response_upper:
                return label
        
        # Default if no label found
        return 'NOT_ENOUGH_INFO'
    
    def _extract_rationale(self, response: str) -> str:
        """Extract rationale from model response."""
        if 'Verdict:' in response:
            return response.split('Verdict:')[0].strip()
        if 'VERDICT:' in response:
            return response.split('VERDICT:')[0].strip()
        return response


def get_generator(task_type: str, **kwargs) -> TaskGenerator:
    """
    Factory function to get task-specific generator.
    
    Args:
        task_type: 'qa' or 'fact_verification'
        
    Returns:
        TaskGenerator instance
    """
    generators = {
        'qa': QAGenerator,
        'fact_verification': FactVerificationGenerator,
    }
    
    if task_type not in generators:
        raise ValueError(f"Unknown task type: {task_type}. Valid: {list(generators.keys())}")
    
    return generators[task_type](**kwargs)
