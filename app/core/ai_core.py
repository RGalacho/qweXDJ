"""
AI Service Module for integrating with tinyLlama model
"""
import logging
import torch
import torch.nn.functional as torchF

from typing import Dict, Any
from core.config import AIModelConfig
from core.chroma_manager import chroma_service
from domain.models import GenerationRequest, ContinuationRequest

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
    HAS_CTRANSFORMERS = True
except ImportError:
    HAS_CTRANSFORMERS = False
    logging.warning("ctransformers not available. AI functionality will be limited.")

logger = logging.getLogger(__name__)


class AIService:
    """
    Service for processing natural language prompts using AI models
    """
    def __init__(self, config: AIModelConfig = AIModelConfig()):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

        if not HAS_CTRANSFORMERS:
            logger.warning("ctransformers not installed. AI service will not function without model.")
            return
            
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            raise


    def _load_model(self):
        """Load the AI model with appropriate configuration"""
        logger.info("_load_model - Init")
        if not HAS_CTRANSFORMERS:
            return
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"_load_model - Working with {device} device")

            # noinspection none-function-assignment
            self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    # padding_side = "right", # Padding should be left when generating and right when training/tuning
            )

            if self.tokenizer.pad_token is None:
                # noinspection unresolved-references
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # noinspection unresolved-references
            model_config = AutoConfig.from_pretrained(self.config.model_path,
                    device_map= device, # "auto" autoconfig
                    num_return_sequences= 1,
                    # dtype = torch.float16,
                    # For local inference, we might need to disable CUDA if not available
                    # gpu_layers = 0  # Disable GPU layers for CPU-only inference by default
            )

            # Load model using ctransformers
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"_load_model - Successfully loaded AI model: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"_load_model - Failed to load model {self.config.model_name}: {e}")
            self.is_loaded = False
            raise


    def generate_prompt_response(self, prompt: str, chromadb_context: str) -> tuple[str, float | int]:
        """
        Generate a response to the given prompt using the AI model
        
        Args:
            :param prompt: The natural language prompt to process
            :param chromadb_context: Local collection context

        Returns:
            (str, float): Generated response from AI model and confidence_score
        """
        logger.info("generate_prompt_response - Init")

        if not self.is_loaded or self.model is None:
            raise RuntimeError("generate_prompt_response - AI model not loaded. Cannot process prompt.")
            
        try:
            role_user_content = f"""User prompt: {prompt}

Instructions: You have exclusively available the following local collection:
{chromadb_context}

Do not limit yourself. Scan the entire collection and output every single sharing a similar tone. Maximize the output list to include every valid candidate found in the text.
Format strictly as:
`Artist name` - `Song name` - `duration` [`file path`]
"""

            messages = [
                {"content": """You are QweXDJ, an elite, hyper-inclusive AI DJ. Your primary directive is to build the LONGEST possible playlist matching the user's concept from the local collection.
Metacognitive Rules & BPM Mapping:
You must map user concepts to the song metadata using the following rhythm criteria:
- "Slow and Relaxed": bpm < 80 (Ideal for sleep, deep reading, meditation)
- "Moderate and Casual": 80 <= bpm <= 115 (Ideal for working, casual listening, reading, relaxing)
- "Lively and Steady": 115 < bpm <= 135 (Ideal for walking, moderate focus, steady tasks)
- "Fast and Energetic (Upbeat)": bpm > 135 (Ideal for exercising, high energy, running)

Core Execution Rules:
1. Use ONLY the songs explicitly listed in the local collection. Never invent songs.
2. EXHAUSTIVE MATCHING: You must scan the entire local collection and include EVERY SINGLE SONG that fits the user's concept according to the BPM mapping, artist and/or genres. Do not stop at arbitrary minimums; include all valid matches (aim for 15+ if available).
3. First, write out your reasoning process inside <think> tags, listing all potential matches.
4. If the local collection contains more matching tracks, you MUST include them all to maximize the playlist length.
5. Then, respond strictly with the final list. Output raw text only, without markdown code blocks.
For each song, include the artist and title, and a placeholder file path:
`Artist name` - `Song name` - `duration` [`file path`]
""",
                 "role": "system"
                },
                {"role": "user", "content": role_user_content}
            ]

            logger.info(f"generate_prompt_response - Generated chat template: {messages}")

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # noinspection calling-non-callable
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(**model_inputs,
                                                do_sample=True,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                max_new_tokens=self.config.max_new_tokens,
                                                temperature=self.config.temperature,
                                                top_p=self.config.top_p,
                                                repetition_penalty=self.config.repetition_penalty,
                                                output_scores=True,
                                                return_dict_in_generate=True)
            generated_ids_full = outputs.sequences
            logits = outputs.scores

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids_full)
            ]

            confidence_score = self.calculate_confidence_score(generated_ids, logits)

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logger.info(f"generate_prompt_response - AI generated response for prompt: {response}\n Confidence score: {confidence_score}")

            return response, confidence_score
            
        except Exception as e:
            logger.error(f"generate_prompt_response - Error generating AI response: {e}")
            raise


    def continuation_prompt_response(self, tracks: list[str], chromadb_context: str) -> tuple[str, float | int]:
        """
        Generate a response to the given prompt using the AI model

        Args:
            :type tracks: Track list to continue
            :param chromadb_context: Local collection context

        Returns:
            (str, float): Generated response from AI model and confidence_score
        """
        logger.info("generate_prompt_response - Init")

        if not self.is_loaded or self.model is None:
            raise RuntimeError("generate_prompt_response - AI model not loaded. Cannot process prompt.")

        try:
            role_user_content = f"""User prompt: Pick 5 tracks who match this songs in a playlist:
{tracks}

Instructions: You have exclusively available the following local collection:
{chromadb_context}

Analyze the 5 input seed songs provided in the prompt. Scan the entire collection to find exactly 5 subsequent tracks that seamlessly harmonize with them (you may include songs by the same artists, similar artists, or matching genres/BPM flows). Do not repeat the input songs.
Format strictly as:
`Artist name` - `Song name` - `duration` [`file path`]
"""

            messages = [{"content": """You are QweXDJ, an elite, hyper-inclusive AI DJ. Your primary directive is to take a seed list of 5 input songs from the local collection and select exactly 5 subsequent tracks that seamlessly harmonize with them, maintaining an ideal flow of energy, mood, and BPM.

Metacognitive Rules & BPM & Transition Mapping:
Analyze the BPM, genre, and energy of the 5 input songs to determine the trajectory of the set. The 5 output tracks must follow a smooth transition curve based on these criteria:
- "Slow and Relaxed": bpm < 80
- "Moderate and Casual": 80 <= bpm <= 115
- "Lively and Steady": 115 < bpm <= 135
- "Fast and Energetic (Upbeat)": bpm > 135

Core Execution Rules:
1. Use ONLY the songs explicitly listed in the local collection. Never invent songs. Do not repeat the 5 input songs in your output.
2. HARMONIC CONTINUATION: Select the 5 best tracks that logically follow the vibe and momentum of the input seed songs. You may include additional tracks by the same artists if they fit the flow, as well as tracks by similar artists or matching genres from the collection.
3. First, write out your reasoning process inside <think> tags, analyzing the input songs' BPM/genres and evaluating potential matches from the local collection for a smooth transition.
4. Then, respond strictly with the final list of 5 tracks. Output raw text only, without markdown code blocks.
For each song, include the artist, title, duration, and the placeholder file path in the following format:
`Artist name` - `Song name` - `duration` [`file path`]
""",
                     "role": "system"
                     },
                    {"role": "user", "content": role_user_content}
                ]

            logger.info(f"continuation_prompt_response - Generated chat template: {messages}")

            text = self.tokenizer.apply_chat_template(messages,
                                                      tokenize=False,
                                                      add_generation_prompt=True)

            # noinspection calling-non-callable
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(**model_inputs,
                                          do_sample=True,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          max_new_tokens=self.config.max_new_tokens,
                                          temperature=self.config.temperature,
                                          top_p=self.config.top_p,
                                          repetition_penalty=self.config.repetition_penalty,
                                          output_scores=True,
                                          return_dict_in_generate=True)
            generated_ids_full = outputs.sequences
            logits = outputs.scores

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in
                zip(model_inputs.input_ids, generated_ids_full)
            ]

            confidence_score = self.calculate_confidence_score(generated_ids, logits)

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            logger.info(f"generate_prompt_response - AI generated response for prompt: {response}\n Confidence score: {confidence_score}")

            return response, confidence_score
        except Exception as e:
            logger.error(f"generate_prompt_response - Error generating AI response: {e}")
            raise


    async def analyze_prompt_and_generate_tracks(self, request: GenerationRequest) -> tuple[str, float | int]:
        """
        Build chroma local collection context and launch the agent
        
        Args:
            request: GenerationRequest with prompt and parameters
            
        Returns:
            List of Track objects that match the prompt requirements
        """
        logger.info("analyze_prompt_and_generate_tracks - Init")

        try:
            # local_candidates = chroma_service.get_local_audio_context(request.prompt, max_resuls=50)
            local_candidates = chroma_service.get_local_audio_context(request.prompt, max_results=50)

            chromadb_context = "\n".join([f"- {c}" for c in local_candidates])

            # ai_response = self.generate_prompt_response(request.prompt, chromadb_context)
            ai_response = self.generate_prompt_response(request.prompt, chromadb_context)


            return ai_response
            
        except Exception as e:
            logger.error(f"analyze_prompt_and_generate_tracks - Error in AI processing of request: {e}")
            raise


    async def analyze_tracks_and_generate_continuation(self, request: ContinuationRequest) -> tuple[str, float | int]:
        """
        Build chroma local collection context and launch the agent

        Args:
            request: ContinuationRequest

        Returns:
            List of Track objects that match the tracks received
        """
        logger.info("analyze_tracks_and_generate_continuation - Init")

        try:
            local_candidates = chroma_service.get_local_audio_continuation_context(request.tracks, max_results=50)

            chromadb_context = "\n".join([f"- {c}" for c in local_candidates])

            ai_response = self.continuation_prompt_response(request.tracks, chromadb_context)

            return ai_response

        except Exception as e:
            logger.error(f"analyze_tracks_and_generate_continuation - Error in AI processing of request: {e}")
            raise


    @staticmethod
    def calculate_confidence_score(generated_ids: list[Any], logits) -> float | int:
        # Confidence score
        generated_tokens = generated_ids[0]
        token_probs = []

        # logits[i] contains the tensor scores for the i-th generated token.
        for i, token_id in enumerate(generated_tokens):
            # Apply softmax to convert logits to probabilities (0.0 - 1.0).
            probs = torchF.softmax(logits[i][0], dim=-1)

            # Retrieve the probability of the actually selected token.
            prob_of_token = probs[token_id].item()
            token_probs.append(prob_of_token)

        # Compute the average confidence score for the whole response.
        confidence_score = (sum(token_probs) / len(token_probs)) if token_probs else 0.0
        return confidence_score


    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "not_loaded", "model": None}
        
        info = {
            "status": "loaded",
            "model_name": self.config.model_name,
            "local_path": self.config.model_path,
            "context_length": self.config.max_context_length,
            "temperature": self.config.temperature
        }
        return info


# Global service instance
ai_service = AIService()