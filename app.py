import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os, time, requests, json, re, logging, pickle
import faiss, markdown2, openai, anthropic

from google.generativeai import GenerativeModel
import google.generativeai as genai


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

genai.configure(api_key=GEMINI_API_KEY)
client = openai.OpenAI()
# ant_client = anthropic.Anthropic()


class DataManager:
    def __init__(
        self,
        file_path="Mental_Health_Conversations_deduplicated_20241020_013104.csv",
        cache_file="cached_data.pkl",
    ):
        self.file_path = file_path
        self.cache_file = cache_file
        self.data = None
        self.embeddings = None
        self.faiss_index = None
        self.load_data()

    def load_data(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.data = cached_data["data"]
                    self.embeddings = cached_data["embeddings"]
                    self.faiss_index = cached_data["faiss_index"]
                    logger.info("Data loaded from cache successfully")
                    return
            except Exception as e:
                logger.error(f"Error loading cached data: {str(e)}")

        try:
            self.data = pd.read_csv(self.file_path)
            self.data["Cleaned_Context"] = self.data["Context"].apply(self.clean_text)
            self.data["Cleaned_Response"] = self.data["Response"].apply(self.clean_text)
            self._prepare_embeddings()
            self._build_faiss_index()

            with open(self.cache_file, "wb") as f:
                pickle.dump(
                    {
                        "data": self.data,
                        "embeddings": self.embeddings,
                        "faiss_index": self.faiss_index,
                    },
                    f,
                )
            logger.info("Data loaded from CSV and cached successfully")
        except Exception as e:
            logger.error(f"Error loading data from CSV: {str(e)}")
            raise

    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    def _prepare_embeddings(self):
        try:
            self.embeddings = []
            batch_size = 2048
            texts = self.data["Cleaned_Context"].tolist()

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                batch_embeddings = [item.embedding for item in response.data]
                self.embeddings.extend(batch_embeddings)

            self.embeddings = np.array(self.embeddings, dtype="float32")
            logger.info(f"Created embeddings for {len(self.embeddings)} texts")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def _build_faiss_index(self):
        try:
            dimension = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(self.embeddings)
            logger.info("FAISS index built successfully")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise

    def find_similar_cases(self, query, num_examples=4):
        try:
            start_time = time.time()
            logger.info("Starting user query embedding")
            response = client.embeddings.create(
                input=self.clean_text(query), model="text-embedding-ada-002"
            )

            logger.info(
                f"Finished user query embedding in {time.time() - start_time:.2f} seconds"
            )
            query_embedding = np.array(
                response.data[0].embedding, dtype="float32"
            ).reshape(1, -1)

            start_time = time.time()
            logger.info("Starting FAISS search")

            # !
            distances, indices = self.faiss_index.search(query_embedding, num_examples)

            logger.info(
                f"Finished FAISS search in {time.time() - start_time:.5f} seconds"
            )

            similar_cases = []
            seen_contexts = set()

            for idx, distance in zip(indices[0], distances[0]):
                if idx == -1:
                    continue

                context = self.data["Context"].iloc[idx]
                cleaned_context = self.clean_text(context)

                if cleaned_context in seen_contexts:
                    continue

                seen_contexts.add(cleaned_context)
                similar_cases.append(
                    {
                        "context": context,
                        "response": self.data["Response"].iloc[idx],
                        "similarity": float(distance),
                    }
                )

            return similar_cases

        except Exception as e:
            logger.error(f"Error finding similar cases: {str(e)}")
            return []


class LLMManager:
    def __init__(self):
        self.api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.gemini_model = GenerativeModel("gemini-1.5-flash")

    def check_ollama_status(self):
        try:
            response = requests.get(f"{self.api_base}")
            response.raise_for_status()
            return True, "Ollama server is running and accessible"
        except requests.RequestException as e:
            return False, f"Error connecting to Ollama server: {str(e)}"

    # def generate_response_ant(self, prompt):
    #     try:
    #         start_time = time.time()
    #         logger.info("Starting chat completion with Anthropic")
    #         message = ant_client.messages.create(
    #             model="claude-3-haiku-20240307",
    #             max_tokens=1024,
    #             messages=[{"role": "user", "content": prompt}],
    #         )
    #         logger.info(
    #             f"Finished chat completion with Anthropic in {time.time() - start_time:.2f} seconds"
    #         )

    #         logger.log(f"Content:\n\n\n{message.content}")
    #         return message.content
    #     except Exception as e:
    #         logger.error(f"Error generating response from Anthropic: {str(e)}")
    #         return self.generate_response_gemini(prompt)

    def generate_response_gemini(self, prompt):  # * Fastest model
        try:
            start_time = time.time()
            logger.info("Starting chat completion with Gemini Pro")
            response = self.gemini_model.generate_content(prompt)
            logger.info(
                f"Finished chat completion with Gemini Pro in {time.time() - start_time:.2f} seconds"
            )

            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Gemini Pro: {str(e)}")
            return self.generate_response_gpt(prompt)

    def generate_response_gpt(self, prompt):
        try:
            start_time = time.time()
            logger.info("Starting chat completion with OpenAI")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced mental health counseling advisor.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
            )
            logger.info(
                f"Finished chat completion with OpenAI in {time.time() - start_time:.2f} seconds"
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from OpenAI: {str(e)}")
            return self.generate_response_ollama(prompt)  # ! only available locally

    def generate_response_ollama(self, prompt):
        url = f"{OLLAMA_API_BASE}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Request error with Ollama: {e}")
            return "An error occurred while communicating with the language model."
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error with Ollama: {e}")
            return "An error occurred while processing the response from the language model."


class ResponseGenerator:
    def __init__(self, data_manager, llm_manager):
        self.data_manager = data_manager
        self.llm_manager = llm_manager

    def generate_guidance(self, user_input):
        similar_cases = self.data_manager.find_similar_cases(user_input)
        prompt = self._create_prompt(user_input, similar_cases)
        llm_response = self.llm_manager.generate_response_gemini(prompt)
        return {"suggestion": llm_response, "similar_cases": similar_cases}

    @staticmethod
    def _create_prompt(user_input, similar_cases=None, similarity_threshold=15):
        prompt = (
            "You are an expert mental health counseling advisor providing AI-powered "
            "suggestions for counseling approaches. Your guidance should be professional, "
            "evidence-based, and formatted for clear web display using HTML markup when needed.\n\n"
            "Core Competencies:\n"
            "- Clinical assessment and intervention planning\n"
            "- Evidence-based therapeutic approaches\n"
            "- Risk assessment and crisis management\n"
            "- Cultural competency and ethical practice\n"
            "- Clear communication and practical guidance\n\n"
        )

        if similar_cases:
            relevant_cases = [case for case in similar_cases if (case['similarity'] * 100) > similarity_threshold]
            
            if relevant_cases:
                prompt += (
                    "Reference Cases Analysis:\n"
                    "The following similar cases have been identified, with similarity "
                    "scores indicating their relevance to the current situation:\n\n"
                )
                
                for case in relevant_cases:
                    prompt += (
                        f"Similar Case (Similarity: {case['similarity']*100:.1f}%):\n"
                        f"Situation: {case['context']}\n"
                        f"Response: {case['response']}\n\n"
                    )


        prompt += (
            "Current Case Details:\n"
            "==========================================\n"
            f"{user_input}\n"
            "==========================================\n\n"
        )

        prompt += (
            "Provide a comprehensive analysis and recommendations covering:\n\n"
            "1. Initial Assessment\n"
            "   - Key risk factors and immediate concerns\n"
            "   - Severity of presenting symptoms\n"
            "   - Relevant contextual factors\n"
            "   - Preliminary diagnostic considerations\n\n"
            "2. Recommended Therapeutic Approach\n"
            "   - Primary therapeutic framework with clear rationale\n"
            "   - Specific therapeutic goals\n"
            "   - Alternative approaches to consider\n"
            "   - Expected treatment timeline\n\n"
            "3. Intervention Strategies\n"
            "   - Specific techniques and interventions\n"
            "   - Session structure recommendations\n"
            "   - Essential resources and tools\n"
            "   - Crisis protocols if needed\n\n"
            "4. Important Considerations\n"
            "   - Cultural and contextual factors\n"
            "   - Anticipated challenges and mitigation strategies\n"
            "   - Ethical considerations\n"
            "   - Progress monitoring approach\n\n"
        )

        prompt += (
            "Guidelines for your response:\n"
            "1. Be concise but thorough\n"
            "2. Use clear, professional language\n"
            "3. Prioritize evidence-based approaches\n"
            "4. Include specific, actionable recommendations\n"
            "5. Address both immediate needs and long-term treatment planning\n"
            "6. Consider cultural and contextual factors throughout\n"
        )

        return prompt


app = Flask(__name__)

try:
    data_manager = DataManager()
    llm_manager = LLMManager()
    status, message = llm_manager.check_ollama_status()
    if not status:
        logger.error(f"Ollama server not accessible: {message}")
    else:
        logger.info("Ollama server is running and accessible")
    response_generator = ResponseGenerator(data_manager, llm_manager)
    logger.info("Application components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing application components: {str(e)}")
    raise


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        try:
            user_input = request.form.get("user_input", "").strip()
            if not user_input:
                return render_template(
                    "index.html", error="Please provide a description of the situation."
                )

            result = response_generator.generate_guidance(user_input)
            suggestion_html = markdown2.markdown(result["suggestion"])
            return render_template(
                "index.html",
                suggestion=suggestion_html,
                similar_cases=result["similar_cases"],
                user_input=user_input,
            )
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return render_template(
                "index.html", error="Error occurred. Please try again."
            )


@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return render_template(
        "index.html", error="An unexpected error occurred. Please try again later."
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# ! keep alive cmd
# nohup bash -c 'while true; do hey -n 1 -c 1 https://legacy.fly.dev/; sleep 120; done' &

# ! to stop
# ps aux | grep 'hey'
# kill <PID>   # replace <PID> with the process ID from the above command
