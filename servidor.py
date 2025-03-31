import os
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  
from gpt4all import GPT4All
from pyngrok import ngrok  
from queue import Queue
import uuid
from PIL import Image  
from diffusers import AutoPipelineForText2Image
import torch

# Configuração do modelo de texto
model_path = "C:/Users/joaot/AppData/Local/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "generated_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Verifica se o modelo existe
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Erro: Modelo não encontrado em {model_path}.")

print("Carregando modelo GPT4All...")
gpt4all = GPT4All(model_path)

# Carregar modelo de geração de imagens
print("Carregando modelo de geração de imagens...")
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo") 
pipe.to("cpu")  # Mantendo na CPU para compatibilidade

# Personalidade da NoellE
personality = (  
    "Você é NoellE, uma pessoa que combina inteligência afiada com um senso de orgulho e atitude forte..."
)

# Fila de mensagens
message_queue = Queue()

# Estruturas para respostas e eventos
responses = {}
events = {}
dict_lock = threading.Lock()
model_lock = threading.Lock()

def get_response(prompt, request_id):
    with model_lock:
        response = gpt4all.generate(prompt, max_tokens=200, temp=0.5)
    
    with dict_lock:
        responses[request_id] = response
        events[request_id].set()  

def process_queue():
    while True:
        user_input, request_id = message_queue.get()
        prompt = f"{personality}\nHumano: {user_input}\nNoellE responde:"

        get_response(prompt, request_id)

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "Envie um JSON com o campo 'message'."}), 400

    user_input = data["message"]
    request_id = str(uuid.uuid4())

    with dict_lock:
        events[request_id] = threading.Event()

    message_queue.put((user_input, request_id))

    events[request_id].wait()

    with dict_lock:
        response = responses.pop(request_id, "Erro ao gerar resposta.")
        events.pop(request_id, None)  

    return jsonify({"response": response})

@app.route("/image-chat", methods=["POST"])
def image_chat():
    if "image" not in request.files:
        return jsonify({"error": "Envie uma imagem no campo 'image'."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "Nome de arquivo inválido."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(file_path)

    try:
        img = Image.open(file_path)
        width, height = img.size
        img.close()
        img_info = f"Imagem recebida com resolução {width}x{height}."
    except Exception as e:
        img_info = f"Erro ao processar a imagem: {str(e)}"

    prompt = f"{personality}\nHumano: {img_info}\nNoellE responde:"
    request_id = str(uuid.uuid4())

    with dict_lock:
        events[request_id] = threading.Event()

    message_queue.put((prompt, request_id))

    events[request_id].wait()

    with dict_lock:
        response = responses.pop(request_id, "Erro ao gerar resposta.")
        events.pop(request_id, None)  

    return jsonify({"response": response, "image_info": img_info})

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "Envie um JSON com o campo 'prompt'."}), 400

    prompt = data["prompt"]
    image_name = f"{uuid.uuid4()}.png"
    image_path = os.path.join(app.config["OUTPUT_FOLDER"], image_name)

    try:
        image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Erro ao gerar imagem: {str(e)}"}), 500

    return jsonify({"image_url": f"/get-image/{image_name}"})

@app.route("/get-image/<filename>")
def get_image(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    threading.Thread(target=process_queue, daemon=True).start()
    
    public_url = "https://legal-fish-simply.ngrok-free.app"
    print("Ngrok tunnel URL:", public_url)

    app.run(host="0.0.0.0", port=575)
