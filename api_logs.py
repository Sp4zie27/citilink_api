from flask import Flask, request, jsonify
import torch
import warnings
import logging
import os
import sys
import time
import numpy as np
from joblib import load
from transformers import AutoModelForNextSentencePrediction, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from modelos.extracao_votos.modeling_deberta_crf import DebertaCRFForTokenClassification


warnings.filterwarnings("ignore")


# ----------------------------------- Logs -----------------------------------


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("API")


# ---------------------------- CONFIGURAÇÃO INICIAL ---------------------------

app = Flask(__name__)

logger.info("API Iniciada")
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Dispositivo ativo: {device.upper()}")

Modelo_1 = "modelos/divisao_documento"
Modelo_2 = "modelos/extracao_metadados"
Modelo_3 = "modelos/divisao_segmentos"
Modelo_4 = "modelos/anonimizacao"
Modelo_5 = "modelos/sumarizacao_segmentos"
Modelo_6 = "modelos/extracao_votos"
Modelo_7 = "modelos/classificacao_topicos"

logger.info(f"Caminhos dos modelos: {Modelo_1}, {Modelo_2}, {Modelo_3}, {Modelo_4}, {Modelo_5}, {Modelo_6}, {Modelo_7}")


# ------------------------------- CACHE DE MODELOS -------------------------------

model_cache = {}

def get_modelo_divisao_documento():
    if "qa" not in model_cache:
        logger.info("Carregando modelo Divisão de Documento (FastTokenizer)")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_1, use_fast=True)  # <- aqui
        model = AutoModelForQuestionAnswering.from_pretrained(Modelo_1)
        model.eval()
        model_cache["qa"] = (tokenizer, model)
    return model_cache["qa"]


def get_modelo_extracao_metadados():
    if "ner" not in model_cache:
        logger.info("Carregando modelo Extração de Metadados")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_2)
        model = AutoModelForTokenClassification.from_pretrained(Modelo_2)
        extractor = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        model_cache["ner"] = (tokenizer, extractor)
    return model_cache["ner"]


def get_modelo_divisao_segmentos():
    if "segmentador" not in model_cache:
        logger.info("Carregando modelo de Segmentação de Texto (local)")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_3)
        model = AutoModelForNextSentencePrediction.from_pretrained(Modelo_3)
        model.eval()
        model_cache["segmentador"] = (tokenizer, model)
    return model_cache["segmentador"]


def get_modelo_anonimizacao():
    if "anon" not in model_cache:
        logger.info("Carregando modelo de Anonimização")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_4)
        model = AutoModelForTokenClassification.from_pretrained(Modelo_4)
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        model_cache["anon"] = (tokenizer, ner_pipeline)
    return model_cache["anon"]


def get_modelo_sumarizacao():
    if "summarizer" not in model_cache:
        logger.info("Carregando modelo de Sumarização")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_5)
        model = AutoModelForSeq2SeqLM.from_pretrained(Modelo_5)
        model.eval()
        model_cache["summarizer"] = (tokenizer, model)
    return model_cache["summarizer"]


def get_modelo_extracao_votos():
    if "votos" not in model_cache:
        logger.info("Carregando modelo de Extração de Votos")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_6, trust_remote_code=True)
        model = DebertaCRFForTokenClassification.from_pretrained(Modelo_6, trust_remote_code=True)
        model.eval()
        model_cache["votos"] = (tokenizer, model)
    return model_cache["votos"]


def get_modelo_classificacao_topicos():
    if "topicos" not in model_cache:
        logger.info("Carregando modelo 7 - Classificação de Tópicos")
        tokenizer = AutoTokenizer.from_pretrained(Modelo_7)
        model = AutoModelForSequenceClassification.from_pretrained(Modelo_7)
        thresholds = np.load(os.path.join(Modelo_7, "optimal_thresholds.npy"))
        mlb = load(os.path.join(Modelo_7, "mlb_encoder.joblib"))
        model.eval()
        model_cache["topicos"] = (tokenizer, model, thresholds, mlb)
    return model_cache["topicos"]



# ---------------------- MODELO 1: Divisão de documentos ----------------------


def responder_pergunta(tokenizer, model, context, question, max_length=512, doc_stride=128):

    inputs = tokenizer(
        question,
        context,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        max_length=max_length,
        stride=doc_stride,
        truncation=True,
        padding=False
    )

    all_start_logits = []
    all_end_logits = []

    for i in range(len(inputs["input_ids"])):
        input_ids = torch.tensor([inputs["input_ids"][i]])
        attention_mask = torch.tensor([inputs["attention_mask"][i]])

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        all_start_logits.append(outputs.start_logits)
        all_end_logits.append(outputs.end_logits)

    # Escolher o melhor span
    best_score = -float("inf")
    best_answer = ""
    for i, offset_mapping in enumerate(inputs["offset_mapping"]):
        start_idx = torch.argmax(all_start_logits[i])
        end_idx = torch.argmax(all_end_logits[i])
        if end_idx >= start_idx:  # considerar apenas spans válidos
            score = all_start_logits[i][0, start_idx] + all_end_logits[i][0, end_idx]
            if score > best_score:
                start_char, end_char = offset_mapping[start_idx][0], offset_mapping[end_idx][1]
                best_answer = context[start_char:end_char]
                best_score = score

    return best_answer.strip()


def dividir_ata(tokenizer, model, texto_original):

    logger.info("Iniciando divisão da ata via QA")

    opening_question = "No início da ata há um segmento de abertura. Qual é a última frase desse segmento de abertura?"
    closing_question = "No final da ata há um segmento de encerramento. Qual é a primeira frase desse segmento de encerramento?"

    opening_marker = responder_pergunta(tokenizer, model, texto_original, opening_question)
    closing_marker = responder_pergunta(tokenizer, model, texto_original, closing_question)

    if not opening_marker or not closing_marker:
        logger.warning("Não foi possível localizar marcadores, retornando texto completo")
        return {"metadados_iniciais": "", "corpo": texto_original, "metadados_finais": ""}

    inicio_opening = texto_original.find(opening_marker) + len(opening_marker)
    inicio_closing = texto_original.find(closing_marker)

    if inicio_closing <= inicio_opening:
        logger.warning("Marcadores invertidos, retornando texto completo")
        return {"metadados_iniciais": "", "corpo": texto_original, "metadados_finais": ""}

    metadados_iniciais = texto_original[:inicio_opening].strip()
    corpo = texto_original[inicio_opening:inicio_closing].strip()
    metadados_finais = texto_original[inicio_closing:].strip()

    logger.info("Divisão da ata concluída")
    return {"metadados_iniciais": metadados_iniciais, "corpo": corpo, "metadados_finais": metadados_finais}



# ---------------------- MODELO 2: Extração de metadados ----------------------


def agrupar_entidades(entidades):
    logger.info(f"Agrupando {len(entidades)} entidades detectadas")
    agrupadas = []
    atual = None
    try:
        for ent in entidades:
            entity_tag = ent["entity_group"]
            text = ent["word"].strip()
            label = entity_tag.replace("B-", "").replace("I-", "")
            if text.startswith("##") and atual:
                atual["word"] += text[2:]
                continue
            if entity_tag.startswith("B-") or not atual or label != atual["entity_group"]:
                if atual:
                    agrupadas.append(atual)
                atual = {"entity_group": label, "word": text}
            else:
                atual["word"] += " " + text
        if atual:
            agrupadas.append(atual)
        logger.info(f"Entidades agrupadas: {len(agrupadas)}")
        return agrupadas
    except Exception as e:
        logger.exception(f"Erro ao agrupar entidades: {e}")
        return agrupadas

def dividir_texto_por_tokens(tokenizer, texto, max_tokens=500, overlap=20):
    try:
        tokens = tokenizer.tokenize(texto)
        total_tokens = len(tokens)
        logger.info(f"Dividindo texto em chunks: {total_tokens} tokens, max {max_tokens}, overlap {overlap}")
        for i in range(0, total_tokens, max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            yield chunk_text
    except Exception as e:
        logger.exception(f"Erro ao dividir texto por tokens: {e}")

def extrair_entidades(pipeline_ner, tokenizer, texto):
    logger.info("Iniciando extração de entidades")
    start_time = time.time()
    entidades = []
    try:
        for chunk in dividir_texto_por_tokens(tokenizer, texto, max_tokens=500, overlap=20):
            entidades_chunk = pipeline_ner(chunk)
            entidades.extend(entidades_chunk)
        agrupadas = agrupar_entidades(entidades)
        logger.info(f"Extração concluída em {time.time()-start_time:.2f}s")
        return agrupadas
    except Exception as e:
        logger.exception(f"Erro ao extrair entidades: {e}")
        return []

def separar_participantes(texto, label):
    try:
        nomes = re.findall(r'[A-ZÁÉÍÓÚÂÊÎÔÛÇ][a-záéíóúâêîôûç]+(?: [A-ZÁÉÍÓÚÂÊÎÔÛÇ][a-záéíóúâêîôûç]+)+', texto)
        logger.info(f"{len(nomes)} participantes encontrados para label {label}")
        return [f"{n.strip()} ({label})" for n in nomes]
    except Exception as e:
        logger.exception(f"Erro ao separar participantes: {e}")
        return []

def extrair_campos(pipeline_ner, tokenizer, texto):
    logger.info("Extraindo campos do texto")
    try:
        entidades = extrair_entidades(pipeline_ner, tokenizer, texto)
        campos = {
            "data": None,
            "hora_inicio": None,
            "hora_fim": None,
            "participantes": [],
            "local": None,
            "tipo_reuniao": None,
            "numero_reuniao": None
        }
        for ent in entidades:
            label = ent["entity_group"]
            texto_ent = ent["word"]
            if "DATA" in label:
                campos["data"] = texto_ent
            elif "HORARIO-INICIO" in label:
                campos["hora_inicio"] = texto_ent
            elif "HORARIO-FIM" in label:
                campos["hora_fim"] = texto_ent
            elif "LOCAL" in label:
                campos["local"] = texto_ent
            elif "NUMERO-ATA" in label:
                campos["numero_reuniao"] = texto_ent
            elif "TIPO-REUNIAO" in label:
                if "ORDINARIA" in label:
                    campos["tipo_reuniao"] = "ordinária"
                elif "EXTRAORDINARIA" in label:
                    campos["tipo_reuniao"] = "extraordinária"
            elif "PARTICIPANTE" in label:
                campos["participantes"].extend(separar_participantes(texto_ent, label))
        logger.info(f"Campos extraídos: {campos}")
        return campos
    except Exception as e:
        logger.exception(f"Erro ao extrair campos: {e}")
        return {}

def carregar_modelo_ner():
    logger.info("Carregando modelo NER")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(Modelo_2)
        model = AutoModelForTokenClassification.from_pretrained(Modelo_2)
        extractor = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        logger.info(f"Modelo NER carregado em {time.time()-start_time:.2f}s")
        return tokenizer, extractor
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo NER: {e}")
        raise

def extrair_metadados(metadados_iniciais, metadados_finais, pipeline_ner):
    logger.info("Iniciando extração de metadados")
    try:
        resultados_iniciais = extrair_campos(pipeline_ner, pipeline_ner.tokenizer, metadados_iniciais)
        entidades_finais = extrair_entidades(pipeline_ner, pipeline_ner.tokenizer, metadados_finais)
        hora_fim = None
        for ent in entidades_finais:
            if "HORARIO-FIM" in ent["entity_group"]:
                hora_fim = ent["word"]
                break
        resultados_finais = {"hora_fim": hora_fim}
        logger.info("Extração de metadados concluída")
        return {"metadados_iniciais": resultados_iniciais, "metadados_finais": resultados_finais}
    except Exception as e:
        logger.exception(f"Erro ao extrair metadados: {e}")
        return {"metadados_iniciais": {}, "metadados_finais": {}}


# ---------------------- MODELO 3: Divisão de segmentos ----------------------

def preparar_sentencas_local(texto):
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        nltk.download("punkt", quiet=True)
        sentencas = sent_tokenize(texto)
    except Exception:
        sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())
    return [s.strip() for s in sentencas if s.strip()]


def segmentar_texto(sentencas, threshold=0.65):
    tokenizer, model = get_modelo_divisao_segmentos()
    model.to(device)

    segmentos = []
    if not sentencas:
        return segmentos

    buffer = [sentencas[0]]
    max_len = min(getattr(tokenizer, "model_max_length", 512), 512)

    for i in range(len(sentencas) - 1):
        sent_a = sentencas[i]
        sent_b = sentencas[i + 1]

        inputs = tokenizer(
            sent_a,
            sent_b,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_len
        )

        inputs = {k: v.long().to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            not_next_prob = probs[0][1].item()

        if not_next_prob > threshold:
            segmentos.append(" ".join(buffer))
            buffer = [sent_b]
        else:
            buffer.append(sent_b)
    if buffer:
        segmentos.append(" ".join(buffer))
    return segmentos


# ---------------------- MODELO 4: Anonimização ----------------------


from flask import request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger("API")

# Função de normalização de nomes
def normalizar_nome(nome):
    nome = re.sub(r"\(.*?\)", "", nome)  # remove parênteses
    nome = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s]", "", nome)  # remove pontuação
    return nome.strip().lower()

# Verifica se uma entidade é um participante
def eh_nome_participante(ent_texto, participantes_norm, limiar=0.6):
    ent_norm = normalizar_nome(ent_texto)
    for nome_part in participantes_norm:
        ratio = SequenceMatcher(None, ent_norm, nome_part).ratio()
        if ratio >= limiar:
            return True
        if ent_norm in nome_part or nome_part in ent_norm:
            return True
    return False

# Função de anonimização usando apenas o pipeline do modelo local
def anonimizar_texto(texto, ner_pipeline, participantes_norm=None):
    if not isinstance(texto, str) or not texto.strip():
        return texto

    entidades = ner_pipeline(texto)
    if not entidades:
        return texto

    # Ordena entidades pelo início no texto
    entidades = sorted(entidades, key=lambda x: x["start"])
    texto_anonimizado = list(texto)

    # Substitui entidades por asteriscos
    for ent in reversed(entidades):
        start, end = ent["start"], ent["end"]
        if end - start < 3:
            continue
        ent_texto = texto[start:end]
        if participantes_norm and eh_nome_participante(ent_texto, participantes_norm):
            continue
        tokens = re.findall(r"\S+", ent_texto)
        texto_anonimizado[start:end] = list(" ".join(["*****" for _ in tokens]))

    return "".join(texto_anonimizado)

# Aplica anonimização a uma ata inteira
def anonimizar_ata(data, ner_pipeline):
    participantes_norm = []
    if "participantes" in data and isinstance(data["participantes"], list):
        participantes_norm = [normalizar_nome(p) for p in data["participantes"]]

    resultado = {}
    for key, value in data.items():
        if isinstance(value, str):
            resultado[key] = anonimizar_texto(value, ner_pipeline, participantes_norm)
        else:
            resultado[key] = value

    return resultado

# ---------------------- MODELO 5: Sumarização de Segmentos ----------------------


tokenizer_5 = AutoTokenizer.from_pretrained(Modelo_5)
model_5 = AutoModelForSeq2SeqLM.from_pretrained(Modelo_5)
model_5.eval()

def summarize_text(text, tokenizer, model, max_input_length=1024, max_output_length=150):

    logger = logging.getLogger("API")
    logger.info(f"Iniciando sumarização de texto com {len(text)} caracteres")
    start_time = time.time()

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
        summary_ids = model.generate(
            **inputs,
            max_length=max_output_length,
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info(f"Sumarização concluída em {time.time() - start_time:.2f}s")
        return summary
    except Exception as e:
        logger.exception(f"Erro ao resumir texto: {e}")
        return text

def summarize_segments(data):
    logger.info("Iniciando sumarização de segmentos")
    start_time = time.time()
    resultado = {}
    segment_count = 0
    for key, value in data.items():
        if isinstance(value, str) and key.startswith("segmento"):
            segment_count += 1
            logger.info(f"Sumarizando segmento '{key}' ({len(value)} caracteres)")
            resultado[key] = summarize_text(value)
        else:
            resultado[key] = value  # mantém metadados
    logger.info(f"Total de segmentos processados: {segment_count}")
    logger.info(f"Sumarização de segmentos concluída em {time.time() - start_time:.2f}s")
    return resultado


# ---------------------- MODELO 6: Extração de Votos ----------------------


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

tokenizer_6 = AutoTokenizer.from_pretrained(Modelo_6, trust_remote_code=True)
model_6 = DebertaCRFForTokenClassification.from_pretrained(Modelo_6, trust_remote_code=True)
model_6.eval()


def extrair_entidades_votos(texto):
    logger.info(f"Iniciando extração de entidades de um texto com {len(texto)} caracteres")
    start_time = time.time()
    entities = {}
    try:
        inputs = tokenizer_6(texto, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            predictions = model_6.decode(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                tokenizer=tokenizer_6,
                text=texto
            )

        current_entity = []
        current_type = None

        for pred in predictions:
            label = pred["label"]
            word = pred["word"]

            if label.startswith("B-"):
                if current_entity:
                    entity_type = current_type.replace("B-", "").replace("I-", "")
                    entities.setdefault(entity_type, []).append(" ".join(current_entity))
                current_entity = [word]
                current_type = label
            elif label.startswith("I-") and current_entity:
                current_entity.append(word)
            else:
                if current_entity:
                    entity_type = current_type.replace("B-", "").replace("I-", "")
                    entities.setdefault(entity_type, []).append(" ".join(current_entity))
                current_entity = []
                current_type = None

        if current_entity:
            entity_type = current_type.replace("B-", "").replace("I-", "")
            entities.setdefault(entity_type, []).append(" ".join(current_entity))

        logger.info(f"Extração concluída em {time.time() - start_time:.2f}s com {len(entities)} tipos de entidades")
        return entities
    except Exception as e:
        logger.exception(f"Erro na extração de entidades: {e}")
        return {}


def processar_segmentos(data):
    logger.info("Iniciando processamento de segmentos para extração de votos")
    start_time = time.time()
    resultados = []
    segmentos = [v for k, v in data.items() if k.startswith("segmento")]
    logger.info(f"Total de segmentos a processar: {len(segmentos)}")

    for i, texto in enumerate(segmentos):
        logger.info(f"Processando segmento {i + 1} ({len(texto)} caracteres)")
        entidades = extrair_entidades_votos(texto)
        resultados.append({
            "segmento_id": i + 1,
            "texto": texto,
            "entidades": entidades
        })

    logger.info(f"Processamento de segmentos concluído em {time.time() - start_time:.2f}s")
    return resultados


# ---------------------- MODELO 7: Classificação de Tópicos ----------------------

tokenizer_7, model_7, thresholds_7, mlb_7 = get_modelo_classificacao_topicos()

def predict_topicos(text, tokenizer, model, thresholds, mlb, top_k=3, max_length=512, device=None):

    if not text or not text.strip():
        return []

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits.cpu()).numpy().ravel()

    preds_bool = (probs >= thresholds).astype(int)
    predicted_labels = list(mlb.inverse_transform(preds_bool.reshape(1, -1))[0])


    if not predicted_labels:
        top_indices = np.argsort(probs)[::-1][:top_k]
        predicted_labels = [mlb.classes_[i] for i in top_indices]

    return predicted_labels


def classificar_topicos(data, tokenizer, model, thresholds, mlb):
    logger.info("Iniciando classificação de tópicos dos segmentos")
    resultado = {}
    total_segmentos = 0

    for key, value in data.items():
        if isinstance(value, str) and key.startswith("segmento_"):
            total_segmentos += 1
            logger.info(f"Classificando segmento '{key}' ({len(value)} caracteres)")
            topicos = predict_topicos(value, tokenizer, model, thresholds, mlb)
            resultado[key] = value
            resultado[f"topico_{key.split('_')[1]}"] = topicos
        else:
            resultado[key] = value

    logger.info(f"Classificação de tópicos concluída para {total_segmentos} segmentos")
    return resultado


# ------------------------------- ROTAS ------------------------------------

@app.route("/", methods=["GET"])
def health_check():
    logger.info("Health check recebido.")
    return {"mensagem": "API Ativa"}, 200


@app.before_request
def log_request_info():
    logger.info(f"[{request.method}] {request.path} de {request.remote_addr} — tamanho entrada: {len(request.data)} bytes")


@app.after_request
def log_response_info(response):
    logger.info(f"{request.path} retornou {response.status_code}")
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception("Exceção não tratada:")
    return jsonify({"erro": str(e)}), 500


@app.route("/divisao_documento", methods=["POST"])
def rota_divisao_documento():
    logger.info("Rota /divisao_documento chamada")
    data = request.get_json()
    if not data or "texto_ata" not in data:
        logger.warning("JSON inválido recebido na rota /divisao_documento")
        return jsonify({"erro": "JSON inválido. Deve conter a chave 'texto_ata'."}), 400

    texto_original = data["texto_ata"].strip()
    if not texto_original:
        return jsonify({"erro": "Texto da ata vazio."}), 400

    # Usar seu modelo local já carregado
    tokenizer_qa, model_qa = get_modelo_divisao_documento()

    resultado_divisao = dividir_ata(tokenizer_qa, model_qa, texto_original)
    return jsonify(resultado_divisao), 200


@app.route("/extracao_metadados", methods=["POST"])
def rota_extracao_metadados():
    logger.info("Rota /extracao_metadados chamada")
    data = request.get_json()
    if not data or "metadados_iniciais" not in data or "metadados_finais" not in data:
        logger.warning("JSON inválido recebido na rota /extracao_metadados")
        return jsonify({"erro": "JSON inválido. Deve conter 'metadados_iniciais' e 'metadados_finais'"}), 400

    metadados_iniciais = data["metadados_iniciais"]
    metadados_finais = data["metadados_finais"]

    tokenizer_ner, extractor_ner = get_modelo_extracao_metadados()

    resultado_metadados = extrair_metadados(metadados_iniciais, metadados_finais, extractor_ner)
    logger.info("Extração de metadados concluída")
    return jsonify(resultado_metadados), 200


@app.route("/divisao_segmentos", methods=["POST"])
def rota_divisao_segmentos():
    logger.info("Rota /divisao_segmentos chamada")
    data = request.get_json()

    if not data or "corpo" not in data:
        return jsonify({"erro": "JSON inválido. Deve conter o campo 'corpo'"}), 400

    texto = data["corpo"].strip()
    if not texto:
        return jsonify({"erro": "O campo 'corpo' está vazio"}), 400

    # Preparar sentenças e segmentar usando o modelo local
    sentencas = preparar_sentencas_local(texto)
    segmentos = segmentar_texto(sentencas)

    # Monta resultado mantendo metadados
    resultado = {"metadados_iniciais": data.get("metadados_iniciais", {})}
    for i, seg in enumerate(segmentos, 1):
        resultado[f"segmento_{i}"] = seg
    resultado["metadados_finais"] = data.get("metadados_finais", {})

    logger.info(f"Total de segmentos gerados: {len(segmentos)}")
    return jsonify(resultado), 200


@app.route("/anonimizacao", methods=["POST"])
def rota_anonimizacao():
    logger.info("Rota /anonimizacao chamada")
    data = request.get_json()
    if not data:
        logger.warning("JSON inválido recebido na rota /anonimizacao")
        return jsonify({"erro": "JSON inválido"}), 400

    tokenizer_anon, ner_pipeline_anon = get_modelo_anonimizacao()

    if isinstance(data, list):
        data_anon = [anonimizar_ata(ata, ner_pipeline_anon) for ata in data]
        logger.info(f"Anonimização realizada em {len(data)} itens")
    else:
        data_anon = anonimizar_ata(data, ner_pipeline_anon)
        logger.info("Anonimização realizada em 1 item")

    return jsonify(data_anon), 200


@app.route("/sumarizacao_segmentos", methods=["POST"])
def rota_sumarizacao():
    logger.info("Rota /sumarizacao_segmentos chamada")
    data = request.get_json()

    if not data:
        logger.warning("JSON inválido recebido na rota /sumarizacao_segmentos")
        return jsonify({"erro": "JSON inválido"}), 400

    # Carrega o modelo de sumarização
    tokenizer_sum, model_sum = get_modelo_sumarizacao()

    summarized_data = {}
    try:
        for key, value in data.items():
            if isinstance(value, str) and key.startswith("segmento"):
                logger.info(f"Sumarizando segmento '{key}' ({len(value)} caracteres)")
                summarized_data[key] = summarize_text(value, tokenizer_sum, model_sum)
            else:
                summarized_data[key] = value  # mantém metadados

        logger.info("Sumarização de segmentos concluída")
        return jsonify(summarized_data), 200

    except Exception as e:
        logger.exception("Erro durante sumarização")
        return jsonify({"erro": str(e)}), 500


@app.route("/extracao_votos", methods=["POST"])
def rota_extracao_votos():
    logger.info("Rota /extracao_votos chamada")
    data = request.get_json()
    if not data:
        logger.warning("JSON inválido recebido na rota /extracao_votos")
        return jsonify({"erro": "JSON inválido"}), 400

    # Modelos já carregados globalmente, não precisa passar como argumento
    try:
        resultados = processar_segmentos(data)
        logger.info(f"Extração de votos concluída com {len(resultados)} segmentos processados")
        return jsonify(resultados), 200
    except Exception as e:
        logger.exception("Erro durante a extração de votos")
        return jsonify({"erro": str(e)}), 500


@app.route("/classificacao_topicos", methods=["POST"])
def rota_classificacao_topicos():
    logger.info("Rota /classificacao_topicos chamada")
    data = request.get_json()

    if not data:
        logger.warning("JSON inválido recebido na rota /classificacao_topicos")
        return jsonify({"erro": "JSON inválido"}), 400

    try:
        if isinstance(data, list):
            resultados = [
                classificar_topicos(item, tokenizer_7, model_7, thresholds_7, mlb_7)
                for item in data
            ]
            logger.info(f"Classificação de tópicos realizada em {len(data)} itens")
        else:
            resultados = classificar_topicos(data, tokenizer_7, model_7, thresholds_7, mlb_7)
            logger.info("Classificação de tópicos realizada em 1 item")

        return jsonify(resultados), 200

    except Exception as e:
        logger.exception("Erro durante a classificação de tópicos")
        return jsonify({"erro": str(e)}), 500


# ------------------------------- INICIALIZAÇÃO -------------------------------


if __name__ == "__main__":
    logger.info("Servidor Flask iniciado em 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5060, debug=False)
