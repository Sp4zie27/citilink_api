from flask import Flask, request, jsonify
import torch
import warnings
import re
import logging
import os
import sys
from transformers import BertTokenizer, BertForQuestionAnswering, AutoModelForNextSentencePrediction, AutoModelForTokenClassification, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from modelos.extracao_votos.modeling_deberta_crf import DebertaCRFForTokenClassification
import time
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np
from joblib import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        logger.info("Carregando modelo Divisão de Documento")
        tokenizer = BertTokenizer.from_pretrained(Modelo_1)
        model = BertForQuestionAnswering.from_pretrained(Modelo_1)
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
        logger.info("Carregando modelo de Segmentação de Texto")
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


def normalizar_texto_comparacao(texto):
    try:
        texto = re.sub(r"\n+", " ", texto)
        texto = re.sub(r"\s+", " ", texto)
        texto = texto.strip().lower()
        texto = re.sub(r"[.:;,]+", " ", texto)
        texto = re.sub(r"\s+", " ", texto)
        logger.debug("Texto normalizado com sucesso")
        return texto
    except Exception as e:
        logger.exception(f"Erro ao normalizar texto: {e}")
        return texto

def mapear_posicao_para_original(texto_original, texto_busca, pos_normalizada):
    try:
        texto_norm = normalizar_texto_comparacao(texto_original)
        palavras_antes = len(texto_norm[:pos_normalizada].split())
        palavras_orig = re.findall(r"\S+", texto_original)
        if palavras_antes >= len(palavras_orig):
            logger.warning("Número de palavras antes maior que original")
            return None
        palavra_alvo = palavras_orig[palavras_antes] if palavras_antes < len(palavras_orig) else None
        if palavra_alvo:
            pos_inicio = texto_original.find(palavra_alvo)
            if pos_inicio != -1:
                pos_fim = pos_inicio
                for i in range(palavras_antes, min(palavras_antes + 10, len(palavras_orig))):
                    pos_fim = texto_original.find(palavras_orig[i], pos_fim) + len(palavras_orig[i])
                logger.info(f"Posição mapeada: {pos_inicio}-{pos_fim}")
                return (pos_inicio, pos_fim)
    except Exception as e:
        logger.exception(f"Erro ao mapear posição: {e}")
    return None

def encontrar_posicao_avancada(texto_original, marcador):
    try:
        palavras_marcador = re.findall(r"\b[A-ZÀÁÂÃÇÉÊÍÓÔÕÚ][A-ZÀÁÂÃÇÉÊÍÓÔÕÚa-zàáâãçéêíóôõú]+\b", marcador)
        if len(palavras_marcador) >= 3:
            for tamanho in [5,4,3]:
                if len(palavras_marcador) >= tamanho:
                    busca = " ".join(palavras_marcador[:tamanho])
                    pattern = re.compile(re.escape(busca), re.IGNORECASE)
                    match = pattern.search(texto_original)
                    if match:
                        inicio = match.start()
                        fim = match.end()
                        resto = texto_original[fim:]
                        match_fim = re.search(r"[.]\s*(?=[A-ZÀÁÂÃÇÉÊÍÓÔÕÚ\n]|$)", resto)
                        if match_fim:
                            fim += match_fim.end()
                        logger.info(f"Marcador encontrado na posição {inicio}-{fim}")
                        return (inicio, fim)
        frases = re.split(r"[.!?]\s+", marcador)
        for frase in frases:
            if len(frase.strip()) > 20:
                frase_limpa = frase.strip()
                palavras = frase_limpa.split()[:8]
                busca = " ".join(palavras)
                busca_norm = normalizar_texto_comparacao(busca)
                texto_norm = normalizar_texto_comparacao(texto_original)
                pos = texto_norm.find(busca_norm)
                if pos != -1:
                    resultado = mapear_posicao_para_original(texto_original, busca_norm, pos)
                    logger.info(f"Marcador aproximado encontrado: {resultado}")
                    return resultado
        logger.warning("Marcador não encontrado")
    except Exception as e:
        logger.exception(f"Erro ao procurar marcador: {e}")
    return None

def carregar_modelo_qa(model_path):
    start_time = time.time()
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForQuestionAnswering.from_pretrained(model_path)
        model.eval()
        logger.info(f"Modelo carregado em {time.time()-start_time:.2f}s")
        return tokenizer, model
    except Exception as e:
        logger.exception(f"Erro ao carregar modelo: {e}")
        raise

def responder_pergunta_chunks(tokenizer, model, contexto, pergunta, chunk_size=500, stride=20):
    logger.info(f"Respondendo pergunta: '{pergunta[:50]}...'")
    start_time = time.time()
    respostas = []
    try:
        all_tokens = tokenizer.tokenize(contexto)
        start = 0
        while start < len(all_tokens):
            chunk_tokens = all_tokens[start:start+chunk_size]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            inputs = tokenizer.encode_plus(
                pergunta,
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=chunk_size
            )
            with torch.no_grad():
                outputs = model(**inputs)
            start_scores, end_scores = outputs.start_logits, outputs.end_logits
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)+1
            tokens_resposta = inputs["input_ids"][0][start_index:end_index]
            resposta = tokenizer.decode(tokens_resposta, skip_special_tokens=True).strip()
            if resposta:
                respostas.append(resposta)
            start += chunk_size - stride
        logger.info(f"Resposta gerada em {time.time()-start_time:.2f}s")
        return " ".join(respostas)
    except Exception as e:
        logger.exception(f"Erro ao responder pergunta: {e}")
        return ""

def dividir_ata(texto_original, opening_marker, closing_marker):
    logger.info("Iniciando divisão da ata")
    try:
        pos_opening = encontrar_posicao_avancada(texto_original, opening_marker)
        pos_closing = encontrar_posicao_avancada(texto_original, closing_marker)
        if not pos_opening or not pos_closing:
            logger.warning("Marcadores não encontrados, retornando texto completo")
            return {"metadados_iniciais": "", "corpo": texto_original, "metadados_finais": ""}
        inicio_opening, fim_opening = pos_opening
        inicio_closing, fim_closing = pos_closing
        if inicio_closing <= fim_opening:
            logger.warning("Marcadores invertidos, retornando texto completo")
            return {"metadados_iniciais": "", "corpo": texto_original, "metadados_finais": ""}
        metadados_iniciais = texto_original[:fim_opening].strip()
        corpo = texto_original[fim_opening:inicio_closing].strip()
        metadados_finais = texto_original[inicio_closing:].strip()
        logger.info("Divisão da ata concluída")
        return {"metadados_iniciais": metadados_iniciais, "corpo": corpo, "metadados_finais": metadados_finais}
    except Exception as e:
        logger.exception(f"Erro ao dividir a ata: {e}")
        return {"metadados_iniciais": "", "corpo": texto_original, "metadados_finais": ""}



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


tokenizer_3 = AutoTokenizer.from_pretrained(Modelo_3)
model_3 = AutoModelForNextSentencePrediction.from_pretrained(Modelo_3)
model_3.eval()

def preparar_sentencas(texto):
    logger.info("Preparando sentenças")
    start_time = time.time()
    try:
        texto = re.sub(r'(\d+)\.\s+', r'\1§§§', texto)

        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            nltk.download('punkt', quiet=True)
            sentencas = sent_tokenize(texto)
        except Exception:
            sentencas = re.split(r'(?<=[.!?])\s+', texto.strip())

        sentencas = [s.replace('§§§', '. ') for s in sentencas]

        sentencas_limpa = []
        i = 0
        while i < len(sentencas):
            s = sentencas[i].strip()
            if re.match(r'^\d+\.$', s) and i + 1 < len(sentencas):
                sentencas_limpa.append(f"{s} {sentencas[i + 1].strip()}")
                i += 2
            else:
                sentencas_limpa.append(s)
                i += 1

        sentencas_final = [s for s in sentencas_limpa if s]
        logger.info(f"Preparação concluída: {len(sentencas_final)} sentenças encontradas em {time.time()-start_time:.2f}s")
        return sentencas_final
    except Exception as e:
        logger.exception(f"Erro ao preparar sentenças: {e}")
        return []

def segmentar_texto(sentencas, tokenizer, model, threshold=0.65):

    logger = logging.getLogger("API")
    logger.info(f"Iniciando segmentação de texto com threshold={threshold}")
    start_time = time.time()

    segmentos = []
    try:
        if not sentencas:
            logger.warning("Nenhuma sentença fornecida para segmentação")
            return segmentos

        buffer = [sentencas[0]]
        for i in range(len(sentencas) - 1):
            sent_a = sentencas[i]
            sent_b = sentencas[i + 1]

            inputs = tokenizer(sent_a, sent_b, return_tensors="pt", truncation=True, max_length=500)

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                not_next_prob = probs[0][1].item()

            logger.debug(f"Sentença {i}-{i+1} | Probabilidade não sequencial: {not_next_prob:.4f}")

            if not_next_prob > threshold:
                segmentos.append(" ".join(buffer))
                logger.info(f"Novo segmento criado com {len(buffer)} sentenças")
                buffer = [sent_b]
            else:
                buffer.append(sent_b)

        if buffer:
            segmentos.append(" ".join(buffer))
            logger.info(f"Último segmento criado com {len(buffer)} sentenças")

        logger.info(f"Segmentação concluída: {len(segmentos)} segmentos em {time.time()-start_time:.2f}s")
        return segmentos

    except Exception as e:
        logger.exception(f"Erro ao segmentar texto: {e}")
        return []



# ---------------------- MODELO 4: Anonimização ----------------------

tokenizer_4 = AutoTokenizer.from_pretrained(Modelo_4)
model_4 = AutoModelForTokenClassification.from_pretrained(Modelo_4)
ner_pipeline_4 = pipeline(
    "ner",
    model=model_4,
    tokenizer=tokenizer_4,
    aggregation_strategy="simple"
)


def normalizar_nome(nome):
    nome = re.sub(r"\(.*?\)", "", nome)  # remove conteúdo entre parênteses
    nome = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\s]", "", nome)  # remove pontuação
    return nome.strip().lower()

def eh_nome_participante(ent_texto, participantes_norm, limiar=0.6):
    ent_norm = normalizar_nome(ent_texto)
    for nome_part in participantes_norm:
        ratio = SequenceMatcher(None, ent_norm, nome_part).ratio()
        if ratio >= limiar:
            return True
        if ent_norm in nome_part or nome_part in ent_norm:
            return True
    return False

def dividir_texto(texto, max_tokens=500, overlap=20):
    logger.info(f"Dividindo texto em chunks (max_tokens={max_tokens}, overlap={overlap})")
    tokens = tokenizer_4.encode(texto, add_special_tokens=False)
    passo = max_tokens - overlap
    for i in range(0, len(tokens), passo):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer_4.decode(chunk_tokens, skip_special_tokens=True)
        prefix_tokens = tokens[:i]
        prefix_text = tokenizer_4.decode(prefix_tokens, skip_special_tokens=True)
        yield chunk_text, len(prefix_text)


def anonimizar_texto(texto, ner_pipeline, participantes_norm=None):
    if not isinstance(texto, str):
        logger.warning("Texto não é string. Retornando valor original.")
        return texto

    logger.info(f"Iniciando anonimização de texto com {len(texto)} caracteres")
    start_time = time.time()
    entidades = []

    try:
        chunk_count = 0
        for chunk, offset in dividir_texto(texto):
            chunk_count += 1
            chunk_entidades = ner_pipeline_4(chunk)
            logger.info(f"Chunk {chunk_count}: {len(chunk_entidades)} entidades encontradas")
            for ent in chunk_entidades:
                ent["start"] += offset
                ent["end"] += offset
            entidades.extend(chunk_entidades)

        if not entidades:
            logger.info("Nenhuma entidade encontrada. Retornando texto original")
            return texto

        entidades = sorted(entidades, key=lambda x: x["start"])
        palavras = list(re.finditer(r"\S+", texto))
        entidades_expandidas = []

        for ent in entidades:
            start, end = ent["start"], ent["end"]
            for match in palavras:
                if (match.start() <= start < match.end()) or (match.start() < end <= match.end()):
                    ent["start"] = min(ent["start"], match.start())
                    ent["end"] = max(ent["end"], match.end())
            entidades_expandidas.append(ent)

        entidades_expandidas = sorted(entidades_expandidas, key=lambda x: x["start"])
        mescladas = []
        for ent in entidades_expandidas:
            if not mescladas:
                mescladas.append(ent)
            else:
                last = mescladas[-1]
                if ent["start"] <= last["end"]:
                    last["end"] = max(last["end"], ent["end"])
                else:
                    mescladas.append(ent)

        logger.info(f"{len(mescladas)} entidades mescladas no total")
        texto_anonimizado = list(texto)

        for ent in reversed(mescladas):
            start, end = ent["start"], ent["end"]
            if end - start < 3:
                continue

            ent_texto = texto[start:end]

            if participantes_norm and eh_nome_participante(ent_texto, participantes_norm):
                logger.info(f"Nome de participante detectado, não anonimizado: {ent_texto}")
                continue

            sub_text = texto[start:end]
            tokens = re.findall(r"\S+", sub_text)
            sub_anon = " ".join(["*****" for _ in tokens])

            texto_anonimizado[start:end] = list(sub_anon)

        logger.info(f"Anonimização concluída em {time.time()-start_time:.2f}s")
        return "".join(texto_anonimizado)

    except Exception as e:
        logger.exception(f"Erro durante anonimização: {e}")
        return texto


def anonimizar_ata(data, ner_pipeline):
    logger.info("Iniciando anonimização de ata")

    participantes_norm = []
    if "participantes" in data and isinstance(data["participantes"], list):
        participantes_norm = [normalizar_nome(p) for p in data["participantes"]]

    resultado = {}
    for key, value in data.items():
        if isinstance(value, str):
            resultado[key] = anonimizar_texto(value, ner_pipeline, participantes_norm)
        else:
            resultado[key] = value

    logger.info("Anonimização de ata concluída")
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
    logger.info("🩺 Health check recebido.")
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
    logger.info(f"Tamanho do texto recebido: {len(texto_original)} caracteres")
    if not texto_original:
        return jsonify({"erro": "Texto da ata vazio."}), 400

    tokenizer_qa, model_qa = get_modelo_divisao_documento()

    opening_marker = responder_pergunta_chunks(
        tokenizer_qa, model_qa, texto_original,
        "No início da ata há um segmento de abertura. Qual é a última frase desse segmento de abertura?"
    )
    closing_marker = responder_pergunta_chunks(
        tokenizer_qa, model_qa, texto_original,
        "No final da ata há um segmento de encerramento. Qual é a primeira frase desse segmento de encerramento?"
    )

    resultado_divisao = dividir_ata(texto_original, opening_marker, closing_marker)
    logger.info("Divisão do documento concluída")
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
        logger.warning("JSON inválido recebido na rota /divisao_segmentos")
        return jsonify({"erro": "JSON inválido. Deve conter o campo 'corpo'"}), 400

    texto = data["corpo"].strip()
    logger.info(f"Tamanho do corpo recebido: {len(texto)} caracteres")
    if not texto:
        return jsonify({"erro": "O campo 'corpo' está vazio"}), 400

    tokenizer_seg, model_seg = get_modelo_divisao_segmentos()

    sentencas = preparar_sentencas(texto)
    segmentos = segmentar_texto(sentencas, tokenizer_seg, model_seg)
    logger.info(f"Total de segmentos gerados: {len(segmentos)}")

    resultado = {"metadados_inicial": data.get("metadados_iniciais", {})}
    for i, seg in enumerate(segmentos, 1):
        resultado[f"segmento_{i}"] = seg
    resultado["metadados_final"] = data.get("metadados_finais", {})

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
