# ============================================================
# visualisasi.py — Gradio app untuk deteksi AI vs Human (RAID)
# ============================================================

import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
import gradio as gr


# ------------------------------------------------------------
# 1️⃣ Helper timing function
# ------------------------------------------------------------
def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


# ------------------------------------------------------------
# 2️⃣ Setup Spark Session (local mode untuk inference)
# ------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("RAID-INFERENCE-GRADIO")
    .master("local[*]")
    .getOrCreate()
)

# ------------------------------------------------------------
# 3️⃣ Load model pipeline dari lokal
# ------------------------------------------------------------
MODEL_PATH = os.path.abspath("model-minilm-lr_human-gpt4")
print(f"[INFO] Memuat model dari: {MODEL_PATH}")
model = PipelineModel.load(MODEL_PATH)
print("✅ Model berhasil dimuat.")


# ------------------------------------------------------------
# 4️⃣ Load MiniLM (tokenizer & encoder)
# ------------------------------------------------------------
print("[INFO] Memuat MiniLM encoder...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
encoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
print(f"✅ MiniLM siap digunakan (device: {device})")


# ------------------------------------------------------------
# 5️⃣ Fungsi ekstraksi embedding
# ------------------------------------------------------------
def get_embedding(text: str):
    """Ekstraksi embedding 384-dimensi dari teks menggunakan MiniLM"""
    if not text or text.strip() == "":
        return [0.0] * 384
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = encoder(**encoded)
    token_embeddings = output.last_hidden_state
    attention_mask = encoded["attention_mask"]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()[0].tolist()


# ------------------------------------------------------------
# 6️⃣ Fungsi klasifikasi
# ------------------------------------------------------------
def classify_text(text: str):
    if not text or text.strip() == "":
        return "Please enter some text.", ""

    emb, t_emb = measure_time(lambda: get_embedding(text))
    df, t_df = measure_time(lambda: spark.createDataFrame([Row(features=Vectors.dense(emb))]))
    pred_row, t_pred = measure_time(lambda: model.transform(df).select("prediction", "probability").collect()[0])

    label_indexer = model.stages[0]  # StringIndexer pertama
    labels = label_indexer.labels     # contoh: ['gpt4', 'human']

    pred_idx = int(pred_row["prediction"])
    pred_label = labels[pred_idx]
    probs = {labels[i]: float(pred_row["probability"][i]) for i in range(len(labels))}

    total_time = t_emb + t_df + t_pred

    # Mapping nama label untuk output lebih natural
    label_human = "Human" if pred_label.lower() == "human" else "AI"

    output_text = (
        f'Input Text: "{text[:200]}{"..." if len(text) > 200 else ""}"\n\n'
        f"Predicted Label: {label_human}\n\n"
        f"Probabilities:\n"
        f"- Human: {probs.get('human', 0)*100:.2f}%\n"
        f"- AI: {probs.get('gpt4', 0)*100:.2f}%\n\n"
        f"Execution Time: {total_time:.4f} detik"
    )

    return output_text, label_human


# ------------------------------------------------------------
# 7️⃣ Gradio Interface
# ------------------------------------------------------------
description = "Masukkan teks apa pun untuk mendeteksi apakah teks tersebut ditulis oleh **Manusia** atau **AI**."

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(
        label="Masukkan Teks",
        placeholder="Ketik teks di sini...",
        lines=10
    ),
    outputs=[
        gr.Textbox(label="Hasil Lengkap", lines=8),
        gr.Label(label="Predicted Label")
    ],
    title="AI vs Human Text Detection Using MiniLM & Logistic Regression",
    description=description,
    examples=[
        ["In this paper, we present a semi-supervised learning algorithm for classification of text documents. A method of labeling unlabeled text documents is presented. The presented method is based on the principle of divide and conquer strategy. It uses recursive K-means algorithm for partitioning both labeled and unlabeled data collection. The K-means algorithm is applied recursively on each partition till a desired level partition is achieved such that each partition contains labeled documents of a single class. Once the desired clusters are obtained, the respective cluster centroids are considered as representatives of the clusters and the nearest neighbor rule is used for classifying an unknown text document. Series of experiments have been conducted to bring out the superiority of the proposed model over other recent state of the art models on 20Newsgroups dataset."],
        ["Writing radiology reports based on radiographic images is a time-consuming task that demands the expertise of skilled radiologists. Consequently, the integration of technology capable of automated report generation would be advantageous. Developing a coherent predictive text is the main challenge in automatic report generation. It is necessary to develop methods that can increase the relevance of features in producing predictive text. This study constructed a medical report generator model using the transformer approach and image enhancement implementation. To leverage the visual and semantic features, an approach to enhance the noise-prone nature of the medical image is explored in this study along with the transformers method to generate a radiology report based on Chest X-ray images. Four contrast-based image enhancement methods were used to investigate the effect of image enhancement techniques on the radiology report generator. The encoder-decoder model is used with text feature embedding using Bidirectional Encoder Representation from Transformer (BERT) and visual feature extraction utilizing a pre-trained model ChexNet and Multi-Head Attention (MHA) mechanism. The performance of the MHA model with gamma correction is 5% in better with a 0.377 value using the Bilingual Assessment Understudy (BLEU) with 4 n-gram evaluation. MHA also produces 15% better results with a 0.412 value than the baseline model. This method is able to outperform the baseline model and other previous works. It can be concluded that the use of transformer MHA encoder layer and BERT is effective in leveraging visual and text features. Additionally, the inclusion of an image enhancement approach has been found to have a positive impact on the model’s performance."]
    ],
    allow_flagging="never"
)

# ------------------------------------------------------------
# 8️⃣ Jalankan aplikasi
# ------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_api=False)
