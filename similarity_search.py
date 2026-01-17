## create embeddings / link supabase

from supabase import create_client
import clip
import torch
from PIL import Image
import numpy as np

SUPABASE_URL = "https://bpqtlzsqqjdzpwcddqxj.supabase.co"
SUPABASE_KEY = "sb_secret_CTD_8ZqN0sj9vRhIxwdE1Q_-hCDz_Ql"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# print(response.data)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device = device)

def create_image_embeddings(image_path: str) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding /= embedding.norm(dim = -1, keepdim = True)

    return embedding[0].cpu().numpy().tolist()

def create_text_prompt_embeddings(prompt: str) -> list[float]:
    text_input = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(text_input)
        embedding /= embedding.norm(dim = -1, keepdim = True)

    return embedding[0].cpu().numpy().tolist()

#embedding = create_image_embeddings("/Users/katherinelee/dev-code/logo-generator/logo-examples/1de4de6020c4e4fe5867d84db6379df7.jpg")

#create image embeddings for all images in supabase / hacky right now 
response = supabase.table("logo-examples").select("*").execute()

# for row in response.data:
#     image_url = row["image-url"]
#     filename = image_url.split("/")[-1]
#     full_url = "/Users/katherinelee/dev-code/logo-generator/logo-examples/" + filename
#     embedding = create_image_embeddings(full_url)
#     response = (
#         supabase
#         .table("logo-examples")
#         .update({"clip-embeddings": embedding})
#         .eq("id", row["id"])
#         .execute()
#     )

text_embedding = create_text_prompt_embeddings("a logo with the people project")

def similarity_search(text_embedding: list[float], image_embeddings: list[list[float]], top_k = 5):

    text_emb = np.array(text_embedding)
    text_emb = text_emb / np.linalg.norm(text_emb)

    img_embs = np.array(image_embeddings)
    img_embs = img_embs / np.linalg.norm(img_embs, axis = 1, keepdims = True) 
    similarities = img_embs @ text_emb

    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]

    return list(zip(top_indices, top_scores))

results = similarity_search(text_embedding, [row["clip-embeddings"] for row in response.data], top_k = 1)
print(results)
for index, score in results:
    print(f"Top image: {response.data[index]['image-url']}, similarity: {score:.4f}")

