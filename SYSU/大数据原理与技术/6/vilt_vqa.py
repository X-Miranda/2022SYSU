import requests
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering


# 修改为正确的本地路径
model_path = "./models/vilt-b32-finetuned-vqa"  # 需要重新下载正确的模型


processor = ViltProcessor.from_pretrained(model_path)
model = ViltForQuestionAnswering.from_pretrained(model_path)


def vqa(image_path, question):
    try:
        image = Image.open(image_path).convert("RGB")

        encoding = processor(image, question, return_tensors="pt", truncation=True, max_length=40)

        with torch.no_grad():
            outputs = model(**encoding)

        idx = outputs.logits.argmax(-1).item()
        return model.config.id2label[idx]
    except Exception as e:
        print(f"推理错误: {str(e)}")
        return "无法回答问题"


if __name__ == "__main__":
    local_image = r"C:\Users\85013\Desktop\a298552618de948b0c1554a7fe36d72.png"

    en_question = "What kind of dog is in the picture?"
    answer = vqa(local_image, en_question)
    print(f"Q: {en_question}\nA: {answer}")