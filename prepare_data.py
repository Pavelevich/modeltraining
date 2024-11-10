import os
import re

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    # Remove extra spaces and unwanted characters
    text = re.sub(r'\s+', ' ', text)
    # Adjust this regular expression according to your needs
    text = re.sub(r'[^A-Za-z0-9ÁÉÍÓÚáéíóúÑñüÜ\s.,]', '', text)
    text = text.strip()
    return text

def segment_text(text, max_length=512):
    sentences = re.split(r'(?<=[.!?]) +', text)
    segments = []
    current_segment = ""
    for sentence in sentences:
        if len(current_segment) + len(sentence) + 1 <= max_length:
            current_segment += " " + sentence if current_segment else sentence
        else:
            segments.append(current_segment)
            current_segment = sentence
    if current_segment:
        segments.append(current_segment)
    return segments

def load_and_preprocess_data(directory):
    all_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            text = extract_text_from_txt(file_path)
            text = preprocess_text(text)
            all_text += text + " "
    return all_text

def main():
    directory = r"C:\Users\devel\Desktop\pdfToToken\pdfToToken\tokens"
    all_text = load_and_preprocess_data(directory)
    segments = segment_text(all_text)

    with open('segments.txt', 'w', encoding='utf-8') as f:
        for segment in segments:
            f.write(segment + "\n")

    print(f"Total segments generated: {len(segments)}")

if __name__ == "__main__":
    main()
