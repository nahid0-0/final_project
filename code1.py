import tkinter as tk
from tkinter import scrolledtext
from transformers import MarianMTModel, MarianTokenizer
from PyDictionary import PyDictionary

class LanguageLearningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Language Learning Platform")
        
        # Create and place text widgets
        self.create_widgets()

        # Initialize components
        self.translation = Translation()
        self.dictionary = Dictionary()

    def create_widgets(self):
        # Input text area
        self.input_label = tk.Label(self.root, text="Enter text:")
        self.input_label.pack()
        self.input_text_area = scrolledtext.ScrolledText(self.root, height=10)
        self.input_text_area.pack()

        # Output text area
        self.output_label = tk.Label(self.root, text="Translation:")
        self.output_label.pack()
        self.output_text_area = scrolledtext.ScrolledText(self.root, height=10)
        self.output_text_area.pack()

        # Buttons
        self.translate_button = tk.Button(self.root, text="Translate", command=self.on_translate)
        self.translate_button.pack()
        
        self.dict_button = tk.Button(self.root, text="Dictionary Lookup", command=self.on_dict_lookup)
        self.dict_button.pack()

    def on_translate(self):
        input_text = self.input_text_area.get("1.0", tk.END).strip()
        if input_text:
            translated_text = self.translation.contextual_translate(input_text)
            self.output_text_area.delete("1.0", tk.END)
            self.output_text_area.insert(tk.END, translated_text)

    def on_dict_lookup(self):
        input_text = self.input_text_area.get("1.0", tk.END).strip()
        if input_text:
            meaning = self.dictionary.lookup(input_text)
            self.output_text_area.delete("1.0", tk.END)
            self.output_text_area.insert(tk.END, meaning)

class Translation:
    def __init__(self):
        self.model_name = 'Helsinki-NLP/opus-mt-en-fr'
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def contextual_translate(self, text, src_lang='en', tgt_lang='fr'):
        self.tokenizer.src_lang = src_lang
        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = self.model.generate(**encoded_text)
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0]

class Dictionary:
    def __init__(self):
        self.dictionary = PyDictionary()

    def lookup(self, word):
        meaning = self.dictionary.meaning(word)
        if meaning:
            meaning_str = ""
            for key, value in meaning.items():
                meaning_str += f"{key}: {', '.join(value)}\n"
            return meaning_str
        else:
            return "No meaning found."

if __name__ == "__main__":
    root = tk.Tk()
    app = LanguageLearningApp(root)
    root.mainloop()
