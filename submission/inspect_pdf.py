import PyPDF2
import os

pdf_path = "AI_ML intern Assignment.pdf"
output_path = "pdf_content.txt"

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"Number of pages: {len(reader.pages)}\n")
            for i, page in enumerate(reader.pages):
                out_file.write(f"--- Page {i+1} ---\n")
                text = page.extract_text()
                out_file.write(text)
                out_file.write("\n")
    print(f"Successfully wrote PDF content to {output_path}")
except Exception as e:
    print(f"Error reading PDF: {e}")
