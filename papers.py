from pypdf import PdfReader


paper = "./papers/2402.05546.pdf"

reader = PdfReader(paper)
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()

text_split = text.split("\n")
for t in text_split:
    print(t)
