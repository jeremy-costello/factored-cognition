# i over-commented this because it's doing a lot of things that aren't super clear
import re
import json

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from models import LLama2_7B_Chat_AWQ
from recipes import AuthorSplit


### these two are the only inputs
# path to the pdf
pdf_path = "./papers/2305.04843.pdf"
# whether to use an LLM to augment extraction
use_llm = False

# only consider something a paragraph if it starts with an english letter or number
normal_first_char_pattern = re.compile("[A-Za-z0-9]+")
# for identifying unnumbered sections
common_section_names = ["introduction", "references"]
# for identifying non-paragraphs (e.g. captions)
object_names = ["figure", "table", "equation", "algorithm"]
# for identifying whether a paragraph is complete
ending_characters = [".", "!", "?"]

# dictionary
paper_dict = dict()

# whether the title has been found
title_found = False
# whether the authors have been found
authors_found = False
# whether the abstract has been found
abstract_found = False
# for storing continuing paragraphs (e.g. across page breaks)
hanging_paragraph = ""
# for tracking the high-level section
current_high_level_section = 0
# blah
current_dictionary_section = 0

# load model if required
model = None
if use_llm:
    model = LLama2_7B_Chat_AWQ()

# list for storing paragraphs in the abstract
abstract_list = []
# trigger for breaking the outer loop
break_after = False
# blah
section_string = ""
# loop through pages in the pdf
for pagenum, page in enumerate(extract_pages(pdf_path)):
    # list of paragraphs to order by bounding box later
    paragraph_list = []
    # loop through elements in the page
    for elnum, element in enumerate(page):
        if isinstance(element, LTTextContainer):
            text = element.get_text()
            text_split = text.split("\n")
            average_line_length = sum([len(line) for line in text_split]) / len(text_split)
            if average_line_length >= 5:
                true_text = "".join([line.rstrip("-") if line.endswith("-") else line.strip() + " " for line in text_split]).strip()
                if not title_found:
                    paper_dict["title"] = true_text
                    title_found = True
                elif not authors_found:
                    if use_llm:
                        recipe = AuthorSplit()
                        author_lists = recipe.call_recipe(
                            prompts=[text],
                            model=model
                        )
                        author_list = author_lists[0]
                    else:
                        author_list = true_text
                    paper_dict["authors"] = author_list.lstrip("Answer:").strip()
                    authors_found = True
                elif true_text.lower() == "abstract":
                    continue
                elif true_text.lower() in common_section_names:
                    current_high_level_section += 1
                    paragraph_list.append(("section", current_high_level_section, element, true_text))
                else:
                    first_char = true_text[0]
                    first_word = true_text.split(" ")[0]
                    if not normal_first_char_pattern.fullmatch(first_char):
                        continue
                    if first_char in [str(char) for char in list(range(10))]:
                        first_number_maybe = first_word.split(".")[0]
                        try:
                            int(first_number_maybe)
                            section = True
                        except ValueError:
                            section = False
                        first_space = section_num = true_text.split(" ")[0]
                        if section:
                            section_num = first_space
                            current_high_level_section = int(section_num.split(".")[0])
                            section_name = " ".join(true_text.split(" ")[1:])
                            paragraph_list.append(("section", section_num, element, section_name))
                            continue
                        else:
                            footnote_split = first_space
                            try:
                                int(footnote_split)
                            except ValueError:
                                continue
                    if first_word.lower() in object_names:
                        object_number_maybe = true_text.split(" ")[1].rstrip(":").rstrip(".")
                        try:
                            int(object_number_maybe)
                            object_ = True
                        except ValueError:
                            object_ = False
                        if object_:
                            # f.write(f"{first_word} {object_number_maybe}\n\n")
                            continue
                    word_split = true_text.split(" ")
                    if len(word_split) < 5:
                        continue
                    
                    dot_split = true_text.split(".")
                    last_dot = dot_split[-1]
                    try:
                        last_int = int(last_dot)
                        strip = True
                    except ValueError:
                        strip = False
                    
                    if strip:
                        true_text = true_text.rstrip(str(last_int))
                    
                    if current_high_level_section == 0:
                        abstract_list.append(true_text)
                    else:
                        if not abstract_found:
                            paper_dict["abstract"] = "\n".join(abstract_list)
                            abstract_found = True
                        paragraph_list.append(("paragraph", None, element, true_text))
    
    paragraph_dict = {
        "left": dict(),
        "right": dict()
    }
    
    for paragraph_tuple in paragraph_list:
        element = paragraph_tuple[2]
        if element.x0 < 0.48 * page.x1:
            side = "left"
        else:
            side = "right"
        paragraph_dict[side][element.y1] = paragraph_tuple
    
    sorted_paragraph_list = \
        [value for key, value in sorted(paragraph_dict["left"].items(), key=lambda item: item[0], reverse=True)] + \
        [value for key, value in sorted(paragraph_dict["right"].items(), key=lambda item: item[0], reverse=True)]
    
    for p_type, section_num, element, true_text in sorted_paragraph_list:
        if true_text == paper_dict["title"]:
            continue
        
        if p_type == "section":
            if hanging_paragraph:
                section_string += f"{hanging_paragraph.strip()}\n\n"
                hanging_paragraph = ""
        
        if section_num is not None:
            if isinstance(section_num, str):
                high_level_section = int(section_num.split(".")[0])
            else:
                high_level_section = section_num
            
            if high_level_section != current_dictionary_section:
                if high_level_section > current_dictionary_section + 3 or high_level_section <= 0:
                    continue
                if current_dictionary_section == 0:
                    paper_dict["sections"] = dict()
                    current_section_name = true_text
                else:
                    paper_dict["sections"][current_dictionary_section] = {
                        "name": current_section_name,
                        "text": section_string
                    }
                    current_section_name = true_text
                    section_string = ""
                current_dictionary_section = high_level_section
                
        if p_type == "paragraph":
            last_char = true_text[-1]
            
            if last_char == "-":
                hanging_paragraph += true_text.rstrip("-")
            else:
                hanging_paragraph += true_text + " "
            
            if last_char in ending_characters:
                section_string += f"{hanging_paragraph.strip()}\n\n"
                hanging_paragraph = ""
    
paper_dict["sections"][current_dictionary_section] = {
    "name": current_section_name,
    "text": section_string
}
    
json_path = ".".join(pdf_path.split(".")[:-1]) + ".json"
with open(json_path, "w") as json_file:
    json.dump(paper_dict, json_file, indent=4)
