# i over-commented this because it's doing a lot of things that aren't super clear
import re

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from typing import Dict, Any, List, Tuple, Union

from models import LLama2_7B_Chat_AWQ
from recipes import AuthorSplit


def extract_paper_from_pdf(pdf_path: str, use_llm: bool) -> Dict[str, Union[List, Any]]:
    """Extracts a dictionary of title, authors, abstract, sections from a paper in pdf format.

    Args:
        pdf_path (str): Path to the pdf file.
        use_llm (bool): Whether to use additional LLM augmentation for extraction.

    Returns:
       Dict[str, Union[List, Any]]: Title, authors, abstract, sections.
    """

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
    # dictionary high-level section for tracking section changes
    dictionary_high_level_section = 0

    # load model if required
    model = None
    if use_llm:
        model = LLama2_7B_Chat_AWQ()

    # list for storing paragraphs in the abstract
    abstract_list = []
    # list of paragraphs for the current section
    section_paragraph_list = []
    # loop through pages in the pdf
    for pagenum, page in enumerate(extract_pages(pdf_path)):
        # list of paragraphs to order by bounding box later
        paragraph_list = []
        # loop through elements in the page
        for element in page:
            # if element is a text container
            if isinstance(element, LTTextContainer):
                text = element.get_text()
                # split text by newline
                text_split = text.split("\n")
                # average length of lines
                average_line_length = sum([len(line) for line in text_split]) / len(text_split)
                # if average length of lines is less than 3, ignore.
                if average_line_length >= 3:
                    # join lines by spaces or by removing the - if there is a line continuation through a word
                    true_text = "".join([line.rstrip("-") if line.endswith("-") else line.strip() + " " for line in text_split]).strip()
                    # assume first valid text container is the title
                    if not title_found:
                        paper_dict["title"] = [true_text]
                        title_found = True
                    # assume second valid text container is the list of authors
                    elif not authors_found:
                        if use_llm:
                            # LLM will take the unformatted list of authors and return a formatted list of authors (full names separated by commas)
                            recipe = AuthorSplit()
                            author_lists = recipe.call_recipe(
                                prompts=[text],
                                model=model
                            )
                            author_list = author_lists[0]
                        else:
                            # unformatted list of authors
                            author_list = true_text
                        paper_dict["authors"] = [author_list.lstrip("Answer:").strip()]
                        authors_found = True
                    # skip section name if the name is abstract
                    elif true_text.lower() == "abstract":
                        continue
                    # common section names that may be unnumberged (e.g. introduction, references)
                    elif true_text.lower() in common_section_names:
                        current_high_level_section += 1
                        paragraph_list.append(("section", current_high_level_section, element, true_text, pagenum))
                    else:
                        # first character
                        first_char = true_text[0]
                        # first word
                        first_word = true_text.split(" ")[0]
                        # skip if the first character does not match the regex pattern
                        if not normal_first_char_pattern.fullmatch(first_char):
                            continue
                        # if first character is a number
                        if first_char in [str(char) for char in list(range(10))]:
                            first_number_maybe = first_word.split(".")[0]
                            # this will separate between sections, footnotes, and normal paragraphs
                            # sections are 1.1.1, footnotes are 1Word, paragraphs would be starting with a number (e.g. 100)
                            try:
                                int(first_number_maybe)
                                section = True
                            except ValueError:
                                section = False
                            if section:
                                # section (e.g. 1.1.1)
                                section_num = first_word
                                # high-level section (e.g. 1)
                                current_high_level_section = int(section_num.split(".")[0])
                                section_name_list = true_text.split(" ")[1:]
                                # assumes section names are shorter than 10 words
                                if len(section_name_list) < 10:
                                    section_name = " ".join(section_name_list)
                                    paragraph_list.append(("section", section_num, element, section_name, pagenum))
                                    continue
                            else:
                                # possible footnote (e.g. 1Word) or normal paragraph (e.g. 100)
                                footnote_split = first_word
                                try:
                                    # if the first word is a number (e.g. 100), assume it is a normal paragraph
                                    int(footnote_split)
                                except ValueError:
                                    # if the first word is not a number (e.g. 1Word), assume it is a footnote
                                    continue
                        # paper object (e.g. figure, table, algorithm, equation). usually a caption
                        if first_word.lower() in object_names:
                            # strip colon and dot from the second word of the potential caption (e.g. 1: in Figure 1:)
                            object_number_maybe = true_text.split(" ")[1].rstrip(":").rstrip(".")
                            try:
                                # if stripped second word is a number, assume it is an object (caption) and skip
                                int(object_number_maybe)
                                continue
                            except ValueError:
                                # else, assume it is a normal paragraph
                                pass
                        
                        # skip if text is less than 3 words
                        word_split = true_text.split(" ")
                        if len(word_split) < 3:
                            continue
                        
                        # strip any footnote notations from the end of the text (e.g. strip 1 from word.1)
                        dot_split = true_text.split(".")
                        last_dot = dot_split[-1]
                        try:
                            last_int = int(last_dot)
                            true_text = true_text.rstrip(str(last_int))
                        except ValueError:
                            pass
                        
                        if current_high_level_section == 0 and not abstract_found:
                            # if no sections have been found, assume the text is part of the abstract
                            abstract_list.append(true_text)
                        else:
                            if not abstract_found:
                                # upon finding first section, write all previous text (besides title and authors) as abstract
                                paper_dict["abstract"] = abstract_list
                                abstract_found = True
                            paragraph_list.append(("paragraph", None, element, true_text, pagenum))
        
        # dictionary for splitting two-column papers into left column and right column
        paragraph_dict = {
            "left": dict(),
            "right": dict()
        }
        
        # iterate through all tuples in the paragraph list
        for paragraph_tuple in paragraph_list:
            # extract element from the tuple
            element = paragraph_tuple[2]
            if element.x0 < 0.48 * page.x1:
                # assume left column if the element starts on the left 48% of the page
                side = "left"
            else:
                # else, assume right column
                side = "right"
            
            # add to side dict with key as y-coordinate of the top of the element
            # pdfminer has y-coordinates start (at 0) from the bottom of the page
            paragraph_dict[side][element.y1] = paragraph_tuple
        
        # each side dict by element top y-coordinate (largest to smallest)
        sorted_paragraph_list = \
            [value for key, value in sorted(paragraph_dict["left"].items(), key=lambda item: item[0], reverse=True)] + \
            [value for key, value in sorted(paragraph_dict["right"].items(), key=lambda item: item[0], reverse=True)]
        
        # iterate through tuples:
        ## p_type is text type (section or paragraph)
        ## section_num is the number of the section, or None
        ## element is the pdfminer text element
        ## true_text is the processed text from the element
        ## p_pagenum is the page number the text was extracted from
        for p_type, section_num, element, true_text, p_pagenum in sorted_paragraph_list:
            # if the text is the title, skip
            if true_text == paper_dict["title"]:
                continue
            
            # if new section and there is still a hanging paragraph, add the hanging paragraph to the section string
            if p_type == "section" and hanging_paragraph:
                section_paragraph_list.append(hanging_paragraph.strip())
                hanging_paragraph = ""
            
            if section_num is not None:
                if isinstance(section_num, str):
                    # convert section string (e.g. 1.1.1) to a high-level section (e.g. 1)
                    high_level_section = int(section_num.split(".")[0])
                else:
                    # section is already a high-level section
                    high_level_section = section_num
                
                # if potential new high-level section
                if high_level_section != dictionary_high_level_section:
                    # skip if potential new high-level section is more than 2 greater than current section, or less than current section
                    # the +2 is in case a section is missed. trade-off between recovering from missed sections and adding false sections
                    if high_level_section > dictionary_high_level_section + 2 or high_level_section < dictionary_high_level_section:
                        continue
                    
                    # if this is the first section
                    if dictionary_high_level_section == 0:
                        # create the section dictionary
                        paper_dict["sections"] = dict()
                    # for all other sections, create dictionary with key as the previous section number
                    # value is a dict with name as the previous section name and text as the section string (previous section text)
                    else:
                        # place final paragraph list into previous subsection in section dict
                        current_section_dict[dictionary_low_level_section] = {
                            "name": current_low_level_section_name,
                            "page": current_low_level_page,
                            "paragraphs": section_paragraph_list
                        }
                        # place high-level section dict into paper dict
                        paper_dict["sections"][dictionary_high_level_section] = {
                            "name": current_high_level_section_name,
                            "page": current_high_level_page,
                            "subsections": current_section_dict
                        }
                        # reset paragraph list
                        section_paragraph_list = []
                    # set current section names as the section text
                    current_high_level_section_name = true_text
                    current_low_level_section_name = true_text
                    # set high- and low-level sections for the dictionary
                    dictionary_high_level_section = high_level_section
                    dictionary_low_level_section = str(high_level_section)
                    # set high- and low-level pages for the dictionary
                    current_high_level_page = p_pagenum + 1
                    current_low_level_page = p_pagenum + 1
                    # create dict for the new section
                    current_section_dict = dict()
                else:
                    # place paragraph list into previous subsection in section dict
                    current_section_dict[dictionary_low_level_section] = {
                        "name": current_low_level_section_name,
                        "page": current_low_level_page,
                        "paragraphs": section_paragraph_list
                    }
                    current_low_level_section_name = true_text
                    dictionary_low_level_section = section_num
                    current_low_level_page = p_pagenum + 1
                    section_paragraph_list = []
            
            # if paragraph
            if p_type == "paragraph":
                # last character of paragraph
                last_char = true_text[-1]
                
                if last_char == "-":
                    # strip last character if it is a dash
                    hanging_paragraph += true_text.rstrip("-")
                else:
                    # add a space otherwise
                    hanging_paragraph += true_text + " "
                
                # if the last character is a period, question mark, or exclamation point
                if last_char in ending_characters:
                    # add hanging paragraph to section string
                    section_paragraph_list.append(hanging_paragraph.strip())
                    # reset hanging paragraph
                    hanging_paragraph = ""
    
    # place final paragraph list into final section dict
    current_section_dict[dictionary_low_level_section] = {
        "name": current_low_level_section_name,
        "page": current_low_level_page,
        "paragraphs": section_paragraph_list
    }
    # place final section dict into paper dict
    paper_dict["sections"][dictionary_high_level_section] = {
        "name": current_high_level_section_name,
        "page": current_high_level_page,
        "subsections": current_section_dict
    }
    
    return paper_dict


def transform_paper_dict_into_paragraph_list(paper_dict: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    """Transforms a paper dictionary into a list of paragraphs with some additional information.

    Args:
        paper_dict (Dict[str, Any]): Dictionary of paper information from 'extract_paper_from_pdf'.

    Returns:
        List[Tuple[str, str, str, str]]: List of (subsection name, subsection number, paragraph number, paragraph)
    """
    paragraph_list = []
    for key, values in paper_dict.items():
        if isinstance(values, list):
            for paragraph_num, paragraph in enumerate(values):
                paragraph_list.append((key, "1", paragraph_num + 1, paragraph))
        else:
            for section_num, section_dict in values.items():
                if section_dict["name"].lower() == "references":
                    break
                for subsection_num, subsection_dict in section_dict["subsections"].items():
                    for paragraph_num, paragraph in enumerate(subsection_dict["paragraphs"]):
                        paragraph_list.append((subsection_dict["name"], subsection_num, str(paragraph_num + 1), paragraph))
    
    return paragraph_list
