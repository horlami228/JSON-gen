#!/usr/bin/env python3

import subprocess
import os
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_JUSTIFY
import matplotlib
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import io
from bs4 import BeautifulSoup
import re
import base64
import json
import requests
from reportlab.lib import colors
import matplotlib as mpl
import html


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def load_json_data(file_path: str) -> list:
    """
    Load JSON data from the given file path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: Parsed JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# with open('Neet_question_json_1.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} questions from raw_input.json")


pdfmetrics.registerFont(TTFont('NotoSansDevanagari', './NotoSansDevanagari-Regular.ttf'))

# Register Gujarati font
pdfmetrics.registerFont(TTFont('NotoSansGujarati', './NotoSansGujarati-Regular.ttf'))

pdfmetrics.registerFont(TTFont('NotoSans', './NotoSans-Regular.ttf'))

pdfmetrics.registerFont(TTFont('DejaVuSans', './DejaVuSans.ttf'))

# Set non-GUI backend for matplotlib
matplotlib.use('Agg')


def clean_latex_with_mathrm(latex_string):
    symbol_map = {
        r'\\times': '×',
        r'\\div': '÷',
        r'\\pm': '±',
        r'\\cdot': '·',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\infty': '∞',
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        r'\\Rightarrow': '⇒',
        r'\\degree': '°',
        r'\\pi': 'π',
        r'\\%': '%',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\epsilon': 'ε',
        r'\\zeta': 'ζ',
        r'\\eta': 'η',
        r'\\theta': 'θ',
        r'\\iota': 'ι',
        r'\\lt': '<',
        r'\\gt': '>',
        r'\\le': '≤',
        r'\\ge': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
        r'\\Omega': 'Ω', 
    }
    
    latex_string = html.unescape(latex_string)

    # This pattern matches LaTeX expressions that are inside \( ... \) or $...$
    def clean_match(match):
        expr = match.group(1)    # Fix \left\llcorner^0 -> \mathrm{L}^0
        expr = re.sub(r'^\\mathrm\{([A-Za-z0-9]+)\}$', r'\\\1', expr)

        expr = re.sub(r'\\left\\llcorner\^0', r'\\mathrm{L}^0', expr)

        # Fix \right]\right. -> \right]
        expr = re.sub(r'(\\right\])\\right\.', r'\1', expr)

        # Optionally remove any trailing \right. (if it appears alone)
        expr = re.sub(r'\\right\.$', '', expr)


        if any(tag in expr for tag in ['\\mathrm{', '\\text', '\\frac', '^',
                                        '\\stackrel', '\\underset', '\\left', '\\right', '\\circ', '\\sqrt', '\\textit', '_', 'Ne', 'F']):
            return f'\\({expr.strip()}\\)'

        else:
            for latex_sym, unicode_sym in symbol_map.items():
                expr = re.sub(latex_sym, unicode_sym, expr)
            expr = re.sub(r'\\', '', expr)
            # return expr[1:-1] + ' ' # Remove delimiters a
            return expr.strip() + ' ' # Return cleaned expression without delimiters

    # Remove \( ... \) expressions that don't contain \mathrm{}
    latex_string = re.sub(r'\\\((.*?)\\\)', clean_match, latex_string)

    # Remove $...$ expressions that don't contain \mathrm{}
    latex_string = re.sub(r'\$(.*?)\$', clean_match, latex_string)
    return latex_string

def split_text_and_latex(text):
    """
    Splits text into a list of (type, content), where type is 'text' or 'latex'.
    It preserves entire <table>...</table> blocks as 'text' parts and avoids splitting inside them.
    """

   
    # First extract all <table>...</table> blocks
    table_pattern = re.compile(r'(<table.*?>.*?</table>)', re.DOTALL | re.IGNORECASE)
    tables = []
    placeholders = []

    def table_replacer(match):
        tables.append(match.group(0))
        placeholder = f"[[TABLE_{len(tables)-1}]]"
        placeholders.append(placeholder)
        return placeholder

    text_with_placeholders = table_pattern.sub(table_replacer, text)

    # Now split LaTeX in the remaining text (outside tables)
    latex_pattern = re.compile(r'(\\\(.+?\\\))')
    parts = []
    last_end = 0
    for match in latex_pattern.finditer(text_with_placeholders):
        start, end = match.span()
        if start > last_end:
            parts.append(('text', text_with_placeholders[last_end:start]))
        latex_content = match.group()
        
        # Fix LaTeX if necessary
        fixed_latex = clean_latex_with_mathrm(latex_content)

        # Check if it still looks like LaTeX (wrapped in \( ... \))
        if fixed_latex.strip().startswith(r'\(') and fixed_latex.strip().endswith(r'\)'):
            parts.append(('latex', fixed_latex))
        else:
            parts.append(('text', fixed_latex))
        
        last_end = end
    if last_end < len(text_with_placeholders):
        parts.append(('text', text_with_placeholders[last_end:]))

    # Replace placeholders back with their actual table content
    final_parts = []
    for part_type, part_content in parts:
        if part_type == 'text':
            # Look for table placeholders
            for idx, placeholder in enumerate(placeholders):
                if placeholder in part_content:
                    # Split around the placeholder
                    before, _, after = part_content.partition(placeholder)
                    if before:
                        final_parts.append(('text', before))
                    final_parts.append(('text', tables[idx]))  # Add back the full table
                    part_content = after  # keep looking in the remaining text
            if part_content:  # add any leftover text
                final_parts.append(('text', part_content))
        else:
            final_parts.append((part_type, part_content))

    return final_parts


def wrap_multilang_text(text):
    lines = text.splitlines(keepends=True)  # Keep \n at the end
    wrapped_lines = []
    for line in lines:
        # Wrap Devanagari
        line = re.sub(r'([\u0900-\u097F]+)', r'<font name="NotoSansDevanagari">\1</font>', line)
        # Wrap Gujarati
        line = re.sub(r'([\u0A80-\u0AFF]+)', r'<font name="NotoSansGujarati">\1</font>', line)
        wrapped_lines.append(line)
        text = "".join(wrapped_lines)
    return text

def render_latex_to_png(latex_string, mcq_style):
    """Render LaTeX string to PNG image, scaled to match font size (~12pt)."""
    try:
        # Convert MathJax \\( ... \\) to LaTeX $ ... $
        latex_string = latex_string.replace('\\(', '$').replace('\\)', '$')

        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        fig.patch.set_visible(False)
        ax.axis('off')

        # Use a smaller fontsize to match typical 12pt paragraph font
        ax.text(0, 0, latex_string, fontsize=10, va='center', ha='center')
     


        # Save to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=500, bbox_inches='tight', pad_inches=0.0, transparent=True)
        plt.close(fig)
        buf.seek(0)

        # Use PIL to resize to match ~12pt (0.17 inch) height
        pil_img = PILImage.open(buf)
        img_width_px, img_height_px = pil_img.size
        target_height_in = 8 / 72  # 12pt = 12/72 inch ≈ 0.17 inch
        dpi = 400
        new_height_px = int(target_height_in * dpi)
        scaling_factor = new_height_px / img_height_px
        new_width_px = int(img_width_px * scaling_factor)

        pil_img_resized = pil_img.resize((new_width_px, new_height_px), PILImage.LANCZOS)

        img_buf = io.BytesIO()
        pil_img_resized.save(img_buf, format='PNG')
        img_buf.seek(0)
        return img_buf
    except Exception as e:
        # print(f"Error rendering LaTeX: {latex_string} - {e}")
        return Paragraph(latex_string, mcq_style)  # Fallback to plain text
    finally:
        if fig is not None:
            plt.close(fig)




def clean_html(text):
    if not text:
        return ""

    text = text.replace("&nbsp;", " ")
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    text = text.replace("</p>", "\n").replace("<p>", "\n").replace("<p/>", "\n")
    text = text.replace("<div>", "\n").replace("</div>", "\n").replace("<div/>", "\n")


    # Extract tables
    table_pattern = re.compile(r'(<table.*?>.*?</table>)', re.DOTALL | re.IGNORECASE)
    tables = []
    def table_replacer(match):
        tables.append(match.group(0))
        return f"[[TABLE_{len(tables)-1}]]"

    text_with_placeholders = table_pattern.sub(table_replacer, text)

    # Clean outside tables
    # text_with_placeholders = re.sub(r'<p[^>]*>', '\n', text_with_placeholders)
    # text_with_placeholders = re.sub(r'</p>', '\n', text_with_placeholders)
    text_with_placeholders = re.sub(r'<(?!img\b|table\b|tr\b|th\b|td\b)[^>]+>', '', text_with_placeholders)
    # text_with_placeholders = re.sub(r'\s+', ' ', text_with_placeholders).strip()
    text_with_placeholders = re.sub(r'[ ]+', ' ', text_with_placeholders)
    text_with_placeholders = re.sub(r'\n+', '\n', text_with_placeholders)
    text_with_placeholders = text_with_placeholders.strip()


    # Put back the tables as they were (untouched)
    for idx, table in enumerate(tables):
        text_with_placeholders = text_with_placeholders.replace(f"[[TABLE_{idx}]]", table)

    return text_with_placeholders


def render_latex_content(content, style, doc_width):
    # print(f"Content: {content}")
    parts = split_text_and_latex(content)

    # send parts to json

    flowables = []
    inline_text = ""

    for part_type, part_content in parts:
        # print("DEBUG part_content:", repr(part_content))
        if part_type == 'latex':
            img_data = render_latex_to_png(part_content, style)
            if isinstance(img_data, Paragraph):
                inline_text += part_content
            else:
                b64_img = base64.b64encode(img_data.read()).decode('utf-8')
                img_tag = f'<img src="data:image/png;base64,{b64_img}" valign="bottom" height="9" width="30"/>'
                inline_text += f" {img_tag} "
        else:
            # Parse the HTML content
            # print(f"part content: {part_content}")
            soup = BeautifulSoup(part_content, 'html.parser')
            for elem in soup.contents:
                if elem.name == 'img':
                    # print("There is image here")
                    # Flush any inline text before this image
                    if inline_text.strip():
                        flowables.append(Paragraph(inline_text, style))
                        inline_text = ""

                    src = elem.get('src', '')
                    alt = elem.get('alt', '[Image not available]')
                    try:
                        response = requests.get(src, timeout=5)
                        if response.status_code == 200:
                            img_buffer = BytesIO(response.content)
                            pil_img = PILImage.open(img_buffer)
                            img_width, img_height = pil_img.size

                            # 🔧 NEW: Set max width to 60% of doc width
                            max_width = doc_width * 0.15
                            max_height = 3 * inch  # Optional: limit height to 3 inches

                            # Calculate scale to fit within both max width and max height (keep aspect ratio)
                            width_scale = max_width / img_width
                            height_scale = max_height / img_height
                            scale = min(1.0, width_scale, height_scale)  # don't upscale

                            img_width *= scale
                            img_height *= scale

                            flowables.append(Spacer(1, 0.3 * inch))

                            reportlab_image = Image(img_buffer, width=img_width, height=img_height)
                            flowables.append(reportlab_image)
                            flowables.append(Spacer(1, 0.1 * inch))
                        else:
                            flowables.append(Paragraph(alt or '[Image not available]', style))
                            flowables.append(Spacer(1, 0.1 * inch))
                    except Exception as e:
                        print(f"Error fetching image {src}: {e}")
                        flowables.append(Paragraph(alt or '[Image not available]', style))
                        flowables.append(Spacer(1, 0.1 * inch))
                
                elif elem.name == 'table':
                    # print("Found table tag")
                    if inline_text.strip():
                        flowables.append(Paragraph(inline_text, style))
                        inline_text = ""

                    rows = elem.find_all('tr')
                    # print(f"{len(rows)} rows found in table")
                    table_data = []
                    for row in rows:
                        cols = row.find_all(['td', 'th'])
                        # print("Columns in row:", len(cols))
                        row_data = []
                        for col in cols:
                            # Recursively process cell content for LaTeX, images, etc.
                            cell_content = col.decode_contents()
                            cell_flowables = render_latex_content(cell_content, style, doc_width / len(cols))

                            if not cell_flowables:
                                row_data.append(Paragraph(" ", style))
                           
                            # If multiple flowables, wrap in a nested table to keep them together
                            elif len(cell_flowables) == 1:
                                row_data.append(cell_flowables[0])
                            else:
                                # nested_table = Table([[f] for f in cell_flowables], colWidths=[doc_width / len(cols)])
                                # row_data.append(nested_table)
                                row_data.append(KeepTogether(cell_flowables))
                        table_data.append(row_data)
                    

                    total_cols = len(table_data[0]) if table_data and table_data[0] else 1
                    col_widths = [doc_width / total_cols] * total_cols
                    table_style = TableStyle([
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                        ('TOPPADDING', (0, 0), (-1, -1), 5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ])
                    tbl = Table(table_data, colWidths=col_widths)
                    tbl.canSplit = True
                    tbl.setStyle(table_style)
                    flowables.append(Spacer(1, 0.2 * inch))
                    flowables.append(tbl)
                    flowables.append(Spacer(1, 0.1 * inch))
                else:
                    # If it's not an <img> or <table>, just treat it as text
                    # inline_text += str(elem)
                    text_part = str(elem)
                    if text_part.strip():  # only add if it’s not pure empty
                     inline_text += text_part
    if inline_text.strip():
        try:
            flowables.append(Paragraph(inline_text.replace('\n', '<br/>'), style))
        except Exception as e:
            print(f"Skipping invalid paragraph: {repr(inline_text)} due to error: {e}")
    return flowables

def format_questions_flowable(questions_data, output_pdf):
    doc = SimpleDocTemplate(output_pdf, pagesize=A4,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontName='DejaVuSans',
        fontSize=10,
        leading=10,

    )

    hindi_style = ParagraphStyle(
    'Hindi',
    parent=styles['Normal'],
    alignment=TA_JUSTIFY,
    fontName='NotoSansDevanagari',
    fontSize=10,
    leading=12,
    wordWrap='CJK',

    )

    gujarati_style = ParagraphStyle(
    'Gujarati',
    parent=styles['Normal'],
    alignment=TA_JUSTIFY,
    fontName='NotoSansGujarati',
  
    leading=12,
    spaceBefore=0,
    leftIndent=0,
    spaceAfter=0,
    allowOrphans=0,
    wordWrap='CJK',
)

    mcq_style = ParagraphStyle(
        'MCQ',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        leading=12,
        spaceBefore=0,
        leftIndent=0,
        spaceAfter=0,
        allowOrphans=0,
        wordWrap='CJK',
    )

    story = []

    lan = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    }

    option_prefixes = {
    'en': ['(A)', '(B)', '(C)', '(D)'],
    'hi': ['(क)', '(ख)', '(ग)', '(घ)'],
    'gu': ['(અ)', '(બ)', '(ક)', '(ડ)'],
}

    for i, question_data in enumerate(questions_data):
        print("Question with id:", question_data.get("id"))
        question_number = i + 1
        
        # Check if the question is in English, Hindi or Gujarati

       # Check which language keys exist
        for lang_code, lang_name in lan.items():
            question_key = f"{lang_code}_question"
            
            if question_key in question_data:
                # print(f"Processing question {question_number} in {lang_name}")
                # Add section header (e.g., "Question 1 (English)")
                section_title = f"{lang_name}"
                section_header = Paragraph(section_title, styles['Normal'])
                story.append(section_header)

                # Get and clean the question text
                question_text = question_data.get(question_key, "").strip()
                # print(f"Original question text (Q{question_number}, {lang_name}): {question_text}")
                # print(f"Before cleaning: {question_text}")
                combined_question = clean_html(question_text)
                # print(f"from combined question after cleaning: {combined_question}")
                combined_question_wrapped = wrap_multilang_text(combined_question)
                # print("combined", combined_question_wrapped)
                question_para = render_latex_content(combined_question_wrapped, body_style, doc.width)


                # Multiple Choice Options (dynamic keys based on lang)

                prefixes = option_prefixes.get(lang_code, option_prefixes['en'])

                option_data = [
                    f"{prefixes[0]} {question_data.get(f'{lang_code}_mcq_1', '').replace(' ', '')}",
                    f"{prefixes[1]} {question_data.get(f'{lang_code}_mcq_2', '').replace(' ', '')}",
                    f"{prefixes[2]} {question_data.get(f'{lang_code}_mcq_3', '').replace(' ', '')}",
                    f"{prefixes[3]} {question_data.get(f'{lang_code}_mcq_4', '').replace(' ', '')}",
                ]


                # Render LaTeX for the options and create flowables
                option_flowables = []
                for option in option_data:
                    cleaned_option = clean_html(option)
                    wrapped_option = wrap_multilang_text(cleaned_option)
                    option_flowables.append(render_latex_content(wrapped_option, mcq_style, doc.width))
                options_table_data = [
                    [option_flowables[0]],
                    [option_flowables[1]],
                    [option_flowables[2]],
                    [option_flowables[3]],
                ]
                options_table = Table(
                    options_table_data,
                    colWidths=[doc.width - 0.1 * inch],
                    rowHeights=None
                )
                options_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 2),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))

                spacer_between_question_and_options = Spacer(1, 0.1 * inch)
                right_cell_content = [question_para, spacer_between_question_and_options, options_table]

                question_table_data = [[Paragraph(f"({question_number})", styles['Normal']), right_cell_content]]
                question_table = Table(
                    question_table_data,
                    colWidths=[0.5 * inch, doc.width - 0.5 * inch]
                )
                question_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ]))

                story.append(question_table)
                story.append(Spacer(1, 0.3 * inch))
                print("processed question number:", question_number)
    print("Finished processing all questions.")
    print("adding to the story")
    doc.build(story)
    return (f"{output_pdf}")
