import markdown
import codecs

def generate_docs():
    with codecs.open('documentation.md', mode='r', encoding='utf-8') as f:
        md_text = f.read()

    with codecs.open('styles.css', mode='r', encoding='utf-8') as f:
        css = f.read()

    html = markdown.markdown(md_text, extensions=['extra', 'codehilite'])

    final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DeforestAI - Documentation</title>
    <style>
        {css}
    </style>
</head>
<body>
    <div class="container">
        {html}
    </div>
</body>
</html>"""

    with codecs.open('documentation.html', mode='w', encoding='utf-8') as f:
        f.write(final_html)
    print("Generated documentation.html successfully!")

if __name__ == "__main__":
    generate_docs()
