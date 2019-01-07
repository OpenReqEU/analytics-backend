import re


def cleaning_text(text):
    # u00E0\u00E1\u00E8\u00E9\u00EC\u00ED\s\u00F2\u00F3\u00F9\u00FA\
    # SAVE: letters, numbers and these symbles: -\/.
    save_ch = r'[^0-9A-z\-\.\\\/]'
    text = re.sub(pattern=save_ch, repl=r' ', string=text, flags=re.U)

    # REMOVE:
    # tab, newline and vertical-tab, multiple dots and dot+space
    text = re.sub(pattern=r'\n', repl=r' ', string=text, flags=re.U)
    text = re.sub(pattern=r'\t', repl=r' ', string=text, flags=re.U)
    text = re.sub(pattern=r'\v', repl=r' ', string=text, flags=re.U)
    text = re.sub(pattern=r'[\.]+', repl=r'\.', string=text, flags=re.U)
    text = re.sub(pattern=r'\.\s', repl=r' ', string=text, flags=re.U)

    # remove multiple spaces
    text = re.sub(pattern='[ ]+', repl=r' ', string=text)

    return text