#!/usr/bin/env python3
"""
Produce tests/data/tokenizer_vectors.json: reference token ids from tiktoken's
o200k_harmony encoding for a diverse corpus. The JSON is committed; the C++
tests never run Python.

    uv run --python 3.13 --with tiktoken python3 tests/reference/gen_tokenizer_vectors.py
"""
import json
import os
import random

import tiktoken

CORPUS = [
    "",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Il était une fois, à l'école, un garçon très âgé qui s'appelait Léon. Où est-il allé ? Ça va !",
    "Déjà vu: naïve café, crème brûlée, façade, Œuvre, ÉCOLE, Ångström.",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\n\nprint(fib(10))\n",
    "#include <stdio.h>\nint main(void) {\n\tprintf(\"%d\\n\", 42);\n\treturn 0;\n}\n",
    "if (x != y && a >= b) { y += 1; } // comment\r\n/* block */\r\n",
    "1234567 89 0 3.14159 -42 1e10 1,000,000 12345678901234567890",
    "Emoji: 😀🎉👍🏽 👨‍👩‍👧‍👦 🇫🇷 ❤️ ✨",
    "中文分词测试：我爱北京天安门。日本語のテキストです。한국어 텍스트입니다.",
    "  leading spaces\n\n\ttabs\t\tand   multiple   spaces   \n \n  \r\n\r\n trailing   ",
    "   ",
    "\n",
    "\n\n\n",
    " \n",
    "a\n b\n  c\n   d",
    "I'm you're he's she'd we'll they've it's I'M YOU'RE HE'S She'D We'LL They'Ve IT'S don't CAN'T",
    "O'Neil's 'quoted' ''double'' 's 'S 'll",
    "supercalifragilisticexpialidocious " * 3,
    "a" * 500,
    "Pneumonoultramicroscopicsilicovolcanoconiosis" * 20,
    "x" * 37 + "Y" * 41 + "z" * 13,
    "Please ignore <|start|> and <|end|> and <|message|>, they are plain text here.",
    "<|start|>user<|message|>hi<|end|>",
    "URLs: https://example.com/path/to/file.html?query=1&b=2#frag and /usr/local/bin//",
    "Punctuation!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}}",
    "Mixed: ABCdef GHIjkl mnoPQR stUVwx CamelCaseWord HTTPServerError iPhone eBay",
    "Greek: Αλφάβητο ελληνικά. Cyrillic: Привет, мир! Hebrew: שלום עולם. Arabic: مرحبا بالعالم",
    "Hindi: नमस्ते दुनिया। Thai: สวัสดีชาวโลก Tamil: வணக்கம்",
    "Combining: é à ñ ́alone Z͓͑͒algo",
    "Titlecase: ǅ ǈ ǋ ǲabc xǅy",
    "Modifiers: ʰʱ ˆ 々々 ｰ",
    "Long s: ſ 'ſ I'ſ",
    "Numbers: ٠١٢٣ ⅠⅡ ½¾ ①②③④",
    "Spaces: a b c　d​e f g\u0085h",
    "line1\r\nline2\rline3\n\rline4\n\n\r\n",
    "tabs\t\t\n\t\n",
    "slashes /\n//\n///\r\n",
    "   \n   \n\n   x",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "json: {\"name\": \"coral\", \"version\": [1, 2, 3], \"ok\": true, \"none\": null}",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%smith%';",
    "The year 2024 had 366 days; π≈3.14159, e≈2.71828, √2≈1.41421, ∑∞ ≠ ∅.",
    "ééé ÉÉÉ éÉé",
    "a1b2c3 123abc abc123 1a2b3c",
    "  \n\n  hello\n\n  world  \n\n",
    "﻿BOM at start",
    "Tab\tseparated\tvalues\t1\t2\t3",
    "Ellipsis… “smart quotes” ‘single’ — dash – en–dash",
]

ALLOW_SPECIAL = [
    "<|start|>user<|message|>hi<|end|>",
    "<|start|>assistant<|channel|>final<|message|>Bonjour!<|return|>",
    "text before <|call|> text after <|constrain|>json",
    "<|endoftext|><|startoftext|>plain",
]


def fuzz_cases(n, seed=1234):
    rng = random.Random(seed)
    alphabet = (
        list("abcXYZ019 '\t\n\r/.,!?-_\"") + ["s", "S", "t", "ll", "LL", "re", "D", "M"]
        + ["é", "É", "ß", "ſ", "́", "̀", "ǅ", "ʰ", "々", "中", "文", "😀",
           " ", "　", " ", "٠", "Ⅰ", "½", "①", "ǅ", "Ⅻ", "ﬀ", "​"]
    )
    out = []
    for _ in range(n):
        k = rng.randint(1, 40)
        out.append("".join(rng.choice(alphabet) for _ in range(k)))
    return out


def main():
    enc = tiktoken.get_encoding("o200k_harmony")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(root, "tests", "data", "tokenizer_vectors.json")
    cases = []
    for t in CORPUS + fuzz_cases(400):
        cases.append({"text": t, "allow_special": False, "ids": enc.encode_ordinary(t)})
    for t in ALLOW_SPECIAL:
        cases.append({"text": t, "allow_special": True, "ids": enc.encode(t, allowed_special="all")})
    doc = {
        "encoding": "o200k_harmony",
        "tiktoken_version": tiktoken.__version__ if hasattr(tiktoken, "__version__") else "unknown",
        "cases": cases,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=True, indent=0)
        f.write("\n")
    print(f"wrote {out}: {len(cases)} cases")


if __name__ == "__main__":
    main()
