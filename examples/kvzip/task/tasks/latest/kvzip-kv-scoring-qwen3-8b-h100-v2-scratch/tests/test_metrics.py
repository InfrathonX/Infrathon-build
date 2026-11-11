from kvzip_eval.metrics import score_answer


def test_normalized_inclusion():
    assert score_answer("include_normalized", "It is in the West Archive, behind the clockmaker's map!",
                        ["In the west archive behind the clockmaker's map."]) == 1.0
    assert score_answer("include_normalized", "The code is RQ-731.", ["RQ-731."]) == 1.0
    assert score_answer("include_normalized", "Paris", ["Berlin", "Paris"]) == 1.0
    assert score_answer("include_normalized", "", ["x"]) == 0.0


def test_gsm_numeric_inclusion_is_number_aware():
    ref = ["She has 3 + 4 = <<3+4=7>>7 left.\n#### 1,250"]
    assert score_answer("numeric_inclusion", "So the answer is 1250 dollars.", ref) == 1.0
    assert score_answer("numeric_inclusion", "That makes $1,250.00 in total.", ref) == 1.0
    assert score_answer("numeric_inclusion", "It is 12500.", ref) == 0.0
    assert score_answer("numeric_inclusion", "It is 125.", ["#### 5"]) == 0.0


def test_ruler_string_match():
    assert score_answer("ruler_all", " 123 and 456", ["123", "456", "789"]) == 2 / 3
    assert score_answer("ruler_any", " france", ["France", "Paris"]) == 1.0
    assert score_answer("ruler_any", " 5 million", ["five million"]) == 0.0


def test_exact_match_and_option_letter():
    assert score_answer("exact_match", " 891 ", ["891"]) == 1.0
    assert score_answer("exact_match", "891.", ["891"]) == 0.0
    assert score_answer("option_letter", "(D) Named Entities", ["(D) Named Entities"]) == 1.0
    assert score_answer("option_letter", "(C) Facts", ["(D) Named Entities"]) == 0.0
    assert score_answer("option_letter", "The error is about named entities.", ["(D) Named Entities"]) == 1.0


def test_rouge_l_and_repoqa():
    assert score_answer("rouge_l", "the cat sat on the mat", ["the cat sat on the mat"]) > 0.99
    fn = "public int add(int a, int b) {\n    return a + b;\n}"
    assert score_answer("repoqa_bleu", f"```java\n{fn}\n```", [fn], {"lang": "java"}) == 1.0
    assert score_answer("repoqa_bleu", "```java\nvoid other() {}\n```", [fn], {"lang": "java"}) == 0.0
    assert score_answer("repoqa_bleu", "   ", [fn], {"lang": "java"}) == 0.0
