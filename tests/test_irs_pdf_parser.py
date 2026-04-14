from src.ingestion import irs_pdf_parser as parser


def test_extract_sections_splits_on_headings():
    lines = [
        "General Instructions:",
        "You should keep detailed records for all deductions and retain supporting receipts for each itemized amount throughout the tax year.",
        "You can use this form to report additional data and attach schedules where required by the instructions in this section.",
        "Specific Instructions:",
        "Line 1. Enter your wages from Form W-2 and include all taxable compensation from employment and related services.",
        "Line 2. Enter taxable interest and include statements from financial institutions to support the reported values.",
    ]

    sections = parser._extract_sections(lines, page_num=1)

    assert len(sections) >= 2
    assert sections[0][0] == "Page 1"
    assert "records" in sections[0][1].lower()
    assert sections[1][0] == "Specific Instructions:"


def test_split_long_text_returns_multiple_windows():
    text = " ".join(
        [
            "Sentence one explains a filing rule.",
            "Sentence two provides an exception.",
            "Sentence three provides another exception.",
            "Sentence four includes details.",
            "Sentence five includes examples.",
            "Sentence six has additional notes.",
            "Sentence seven has edge conditions.",
            "Sentence eight has references.",
        ]
    )

    windows = parser._split_long_text(text, max_chars=120, overlap_chars=30)

    assert len(windows) > 1
    assert all(window.strip() for window in windows)
