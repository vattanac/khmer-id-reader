# khmer-id-reader

Free, open-source Cambodian national ID card OCR library.

Reads all printed fields, verifies MRZ check digits, and cross-checks the Khmer name against the Latin printing.

## Install

```bash
pip install khmer-id-reader
```

## Usage

### Python

```python
from khmer_id_reader import read_card

result = read_card("my_id_photo.jpg")
print(result.name_latin)   # SOK DARA
print(result.id_number)    # 123456789 (01)
print(result.mrz_verified) # True
print(result.to_json())    # full JSON output
```

### Command line

```bash
khmer-id-reader my_id_photo.jpg
```

## What it reads

- Name (Khmer + Latin, cross-verified)
- Date of birth, sex, height
- Place of birth
- Address (verified against national gazetteer)
- ID number
- Validity period
- Nationality
- Full MRZ (3 lines, check-digit verified)

## Models

Models are auto-downloaded from [HuggingFace](https://huggingface.co/vattanac/khmer-ocr-crnn-ctc) on first use (~50 MB).

## License

Apache 2.0

## Author

Sim Vattanac
