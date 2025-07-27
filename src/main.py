import json
import os
from round1a.pdf_structure_extractor import PDFStructureExtractor

def main():
    """
    Main function to run only Round 1A (structure extraction) on all PDFs in the /input directory.
    """
    input_dir = "input"
    output_dir = "output"

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    extractor = PDFStructureExtractor()

    print(f"Running Round 1A: Structure Extraction for {len(pdf_files)} file(s)")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        print(f"  Processing: {pdf_file}")

        try:
            result = extractor.extract_structure(pdf_path)

            output_filename = f"{os.path.splitext(pdf_file)[0]}_structure.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"    ✓ Saved to {output_path}")

        except Exception as e:
            print(f"    ✗ Failed to process {pdf_file}: {str(e)}")

    print("✅ Structure extraction completed for all files.")

if __name__ == "__main__":
    # Ensure __init__.py files exist for importability
    for d in ['src', 'src/common', 'src/round1a']:
        init_file = os.path.join(d, '__init__.py')
        if not os.path.exists(init_file):
            open(init_file, 'a').close()

    main()
